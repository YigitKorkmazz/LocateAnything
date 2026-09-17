"""Differentiable current-policy replay with cached visual features.

Wraps the existing, unmodified `rl.ntp_rl.NTPRolloutReplayer` /
`rl.hybrid_rl.HybridRolloutReplayer` (teacher-forced trajectory scoring) so
it consumes a precomputed group-local visual-feature cache instead of
recomputing `extract_feature` + `mlp1` on every call (see
`generation.cache_group_visual_features` for the equivalence argument), and
provides a selective functional-KV-layer-checkpointing context manager for
the backward pass over all 36 decoder layers.

Checkpointing all 36 layers is required to fit one live-cached backward pass
in 24GiB (see DESIGN.md Sec 1.3 / Sec 11); checkpointing *every* layer costs
a full recompute forward per checkpointed layer during backward, which the
prior probes measured as the single largest cost in the whole pipeline
(~51% of group time). This module exposes the checkpointed-layer *set* as a
parameter so Phase 6 benchmarking can measure the actual VRAM/throughput
trade-off on this exact 256-token config rather than assume the historical
512-token finding transfers unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, Set, Tuple

import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

from rl.grpo import grpo_clipped_loss
from rl.ntp_rl import NTPRolloutReplayer
from rl.replay_memory import _decoder_layers, assert_model_eval_for_replay


class CachedVisualNTPRolloutReplayer(NTPRolloutReplayer):
    """Teacher-forced NTP replay that reuses a group-local visual cache."""

    def __init__(self, *args: Any, cached_visual_features: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cached_visual_features = cached_visual_features

    def _projector_visual_features(self, pixel_values: Any, image_grid_hws: Any) -> Tuple[Any, Any]:
        del pixel_values
        return self.cached_visual_features, image_grid_hws


def _assign_dynamic_layer(layer, key: torch.Tensor, value: torch.Tensor) -> None:
    layer.keys = key
    layer.values = value
    layer.dtype = key.dtype
    layer.device = key.device
    layer.is_initialized = True


def _ensure_layer(cache: DynamicCache, layer_index: int):
    while len(cache.layers) <= layer_index:
        if cache.layer_class_to_replicate is None:
            raise RuntimeError("DynamicCache cannot create the requested layer")
        cache.layers.append(cache.layer_class_to_replicate())
    return cache.layers[layer_index]


def _prior_layer_kv(cache: DynamicCache, layer_index: int):
    if len(cache.layers) <= layer_index:
        return None
    layer = cache.layers[layer_index]
    if not bool(getattr(layer, "is_initialized", False)) or layer.get_seq_length() == 0:
        return None
    return layer.keys, layer.values


@contextmanager
def selective_kv_layer_checkpointing(
    model: Any,
    *,
    checkpoint_layers: Optional[Set[int]] = None,
) -> Iterator[Dict[str, Any]]:
    """Checkpoint only `checkpoint_layers` (default: all layers).

    Layers not in `checkpoint_layers` run as an ordinary live-gradient
    forward (no recompute during backward -> faster, more VRAM). Layers in
    `checkpoint_layers` recompute their forward during backward against an
    immutable K/V snapshot and mutate the live outer cache exactly once,
    identically to `rl.exact_kv_checkpoint.functional_kv_layer_checkpointing`
    (this is that module's technique, generalized to a layer subset).
    """
    report: Dict[str, Any] = {
        "use_reentrant": False,
        "functional_kv": True,
        "checkpoint_calls_by_layer": {},
    }
    assert_model_eval_for_replay(model)
    layers = _decoder_layers(model)
    n_layers = len(layers)
    layer_set = set(range(n_layers)) if checkpoint_layers is None else set(checkpoint_layers)
    report["wrapped_layer_count"] = len(layer_set)
    report["n_layers"] = n_layers
    originals = []

    def wrap(layer: torch.nn.Module, layer_index: int):
        original = layer.forward
        cache_config = getattr(layer.self_attn, "config", None)
        if cache_config is None:
            raise RuntimeError(f"decoder layer {layer_index} has no attention config")

        def forward_with_checkpoint(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            if kwargs:
                raise RuntimeError(f"checkpoint wrapper received unsupported layer kwargs: {sorted(kwargs)}")
            if not bool(use_cache):
                raise RuntimeError("selective KV checkpoint requires use_cache=True")
            if bool(output_attentions):
                raise RuntimeError("selective KV checkpoint requires output_attentions=False")
            if not isinstance(past_key_value, DynamicCache):
                raise RuntimeError("selective KV checkpoint requires transformers.DynamicCache")
            if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                raise RuntimeError("attention mask must be a tensor or None")
            if not isinstance(position_ids, torch.Tensor):
                raise RuntimeError("selective KV checkpoint requires explicit position IDs")

            prior = _prior_layer_kv(past_key_value, layer_index)
            mask_is_none = attention_mask is None

            def run_layer(hidden, positions, *remaining_tensors):
                mask = None if mask_is_none else remaining_tensors[0]
                prior_tensors = remaining_tensors if mask_is_none else remaining_tensors[1:]
                local_cache = DynamicCache(config=cache_config)
                if prior_tensors:
                    if len(prior_tensors) != 2:
                        raise RuntimeError("expected explicit prior key and value tensors")
                    local_layer = _ensure_layer(local_cache, layer_index)
                    _assign_dynamic_layer(local_layer, prior_tensors[0], prior_tensors[1])
                outputs = original(
                    hidden,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_value=local_cache,
                    output_attentions=False,
                    use_cache=True,
                )
                local_layer = _ensure_layer(local_cache, layer_index)
                return outputs[0], local_layer.keys, local_layer.values

            checkpoint_inputs = (hidden_states, position_ids)
            if attention_mask is not None:
                checkpoint_inputs += (attention_mask,)
            if prior is not None:
                checkpoint_inputs += prior
            next_hidden, next_key, next_value = checkpoint(
                run_layer,
                *checkpoint_inputs,
                use_reentrant=False,
                preserve_rng_state=True,
                determinism_check="default",
            )
            outer_layer = _ensure_layer(past_key_value, layer_index)
            _assign_dynamic_layer(outer_layer, next_key, next_value)
            calls = report["checkpoint_calls_by_layer"]
            calls[layer_index] = calls.get(layer_index, 0) + 1
            return next_hidden, past_key_value

        return forward_with_checkpoint

    try:
        for index, layer in enumerate(layers):
            if index in layer_set:
                originals.append((layer, layer.forward))
                layer.forward = wrap(layer, index)  # type: ignore[method-assign]
        yield report
    finally:
        for layer, original in originals:
            layer.forward = original  # type: ignore[method-assign]


def reference_logprob_blocks(
    replayer: CachedVisualNTPRolloutReplayer,
    decoder_kwargs: Dict[str, Any],
    trace: Any,
) -> list:
    """Per-token log-probs under the frozen reference policy pi_ref.

    pi_ref = this same model with its LoRA adapter disabled (PEFT's
    `disable_adapter()` context), i.e. the base pretrained weights this run
    started from -- not a second model copy, so this costs one extra no_grad
    forward pass per rollout and no extra VRAM for weights. Teacher-forced
    over the identical `trace` as the current-policy pass, so the returned
    list lines up token-for-token with `block_logps` from `replayer.score`.
    No checkpointing is needed here (no backward pass over this branch).
    """
    peft_model = replayer.model.language_model
    if not hasattr(peft_model, "disable_adapter"):
        raise RuntimeError("reference-policy KL requires a PEFT-wrapped language_model")
    with torch.no_grad(), peft_model.disable_adapter():
        _, ref_block_logps = replayer.score(
            trace, use_cache=True, legacy_nocache_masks=False, **decoder_kwargs
        )
    return [t.detach() for t in ref_block_logps]


def token_kl_k3(
    current_block_logps: Sequence[torch.Tensor],
    reference_block_logps: Sequence[torch.Tensor],
    *,
    max_per_token_kl: float = 10.0,
) -> torch.Tensor:
    """Unbiased per-token KL estimator D_KL[pi_theta || pi_ref] (the "k3"
    estimator: http://joschu.net/blog/kl-approx.html), the same one
    DeepSeekMath/DeepSeek-R1 GRPO and MedGround-R1 (Eq. 3-4) use:

        KL_i = exp(logp_ref_i - logp_cur_i) - (logp_ref_i - logp_cur_i) - 1

    averaged over the completion's tokens. Always >= 0 (e^x - x - 1 >= 0
    for all real x); differentiable only through `current_block_logps`
    since the reference term is detached by construction.

    A single token where pi_ref is confident but the current policy has
    diverged heavily (logp_ref - logp_cur large and positive) makes exp(.)
    overflow -- observed non-finite in a real 4-GPU run after just one
    optimizer step, not just a theoretical concern. `nan_to_num` + clamp
    bound any one outlier token's contribution to `max_per_token_kl` instead
    of poisoning the whole completion's mean (and hence the window's loss)
    with inf/nan; `torch.clamp`'s (and `nan_to_num`'s) gradient is 0 past
    the bound, so a clamped token simply contributes no KL gradient that
    step rather than corrupting the others'. The bound is generous relative
    to normal per-token KL (observed 7e-5 to 2e-4 after a small synthetic
    step in test_kl_penalty.py), so it only engages on genuine outliers.
    """
    if len(current_block_logps) != len(reference_block_logps):
        raise RuntimeError("current/reference block-logprob count mismatch")
    if not current_block_logps:
        raise RuntimeError("cannot compute KL over an empty completion")
    diffs = [
        ref.to(cur.device) - cur
        for cur, ref in zip(current_block_logps, reference_block_logps)
    ]
    per_token_kl = []
    for d in diffs:
        kl = torch.exp(d) - d - 1.0
        kl = torch.nan_to_num(kl, nan=0.0, posinf=max_per_token_kl, neginf=0.0)
        kl = torch.clamp(kl, min=0.0, max=max_per_token_kl)
        per_token_kl.append(kl)
    return torch.stack(per_token_kl).mean()


def score_and_backward(
    replayer: CachedVisualNTPRolloutReplayer,
    decoder_kwargs: Dict[str, Any],
    trace: Any,
    *,
    old_logp: torch.Tensor,
    advantage: torch.Tensor,
    loss_divisor: float,
    clip_epsilon: float,
    checkpoint_layers: Optional[Set[int]] = None,
    kl_beta: float = 0.0,
) -> Dict[str, Any]:
    """Differentiable current-policy score, GRPO(+KL) loss, one `.backward()`.

    `loss_divisor` folds together both G (rollouts per group) and the number
    of prompt groups accumulated into one optimizer step, so callers can
    `.backward()` sequentially across an arbitrary accumulation window
    without ever calling `zero_grad`/`step` inside this function.

    `kl_beta > 0.0` adds `kl_beta * token_kl_k3(...)` to the per-rollout loss
    before dividing by `loss_divisor`, matching MedGround-R1 Eq. 3-4 (the
    `-beta*D_KL` term inside the objective being maximized becomes
    `+beta*KL` once the sign flips to a loss being minimized). Costs one
    extra no_grad reference-policy forward pass per rollout; `kl_beta=0.0`
    (default) skips that pass entirely and reproduces the prior behavior
    exactly.
    """
    model = replayer.model
    if model.training:
        raise RuntimeError("current-policy replay requires model.eval()")

    reference_block_logps = None
    if kl_beta > 0.0:
        reference_block_logps = reference_logprob_blocks(replayer, decoder_kwargs, trace)

    kl_term: Optional[torch.Tensor] = None
    with selective_kv_layer_checkpointing(model, checkpoint_layers=checkpoint_layers) as report:
        current_logp, block_logps = replayer.score(trace, use_cache=True, legacy_nocache_masks=False, **decoder_kwargs)
        if not current_logp.requires_grad:
            raise RuntimeError("current-policy logprob was unexpectedly detached")
        if not bool(torch.isfinite(current_logp).all()):
            raise RuntimeError("non-finite current-policy replay logprob")
        advantage_tensor = advantage.detach().reshape(1).to(device=current_logp.device, dtype=torch.float32)
        policy_loss = grpo_clipped_loss(
            current_logp.float().reshape(1),
            old_logp.detach().float().reshape(1).to(current_logp.device),
            advantage_tensor,
            clip_epsilon=clip_epsilon,
        )
        if kl_beta > 0.0:
            kl_term = token_kl_k3(block_logps, reference_block_logps)
            if not bool(torch.isfinite(kl_term).all()):
                raise RuntimeError("non-finite KL penalty")
            loss = (policy_loss + float(kl_beta) * kl_term) / float(loss_divisor)
        else:
            loss = policy_loss / float(loss_divisor)
        loss.backward()
    ratio = float(torch.exp(current_logp.detach().float() - old_logp.detach().float().to(current_logp.device)))
    result = {
        "current_log_prob": float(current_logp.detach().float().cpu()),
        "old_log_prob": float(old_logp.detach().float().cpu()),
        "ppo_ratio": ratio,
        "loss": float(loss.detach().float().cpu()),
        "checkpoint_report": report,
    }
    if kl_term is not None:
        result["kl_penalty"] = float(kl_term.detach().float().cpu())
    return result
