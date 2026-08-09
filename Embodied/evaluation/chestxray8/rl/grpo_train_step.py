"""Group GRPO update: scientific live production-cached trajectory gradients."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence

import torch

from rl.cuda_memory import (
    record_stage_peak,
    release_cuda_temporaries,
    reset_peak_memory,
)
from rl.exact_backends import (
    LIVE_PRODUCTION_CACHED,
    SURROGATE_NOT_THESIS_DEFAULT_MSG,
    TRUNCATED_BPTT_SURROGATE,
    exact_backend_comparison_table,
)
from rl.grpo import grpo_clipped_loss
from rl.policy_state import use_policy_snapshot
from rl.replay_memory import (
    assert_low_precision_activations,
    assert_model_eval_for_replay,
)

# Initialization identity: same params => same replay config => ratio ~ 1.
INIT_LOGP_ATOL = 1e-3
INIT_LOGP_RTOL = 1e-3
INIT_RATIO_ATOL = 1e-3
BLOCK_LOGP_ATOL = 1e-5
BLOCK_LOGP_RTOL = 1e-5

# Scientific thesis default: full live-cache trajectory gradients.
PRODUCTION_REPLAY = dict(LIVE_PRODUCTION_CACHED)

CHECKPOINTING_INCOMPATIBLE_MSG = (
    "live production-cached replay is incompatible with gradient checkpointing "
    "over mutable past_key_values: recompute sees a different KV length than "
    "the captured attention mask (e.g. mask (1,1,7,1465) vs kv_len 1458). "
    "Disable training.gradient_checkpointing. Do not fall back to Bfix or "
    "truncated_bptt_surrogate as the thesis path."
)

LIVE_CACHED_OOM_HINT = (
    "Live production-cached enable_grad retains LoRA autograd history in "
    "past_key_values (Case B: ~17.81GB → 23.80GB across blocks on RTX 3090 "
    "24GB; CUDA 'invalid argument' = disguised OOM). Do not switch to "
    "truncated_bptt_surrogate / BFIX. Broad save_on_cpu is diagnostic-only "
    "(A1: backward attn_bias strideH failure, ~185GB CPU RSS). Investigate "
    "selective saved_tensors_hooks (Priority-B), immutable-cache "
    "checkpointing, or multi-GPU sharding."
)


def assert_production_cached_replay_checkpointing_disabled(
    *,
    replay_backend: str = "live_production_cached_autograd",
    gradient_checkpointing: bool,
) -> None:
    if bool(gradient_checkpointing) and replay_backend in {
        "live_production_cached_autograd",
        "sequential_production_cached",
        "live_production_cached_save_on_cpu",
        "live_production_cached_selective_saved_tensor_offload",
        "truncated_bptt_surrogate",
        "sequential_production_matching_truncated_bp",
    }:
        raise RuntimeError(CHECKPOINTING_INCOMPATIBLE_MSG)


def assert_truncated_bptt_not_thesis_default(
    *,
    replay_backend: str,
    allow_truncated_bptt_surrogate: bool = False,
) -> None:
    is_surrogate = replay_backend in {
        "truncated_bptt_surrogate",
        "sequential_production_matching_truncated_bp",
    }
    if is_surrogate and not allow_truncated_bptt_surrogate:
        raise RuntimeError(SURROGATE_NOT_THESIS_DEFAULT_MSG)


def assert_grpo_loss_depends_only_on_trajectory_logprob() -> None:
    signature = inspect.signature(grpo_clipped_loss)
    allowed = {
        "current_log_probs",
        "old_log_probs",
        "advantages",
        "clip_epsilon",
    }
    unexpected = set(signature.parameters) - allowed
    if unexpected:
        raise RuntimeError(
            "GRPO loss depends on unexpected arguments: "
            f"{sorted(unexpected)}"
        )
    source = inspect.getsource(grpo_clipped_loss)
    for token in (
        "hidden_state",
        "hidden_states",
        "block_logit",
        "block_logits",
        "logits",
        "kl_",
        "reference_log",
    ):
        if token in source:
            raise RuntimeError(
                f"GRPO loss source references {token!r}; refusing update"
            )


def replay_semantics_report(model, config: Dict[str, Any]) -> Dict[str, Any]:
    lora = (config.get("model") or {}).get("lora") or {}
    dropout = float(lora.get("dropout", 0.0))
    model.eval()
    assert_model_eval_for_replay(model)
    return {
        "replay_mode_was_eval": True,
        "existing_replayer_calls_model_eval": True,
        "lora_dropout_config": dropout,
        "lora_dropout_active_during_replay": False,
        "reference_in_loss": False,
        "reference_is_metric_only": True,
        "production_replay": dict(PRODUCTION_REPLAY),
        "truncated_bptt_surrogate": dict(TRUNCATED_BPTT_SURROGATE),
        "truncated_bptt_is_thesis_default": False,
        "exact_backend_comparison_table": exact_backend_comparison_table(),
        "bfix_rejected": True,
    }


def score_policy_logprobs_sequential(
    replayer,
    traces: Sequence[Any],
    decoder_kwargs: Dict[str, Any],
    *,
    use_cache: bool = True,
    legacy_nocache_masks: bool = False,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    device = decoder_kwargs["input_ids"].device
    for trace in traces:
        value, _ = replayer.score(
            trace,
            use_cache=use_cache,
            legacy_nocache_masks=legacy_nocache_masks,
            **decoder_kwargs,
        )
        values.append(value.detach().float().reshape(()))
    if not values:
        return torch.zeros(0, device=device, dtype=torch.float32)
    return torch.stack(values)


def assert_initialization_ratios(
    current_logps: Sequence[float],
    old_logps: Sequence[float],
    *,
    atol: float = INIT_RATIO_ATOL,
) -> Dict[str, Any]:
    if len(current_logps) != len(old_logps):
        raise RuntimeError("current/old logp length mismatch")
    ratios = [
        float(torch.exp(torch.tensor(c - o)))
        for c, o in zip(current_logps, old_logps)
    ]
    bad = [r for r in ratios if abs(r - 1.0) > atol]
    if bad:
        raise RuntimeError(
            "initialization PPO ratios must be ~1 with identical replay "
            f"semantics; got {ratios[:8]}"
        )
    return {"ratios": ratios, "atol": atol, "ok": True}


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "out of memory" in text
        or "cuda oom" in text
        or "invalid argument" in text
    )


def _live_production_trajectory_backward(
    *,
    model,
    replayer,
    trace,
    decoder_kwargs: Dict[str, Any],
    old_logp: torch.Tensor,
    advantage: torch.Tensor,
    group_size: int,
    clip_epsilon: float,
) -> Dict[str, float]:
    """One live production-cached trajectory: score + (loss/G).backward().

    ``past_key_values`` remain gradient-bearing across blocks (scientific path).
    """
    assert_grpo_loss_depends_only_on_trajectory_logprob()
    assert_model_eval_for_replay(model)
    try:
        current_logp, _ = replayer.score(
            trace,
            use_cache=True,
            legacy_nocache_masks=False,
            **decoder_kwargs,
        )
        rollout_loss = grpo_clipped_loss(
            current_logp.reshape(1),
            old_logp.detach().reshape(1).to(
                device=current_logp.device, dtype=torch.float32
            ),
            advantage.detach().reshape(1).to(
                device=current_logp.device, dtype=torch.float32
            ),
            clip_epsilon=clip_epsilon,
        )
        (rollout_loss / float(group_size)).backward()
    except RuntimeError as exc:
        if _is_oom(exc):
            raise RuntimeError(
                "OOM/disguised-OOM during live production-cached GRPO replay. "
                + LIVE_CACHED_OOM_HINT
            ) from exc
        raise

    current_f = float(current_logp.detach().float().cpu())
    old_f = float(old_logp.detach().float().cpu())
    ratio = float(torch.exp(torch.tensor(current_f - old_f)))
    loss_f = float(rollout_loss.detach().float().cpu())
    del current_logp, rollout_loss
    release_cuda_temporaries(empty_cache=False)
    return {
        "current_log_prob": current_f,
        "loss_grpo": loss_f,
        "ppo_ratio": ratio,
        "backend": "live_production_cached_autograd",
        "stop_grad_past_key_values": False,
    }


def memory_safe_grpo_optimizer_step(
    *,
    model,
    optimizer,
    replayer,
    traces: Sequence[Any],
    advantages: torch.Tensor,
    decoder_kwargs: Dict[str, Any],
    old_snapshot,
    reference_snapshot=None,
    clip_epsilon: float,
    max_grad_norm: float,
    group_size: int,
    replay_microbatch_size: int = 1,
    gradient_replay_use_cache: bool = False,
    gradient_checkpointing: bool = False,
    score_reference: bool = False,
    assert_init_ratios: bool = False,
    memory_stages: Optional[Dict[str, Any]] = None,
    empty_cache_at_stage_boundaries: bool = False,
    replay_backend: str = "live_production_cached_autograd",
    allow_truncated_bptt_surrogate: bool = False,
) -> Dict[str, Any]:
    """One optimizer step over a G-group with scientific live-cache grads.

    Scientific reduction:
        L = (1/G) * sum_i ell(c_i, o_i, A_i)
    as sequential ``(ell_i / G).backward()`` then one ``optimizer.step()``.
    """
    if replay_backend in {
        "sequential_production_cached",
        "live_production_cached_autograd",
    }:
        replay_backend = "live_production_cached_autograd"
    if replay_backend in {
        "sequential_production_matching_truncated_bp",
        "truncated_bptt_surrogate",
    }:
        replay_backend = "truncated_bptt_surrogate"

    if int(group_size) != len(traces):
        raise RuntimeError("group_size must equal number of traces")
    if int(replay_microbatch_size) != 1:
        raise RuntimeError(
            "only replay_microbatch_size=1 is supported for exact memory-safe GRPO"
        )
    if advantages.numel() != group_size:
        raise RuntimeError("advantages must cover the full G=4 group")
    if replay_backend != "live_production_cached_autograd":
        if replay_backend == "truncated_bptt_surrogate":
            assert_truncated_bptt_not_thesis_default(
                replay_backend=replay_backend,
                allow_truncated_bptt_surrogate=allow_truncated_bptt_surrogate,
            )
            raise RuntimeError(
                "truncated_bptt_surrogate is explicitly allowed only for "
                "isolated diagnostics; the GRPO optimizer step refuses to run "
                "it as a training backend. "
                + SURROGATE_NOT_THESIS_DEFAULT_MSG
            )
        raise RuntimeError(
            f"replay_backend={replay_backend!r} is not the scientific path. "
            "Use live_production_cached_autograd. Bfix is rejected."
        )

    assert_truncated_bptt_not_thesis_default(
        replay_backend=replay_backend,
        allow_truncated_bptt_surrogate=False,
    )
    assert_production_cached_replay_checkpointing_disabled(
        replay_backend=replay_backend,
        gradient_checkpointing=gradient_checkpointing,
    )
    if gradient_replay_use_cache:
        raise RuntimeError(
            "gradient_replay_use_cache=true is rejected historical flag; "
            "scientific live-cache path already uses carry_cache=True. "
            "Set gradient_replay_use_cache=false."
        )

    stages = memory_stages if memory_stages is not None else {}
    device = decoder_kwargs["input_ids"].device
    model.eval()
    assert_model_eval_for_replay(model)
    activation_dtype = assert_low_precision_activations(model)
    stages["activation_dtype"] = activation_dtype
    stages["production_replay"] = dict(PRODUCTION_REPLAY)
    stages["gradient_checkpointing"] = False
    stages["live_cached_autograd"] = True
    stages["stop_grad_past_key_values"] = False
    stages["truncated_bptt_surrogate"] = False
    stages["replay_microbatch_size"] = 1
    stages["reference_is_metric_only"] = True
    stages["reference_in_loss"] = False
    stages["force_math_sdpa"] = False
    stages["bfix_rejected"] = True
    stages["exact_backend_comparison_table"] = exact_backend_comparison_table()

    reset_peak_memory(device)
    with torch.no_grad(), use_policy_snapshot(model, old_snapshot):
        old_logps = score_policy_logprobs_sequential(
            replayer,
            traces,
            decoder_kwargs,
            use_cache=True,
            legacy_nocache_masks=False,
        )
    record_stage_peak(stages, "old_policy_scoring", device)
    release_cuda_temporaries(
        empty_cache=empty_cache_at_stage_boundaries, device=device
    )

    reference_logps: Optional[List[float]] = None
    if score_reference:
        if reference_snapshot is None:
            raise RuntimeError("score_reference requires reference_snapshot")
        reset_peak_memory(device)
        disable = getattr(model.language_model, "disable_adapter", None)
        with torch.no_grad(), use_policy_snapshot(model, reference_snapshot):
            if callable(disable):
                with model.language_model.disable_adapter():
                    ref_tensor = score_policy_logprobs_sequential(
                        replayer,
                        traces,
                        decoder_kwargs,
                        use_cache=True,
                        legacy_nocache_masks=False,
                    )
            else:
                ref_tensor = score_policy_logprobs_sequential(
                    replayer,
                    traces,
                    decoder_kwargs,
                    use_cache=True,
                    legacy_nocache_masks=False,
                )
        reference_logps = [float(x) for x in ref_tensor.cpu()]
        del ref_tensor
        record_stage_peak(stages, "reference_policy_scoring", device)
        release_cuda_temporaries(
            empty_cache=empty_cache_at_stage_boundaries, device=device
        )
    else:
        stages["reference_policy_scoring"] = {
            "skipped": True,
            "reason": "metric-only and not requested",
        }

    optimizer.zero_grad(set_to_none=True)
    current_logps: List[float] = []
    per_rollout_losses: List[float] = []
    per_rollout_ratios: List[float] = []

    for group_index, trace in enumerate(traces):
        reset_peak_memory(device)
        metrics = _live_production_trajectory_backward(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            old_logp=old_logps[group_index],
            advantage=advantages[group_index],
            group_size=group_size,
            clip_epsilon=clip_epsilon,
        )
        current_logps.append(metrics["current_log_prob"])
        per_rollout_losses.append(metrics["loss_grpo"])
        per_rollout_ratios.append(metrics["ppo_ratio"])
        record_stage_peak(
            stages, f"current_replay_trajectory_{group_index}", device
        )
        release_cuda_temporaries(empty_cache=False)

    if assert_init_ratios:
        stages["initialization_ratio_check"] = assert_initialization_ratios(
            current_logps, [float(x) for x in old_logps.cpu()]
        )

    reset_peak_memory(device)
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(max_grad_norm),
    )
    record_stage_peak(stages, "during_grad_clip", device)
    optimizer.step()
    record_stage_peak(stages, "after_optimizer_step", device)
    release_cuda_temporaries(
        empty_cache=empty_cache_at_stage_boundaries, device=device
    )

    mean_loss = float(sum(per_rollout_losses) / group_size)
    return {
        "loss_grpo": mean_loss,
        "loss_total": mean_loss,
        "old_logps": [float(x) for x in old_logps.cpu()],
        "current_logps": current_logps,
        "reference_logps": reference_logps,
        "per_rollout_losses": per_rollout_losses,
        "per_rollout_ratios": per_rollout_ratios,
        "memory_stages": stages,
        "activation_dtype": activation_dtype,
        "production_replay": dict(PRODUCTION_REPLAY),
    }


# Backward-compatible alias.
UNIFIED_REPLAY = PRODUCTION_REPLAY

# Keep surrogate metadata importable for diagnostics only.
SURROGATE_REPLAY = dict(TRUNCATED_BPTT_SURROGATE)
