"""Rollout generation: shared single-row prefill + G independent NTP decodes.

Reimplements, on top of the unmodified `rl.pbd_rl` / `rl.hybrid_rl` / `rl.ntp_rl`
primitives, the validated design from
`results/chestxray8_profile_isolated/v6_cached_old_research/v6_cached_old_logprobs_report.json`:

  1. The G=4 rollouts of one prompt group share an identical prompt prefix
     (same image, same text prompt), so the prompt-prefill forward pass
     (vision+projector features already cached separately, see
     `cache_group_visual_features`) is run once, at batch size 1.
  2. Each of the G rollouts then decodes serially (batch size 1) from a
     *cloned* copy of the post-prefill KV cache, with its own seeded
     `torch.Generator`. This reproduces exactly the same sampled-token
     distribution as running G independent from-scratch generations (the
     prefill forward is deterministic given identical weights/inputs), while
     eliminating 3 of 4 redundant prefill passes per group. Padded
     multi-row batched decoding was tried upstream and rejected: at
     top_p=0.9 a token sampled from a batched nucleus can fall outside the
     serial nucleus, producing -inf log-probs. Serial single-row decode does
     not have this failure mode.
  3. Every sampled token's filtered log-prob (`SlotTrace.log_prob_old`,
     produced by `rl.pbd_rl._sample_slot` at the moment of sampling) is kept
     on the trace and is later summed as `old_log_prob` for the GRPO ratio,
     instead of re-running a full teacher-forced replay pass to recompute
     it. This is proven exact (bit-identical filtered log-probs, 0 support
     changes) against a from-scratch replay in the report above; this
     backend re-verifies it independently in
     `tests/test_generation_parity.py`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rl.final_prediction import box_from_token_span
from rl.hybrid_rl import resolve_hybrid_token_ids
from rl.ntp_rl import (
    NTP_ONLY_DECODER_PATH,
    NTP_ONLY_REWARD_BRANCH,
    validate_ntp_only_trace,
)
from rl.pbd_rl import (
    BlockTrace,
    PBDSamplingConfig,
    RolloutTrace,
    _cache_length,
    _forward_language_model,
    _sample_slot,
    _truncate_legacy_cache,
    _unwrap_lm_output,
)

Box = Tuple[int, int, int, int]


def cache_group_visual_features(model: Any, pixel_values: torch.Tensor, image_grid_hws: Any) -> torch.Tensor:
    """Compute the frozen vision-encoder + projector output once per group.

    Exact because `model.vision_model` and `model.mlp1` are frozen and
    deterministic under `torch.no_grad()`: re-running this per rollout would
    produce a bit-identical tensor. The cache is a plain leaf tensor with
    `requires_grad=False` (checked below) so it can be safely reused across
    generation and replay without holding a stale autograd graph.
    """
    pixel_values = pixel_values.to(model.language_model.dtype)
    if isinstance(image_grid_hws, np.ndarray):
        image_grid_hws = torch.from_numpy(image_grid_hws).to(pixel_values.device, dtype=torch.int32)
    model.vision_model.eval()
    model.mlp1.eval()
    with torch.no_grad():
        visual_features = model.extract_feature(pixel_values, image_grid_hws)
        if image_grid_hws is not None:
            visual_features = model.mlp1(torch.cat(visual_features, dim=0))
    cached = visual_features.detach()
    if cached.requires_grad:
        raise RuntimeError("cached visual representation unexpectedly requires gradients")
    return cached


def _commit_box_state(
    action: int,
    token_ids: Dict[str, int],
    open_box_tokens: Optional[List[int]],
    completed_box_count: int,
    committed_box: Optional[Box],
) -> Tuple[Optional[List[int]], int, Optional[Box]]:
    if open_box_tokens is None:
        if action == int(token_ids["box_start_token_id"]):
            return [action], completed_box_count, committed_box
        return None, completed_box_count, committed_box
    open_box_tokens = list(open_box_tokens)
    open_box_tokens.append(action)
    if action == int(token_ids["box_end_token_id"]):
        box = box_from_token_span(open_box_tokens, token_ids)
        if box is not None:
            completed_box_count += 1
            committed_box = box if completed_box_count == 1 else None
        return None, completed_box_count, committed_box
    return open_box_tokens, completed_box_count, committed_box


def _trace_from_completion(
    *,
    prompt_token_ids: Sequence[int],
    generated_ids: Sequence[int],
    slots: Sequence[Any],
    sampling: PBDSamplingConfig,
    tokenizer: Any,
    token_ids: Dict[str, int],
    stopped: bool,
    truncated: bool,
    stop_reason: str,
    max_new_tokens: int,
    committed_box: Optional[Box],
    completed_box_count: int,
) -> RolloutTrace:
    prompt_length = len(prompt_token_ids)
    blocks: List[BlockTrace] = []
    for index, (action, slot) in enumerate(zip(generated_ids, slots)):
        prefix_length = prompt_length + index
        cache_before = 0 if index == 0 else prefix_length - 1
        blocks.append(
            BlockTrace(
                block_index=index,
                prefix_length=prefix_length,
                cache_length_before=cache_before,
                cache_length_after=prefix_length,
                block_type="ntp",
                position_ids=list(range(cache_before, prefix_length)),
                input_window_ids=(list(prompt_token_ids) if index == 0 else [int(generated_ids[index - 1])]),
                action_token_ids=[int(action)],
                slots=[slot],
                scored_for_grpo=True,
                source=NTP_ONLY_DECODER_PATH,
                rejected_proposal_token_ids=None,
            )
        )
    has_box = committed_box is not None and completed_box_count == 1
    trace = RolloutTrace(
        prompt_token_ids=list(prompt_token_ids),
        generated_token_ids=[int(t) for t in generated_ids],
        blocks=blocks,
        sampling=sampling,
        stopped_on_eos=bool(stopped),
        truncated=bool(truncated),
        decoded_text=tokenizer.decode(list(generated_ids), skip_special_tokens=False),
        decoder_path=NTP_ONLY_DECODER_PATH,
        reward_branch=NTP_ONLY_REWARD_BRANCH if has_box else "none",
        committed_final_box_norm_1000=committed_box if has_box else None,
        has_unambiguous_committed_box=has_box,
        fallback_triggered=False,
        rejected_pbd_proposals=[],
        stop_reason=stop_reason,
        max_new_tokens=int(max_new_tokens),
        max_reachable_generated_length=int(max_new_tokens),
        proposal_events=[],
    )
    validate_ntp_only_trace(trace)
    return trace


def _clone_legacy_cache(past_key_values):
    return tuple((kv[0].clone(), kv[1].clone()) for kv in past_key_values)


def generate_group_shared_prefill(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    cached_visual_features: torch.Tensor,
    *,
    sampling: PBDSamplingConfig,
    max_new_tokens: int,
    seeds: Sequence[int],
) -> List[RolloutTrace]:
    if input_ids.size(0) != 1:
        raise ValueError("shared-prefill NTP generation expects a single prompt row")
    if len(seeds) < 1:
        raise ValueError("at least one seed is required")
    token_ids = resolve_hybrid_token_ids(model)
    device = input_ids.device
    prompt_token_ids = input_ids[0].detach().cpu().tolist()
    prompt_length = int(input_ids.size(1))
    total_length = min(int(tokenizer.model_max_length), prompt_length + int(max_new_tokens))
    features = cached_visual_features.reshape(-1, cached_visual_features.size(-1))
    full_positions = torch.arange(total_length + 1, device=device).unsqueeze(0)

    with torch.no_grad():
        prefix_length = prompt_length
        prepared = model.language_model.prepare_inputs_for_generation(
            input_ids,
            None,
            None,
            inputs_embeds=None,
            use_cache=True,
            position_ids=full_positions[:, :prefix_length],
        )
        outputs = _unwrap_lm_output(_forward_language_model(model, prepared, visual_features=features))
        prompt_logits = outputs.logits[0, -1, :].clone()
        prompt_cache = _truncate_legacy_cache(outputs.past_key_values, prefix_length)
        del outputs

        traces: List[RolloutTrace] = []
        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            generated = input_ids.clone()
            past_key_values = _clone_legacy_cache(prompt_cache)
            completions: List[int] = []
            slots_row: List[Any] = []
            stopped = False
            reason: Optional[str] = None
            open_box_tokens: Optional[List[int]] = None
            completed_box_count = 0
            committed_box: Optional[Box] = None
            first = True
            while generated.size(1) < total_length and not stopped:
                prefix_length = int(generated.size(1))
                if first:
                    logits = prompt_logits
                    first = False
                else:
                    cache_before = _cache_length(past_key_values)
                    prepared = model.language_model.prepare_inputs_for_generation(
                        generated,
                        past_key_values,
                        None,
                        inputs_embeds=None,
                        use_cache=True,
                        position_ids=full_positions[:, cache_before:generated.size(1)],
                    )
                    outputs = _unwrap_lm_output(_forward_language_model(model, prepared, visual_features=None))
                    logits = outputs.logits[0, -1, :]
                    past_key_values = _truncate_legacy_cache(outputs.past_key_values, prefix_length)
                history = prompt_token_ids + completions
                slot = _sample_slot(
                    logits,
                    slot_index=0,
                    support_kind="full",
                    history_ids=history,
                    token_ids=token_ids,
                    config=sampling,
                    generator=generator,
                )
                action = int(slot.action_token_id)
                completions.append(action)
                slots_row.append(slot)
                open_box_tokens, completed_box_count, committed_box = _commit_box_state(
                    action, token_ids, open_box_tokens, completed_box_count, committed_box
                )
                generated = torch.cat(
                    [generated, torch.tensor([action], device=device, dtype=generated.dtype).unsqueeze(0)], dim=1
                )
                if action == int(token_ids["im_end_token_id"]):
                    stopped = True
                    reason = "im_end_token"
            truncated = (not stopped) and (prompt_length + len(completions) >= total_length)
            if reason is None:
                reason = "max_token_budget" if truncated else "completed"
            trace = _trace_from_completion(
                prompt_token_ids=prompt_token_ids,
                generated_ids=completions,
                slots=slots_row,
                sampling=sampling,
                tokenizer=tokenizer,
                token_ids=token_ids,
                stopped=stopped,
                truncated=truncated,
                stop_reason=reason,
                max_new_tokens=max_new_tokens,
                committed_box=committed_box,
                completed_box_count=completed_box_count,
            )
            traces.append(trace)
    return traces


def cached_old_logprob_sum(trace: RolloutTrace, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Sum of generation-time filtered log-probs; used directly as pi_old.

    Proven exact against a from-scratch teacher-forced replay (see module
    docstring); see `tests/test_generation_parity.py` for this backend's own
    re-verification.
    """
    values = [float(block.slots[0].log_prob_old) for block in trace.blocks if block.scored_for_grpo]
    if len(values) != len(trace.generated_token_ids):
        raise RuntimeError("scored old-logprob count does not match completion length")
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("non-finite generation-time old logprob")
    return torch.tensor(sum(values), device=device, dtype=dtype)
