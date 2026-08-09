"""Hybrid-aware stochastic GRPO decoder for LocateAnything.

Mathematical policy note (read before Hybrid viability)
-------------------------------------------------------
The NTP fallback branch is a **deterministic** function of the sampled PBD
proposal ``a_pbd = (a_0,...,a_5)`` via ``handle_pattern``:

    branch = accept  if pattern(a_pbd) in {coord_box, ...}
    branch = fallback if pattern(a_pbd) == error_box

So ``P(branch | a_pbd) = 1``.  Excluding rejected proposal tokens from the
scored log-probability does **not** yield the probability of the complete
Hybrid trajectory.

Let ``τ`` be a Hybrid rollout.  On a fallback step:

    τ = (a_pbd, a_ntp)     # full sampled PBD proposal + NTP completion
    o = (prefix(a_pbd), a_ntp)   # tokens committed into the sequence

Exact full Hybrid trajectory probability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    log π(τ) = Σ_{i=0}^{5} log π(a_i | context_pbd)
             + Σ_j log π(a_ntp,j | context_ntp)

Alternative A — ``full_trajectory``  (**primary Hybrid experiment / default**)
  Score the full sampled PBD proposal (including rejected slots that
  determine the deterministic fallback gate), then the committed NTP
  tokens.  This **is** log π(τ) for the Hybrid rollout random variables
  that were sampled.

Alternative B — ``conditional_committed_output``  (**surrogate ablation only**)
  Score only tokens that enter the committed output sequence:

    log π̃(o) = Σ_{i < |prefix|} log π(a_i | context_pbd)
              + Σ_j log π(a_ntp,j | context_ntp)

  This **omits** Σ_{i ≥ |prefix|} log π(a_i | ...) for rejected proposal
  slots.  It is a **surrogate** objective: the conditional probability of
  committed output tokens after treating the branch/proposal remainder as
  outside the scored measure.  It must **not** be described as the
  probability of the full Hybrid rollout, and must not be used as the
  default Hybrid experiment.

Production Hybrid gate alignment
--------------------------------
Production ``generation_mode='hybrid'`` falls back to AR/NTP only when
``handle_pattern`` returns ``error_box`` for **one** MTP proposal block
(malformed / incomplete coordinate frame after ``decode_bbox_avg``).  It does
**not** fall back merely because the finished sequence contains multiple
otherwise-valid ``coord_box`` blocks.

Therefore Hybrid RL must:

1. Sample each MTP block with **unrestricted** (full-vocab) slot supports so
   ``error_box`` is reachable — the PBD-only ``sample_pbd_block`` path that
   forces coordinate + ``</box>`` supports after ``<box>`` makes ``error_box``
   unreachable and must not be used here.
2. Invoke NTP at the same per-block ``error_box`` gate as production.
3. Treat multi-``coord_box`` sequences like production online decoding
   (commit each valid box).  Single-object ``completed_box_count != 1`` is a
   **post-hoc final-prediction / reward** rule, not an online Hybrid gate.

Reward-scored final bbox (independent of A vs B)
------------------------------------------------
``reward-scored final bbox == decoder-committed final bbox``:

* Accepted PBD ``coord_box`` → decode those six committed tokens
  (``reward_branch="pbd"``).
* Fallback → decode ``prefix(a_pbd) + a_ntp`` ending at ``</box>``
  (``reward_branch="ntp_fallback"``).
* Rejected non-committed proposal slots are never used as the reward box
  and are never selected by GT IoU.
* Unambiguous final box iff ``completed_box_count == 1``;
  ``completed_box_count != 1`` → no unambiguous committed box.

Spatial/semantic rewards are shared with PBD-only via
``ProductionRewardPipeline.score_from_trace``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from rl.final_prediction import box_from_coord_box_tokens, box_from_token_span
from rl.pbd_rl import (
    BlockTrace,
    PBDSamplingConfig,
    RolloutTrace,
    SlotTrace,
    StochasticPBDRLDecoder,
    _cache_length,
    _detach_legacy_cache,
    _forward_language_model,
    _sample_slot,
    _support_ids,
    _truncate_legacy_cache,
    _unwrap_lm_output,
    build_filtered_categorical,
    resolve_token_ids,
)

try:
    from eaglevl.utils.locany.generate_utils import handle_pattern
except ImportError:  # pragma: no cover - fallback for unit tests without package path
    handle_pattern = None  # type: ignore[assignment]


LOGPROB_OBJECTIVE_FULL_TRAJECTORY = "full_trajectory"
LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED = "conditional_committed_output"
VALID_LOGPROB_OBJECTIVES = (
    LOGPROB_OBJECTIVE_FULL_TRAJECTORY,
    LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED,
)

# Primary Hybrid experiment default: exact full-trajectory log π(τ).
DEFAULT_LOGPROB_OBJECTIVE = LOGPROB_OBJECTIVE_FULL_TRAJECTORY


def _set_rotary_debug_context(model, **fields: Any) -> None:
    """Diagnostic metadata only; never participates in decoding or cache state."""
    try:
        from rl.two_gpu_shard import resolve_locateanything_qwen_decoder
        decoder = resolve_locateanything_qwen_decoder(model).decoder
    except Exception:
        return
    context = dict(getattr(decoder, "_chestxray8_rotary_context", {}) or {})
    context.update(fields)
    decoder._chestxray8_rotary_context = context


def _sampling_debug_context(
    model,
    *,
    block_index: int,
    branch: str,
    current_sequence_length: int,
    cache_length: int,
    lm_head_logits: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    try:
        from rl.two_gpu_shard import resolve_locateanything_qwen_decoder
        decoder = resolve_locateanything_qwen_decoder(model).decoder
    except Exception:
        return None
    if not bool(getattr(decoder, "_chestxray8_debug_position_ids", False)):
        return None
    context = dict(getattr(decoder, "_chestxray8_rotary_context", {}) or {})
    context["upstream_hidden_finite_report"] = getattr(
        decoder, "_chestxray8_last_hidden_finite_report", None
    )
    context["layer18_numeric_report"] = getattr(
        decoder, "_chestxray8_layer18_numeric_report", None
    )
    boundary_events = getattr(decoder, "_chestxray8_two_gpu_boundary_events", [])
    context["cross_device_boundary_event"] = (
        dict(boundary_events[-1]) if boundary_events else None
    )
    detached = lm_head_logits.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    context["lm_head_output_finite_report"] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "all_finite": bool(finite.all().item()),
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "positive_inf_count": int(torch.isposinf(detached).sum().item()),
        "negative_inf_count": int(torch.isneginf(detached).sum().item()),
        "finite_min": float(finite_values.min().float().item())
        if finite_values.numel()
        else None,
        "finite_max": float(finite_values.max().float().item())
        if finite_values.numel()
        else None,
    }
    sampling_path = context.get("sampling_diagnostic_json_path")
    if sampling_path:
        context["diagnostic_json_path"] = sampling_path
    context.update({"hybrid_block_index": block_index, "branch": branch,
                    "current_sequence_length": current_sequence_length, "cache_length": cache_length})
    return context


def resolve_hybrid_token_ids(model) -> Dict[str, int]:
    """Token map required by ``handle_pattern`` and Hybrid NTP."""
    out = resolve_token_ids(model)
    config = model.config
    text_config = config.text_config
    out.setdefault(
        "none_token_id", int(getattr(config, "none_token_id", 4064))
    )
    out.setdefault(
        "null_token_id",
        int(getattr(text_config, "null_token_id", 152678)),
    )
    out.setdefault(
        "ref_start_token_id", int(getattr(config, "ref_start_token_id", 151672))
    )
    out.setdefault(
        "ref_end_token_id", int(getattr(config, "ref_end_token_id", 151673))
    )
    return out


def resolve_logprob_objective(value: Optional[str] = None) -> str:
    name = value or DEFAULT_LOGPROB_OBJECTIVE
    if name not in VALID_LOGPROB_OBJECTIVES:
        raise ValueError(
            f"unsupported hybrid.logprob_objective={name!r}; "
            f"expected one of {VALID_LOGPROB_OBJECTIVES}"
        )
    return name


def sample_hybrid_mtp_block(
    block_logits: torch.Tensor,
    *,
    history_ids: Sequence[int],
    token_ids: Dict[str, int],
    config: PBDSamplingConfig,
    generator: Optional[torch.Generator] = None,
    diagnostic_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[SlotTrace]]:
    """Sample one MTP block with full-vocab supports (production-aligned).

    Unlike PBD-only ``sample_pbd_block``, this never forces coordinate /
    ``</box>`` supports after ``<box>``.  Production Hybrid classifies the
    unrestricted MTP proposal with ``handle_pattern``; forcing supports makes
    ``error_box`` unreachable and disables NTP fallback.
    """
    if block_logits.dim() != 2 or block_logits.size(0) != config.block_size:
        raise ValueError(
            f"expected [{config.block_size}, vocab] logits, "
            f"got {tuple(block_logits.shape)}"
        )
    slots: List[SlotTrace] = []
    actions: List[int] = []
    for slot_index in range(config.block_size):
        slot = _sample_slot(
            block_logits[slot_index],
            slot_index=slot_index,
            support_kind="full",
            history_ids=history_ids,
            token_ids=token_ids,
            config=config,
            generator=generator,
            diagnostic_context=diagnostic_context,
        )
        slots.append(slot)
        actions.append(int(slot.action_token_id))
    return actions, slots


def classify_hybrid_mtp_proposal(
    actions: Sequence[int],
    token_ids: Dict[str, int],
) -> Dict[str, Any]:
    """Apply the production Hybrid per-block gate (``handle_pattern``)."""
    return _local_handle_pattern(actions, token_ids)


def forced_pbd_box_proposal_can_emit_error_box() -> bool:
    """Documented invariant: PBD-only forced box supports cannot yield error_box."""
    return False


def _local_handle_pattern(
    actions: Sequence[int],
    token_ids: Dict[str, int],
) -> Dict[str, Any]:
    """Pure-Python hybrid pattern check (mirrors generate_utils.handle_pattern)."""
    x0 = [int(x) for x in actions]
    # Pad / trim to 6 like MTP proposals.
    if len(x0) < 6:
        x0 = x0 + [int(token_ids["null_token_id"])] * (6 - len(x0))
    x0 = x0[:6]
    if handle_pattern is not None:
        return handle_pattern(
            torch.tensor(x0), token_ids, generation_mode="hybrid"
        )

    null_token_id = int(token_ids["null_token_id"])
    im_end_token_id = int(token_ids["im_end_token_id"])
    box_start_token_id = int(token_ids["box_start_token_id"])
    box_end_token_id = int(token_ids["box_end_token_id"])
    none_token_id = int(token_ids["none_token_id"])
    coord_start_token_id = int(token_ids["coord_start_token_id"])
    coord_end_token_id = int(token_ids["coord_end_token_id"])
    ref_end_token_id = int(token_ids.get("ref_end_token_id", -1))

    if x0[0] in (null_token_id, im_end_token_id):
        return {
            "type": "im_end",
            "tokens": [im_end_token_id],
            "need_switch_to_ar": False,
            "is_terminal": True,
        }
    if x0[:2] == [box_start_token_id, none_token_id]:
        return {
            "type": "empty_box",
            "tokens": [box_start_token_id, none_token_id, box_end_token_id],
            "need_switch_to_ar": False,
            "is_terminal": False,
        }
    if x0[0] == box_start_token_id:
        coord_ix = 1
        for coord in x0[1:5]:
            if coord_start_token_id <= coord <= coord_end_token_id:
                coord_ix += 1
            else:
                break
        if coord_ix == 5 and x0[5] == box_end_token_id:
            return {
                "type": "coord_box",
                "tokens": x0,
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        if coord_ix == 3 and x0[3] == box_end_token_id:
            return {
                "type": "point_box",
                "tokens": x0[:4],
                "need_switch_to_ar": False,
                "is_terminal": False,
            }
        return {
            "type": "error_box",
            "tokens": x0[:coord_ix],
            "need_switch_to_ar": True,
            "is_terminal": False,
        }
    trimmed = list(x0)
    for i, token in enumerate(trimmed):
        if token == null_token_id:
            trimmed = trimmed[:i]
            break
    if (
        len(trimmed) >= 2
        and ref_end_token_id >= 0
        and trimmed[-1] == trimmed[-2] == ref_end_token_id
    ):
        trimmed = trimmed[:-1]
    return {
        "type": "ref_object",
        "tokens": trimmed,
        "need_switch_to_ar": False,
        "is_terminal": False,
    }


class StochasticHybridRLDecoder:
    """Stochastic Hybrid rollout with explicit A/B log-prob objective.

    Reward boxes always come from committed tokens only.  GRPO log-probs
    follow ``logprob_objective`` (see module docstring).
    """

    def __init__(
        self,
        model,
        tokenizer,
        sampling: Optional[PBDSamplingConfig] = None,
        *,
        logprob_objective: str = DEFAULT_LOGPROB_OBJECTIVE,
        diagnostic_observer: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sampling = sampling or PBDSamplingConfig()
        self.sampling.validate()
        self.logprob_objective = resolve_logprob_objective(logprob_objective)
        self.token_ids = resolve_hybrid_token_ids(model)
        self._pbd = StochasticPBDRLDecoder(model, tokenizer, self.sampling)
        self.diagnostic_observer = diagnostic_observer

    def _observe(self, event: str, **payload: Any) -> None:
        """Send CPU-only diagnostics when explicitly enabled."""
        if self.diagnostic_observer is None:
            return
        self.diagnostic_observer(
            {
                "event": event,
                "seed": getattr(self, "_diagnostic_seed", None),
                **payload,
            }
        )

    def _tensor_observation(self, value: torch.Tensor) -> Dict[str, Any]:
        detached = value.detach()
        finite = torch.isfinite(detached)
        finite_values = detached[finite]
        return {
            "shape": list(detached.shape),
            "stride": list(detached.stride()),
            "layout": str(detached.layout),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "finite_count": int(finite.sum().item()),
            "element_count": int(detached.numel()),
            "finite_min": float(finite_values.min().float().item())
            if finite_values.numel()
            else None,
            "finite_max": float(finite_values.max().float().item())
            if finite_values.numel()
            else None,
        }

    def _visual_features(self, pixel_values, image_grid_hws):
        return self._pbd._visual_features(pixel_values, image_grid_hws)

    def _scored_slots_for_error_box(self, slots, committed_len: int):
        """Select which proposal slots enter the GRPO log-probability."""
        if self.logprob_objective == LOGPROB_OBJECTIVE_FULL_TRAJECTORY:
            # A: include the full sampled proposal that determines the gate.
            return list(slots)
        # B: surrogate — only committed prefix slots.
        return list(slots[:committed_len])

    def generate(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
        max_new_tokens: int,
        seed: int,
        force_first_box_block: bool = False,
    ) -> RolloutTrace:
        if input_ids.size(0) != 1:
            raise ValueError("hybrid GRPO currently requires batch size 1")
        self._diagnostic_seed = int(seed)
        self.model.eval()
        device = input_ids.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        block_size = self.sampling.block_size
        mask_tail = torch.full(
            (1, block_size - 1),
            int(self.token_ids["default_mask_token_id"]),
            dtype=input_ids.dtype,
            device=device,
        )
        generated = input_ids.clone()
        prompt_length = int(input_ids.size(1))
        total_length = min(
            int(self.tokenizer.model_max_length), prompt_length + int(max_new_tokens)
        )
        full_positions = torch.arange(
            total_length + block_size, device=device
        ).unsqueeze(0)
        visual_features, _ = self._visual_features(pixel_values, image_grid_hws)
        past_key_values = None
        blocks: List[BlockTrace] = []
        stopped = False
        budget_exhausted = False
        use_mtp = True
        fallback_triggered = False
        rejected_proposals: List[Dict[str, Any]] = []
        reward_branch = "none"
        committed_box: Optional[Tuple[int, int, int, int]] = None
        completed_box_count = 0
        open_box_tokens: Optional[List[int]] = None
        stop_reason: Optional[str] = None
        proposal_events: List[Dict[str, Any]] = []

        self._observe(
            "generation_start",
            seed=int(seed),
            prompt_token_ids=input_ids[0].detach().cpu().tolist(),
            prompt_length=prompt_length,
            max_new_tokens=int(max_new_tokens),
            total_length=total_length,
            image={
                "pixel_values": self._tensor_observation(pixel_values),
                "image_grid_hws": torch.as_tensor(image_grid_hws).detach().cpu().tolist(),
            },
            token_ids={key: int(value) for key, value in self.token_ids.items()},
        )

        with torch.no_grad():
            while generated.size(1) < total_length and not stopped:
                prefix_length = int(generated.size(1))
                cache_before = _cache_length(past_key_values)
                remaining = total_length - prefix_length
                if remaining <= 0:
                    budget_exhausted = True
                    stop_reason = "max_token_budget"
                    break

                if use_mtp:
                    _set_rotary_debug_context(self.model, current_hybrid_block_index=len(blocks), branch="PBD proposal")
                    window = torch.cat(
                        (generated, generated[:, -1:].clone(), mask_tail), dim=1
                    )
                    start = cache_before
                    position_ids = full_positions[:, start : window.size(1)].clone()
                    position_ids[0, -block_size:] -= 1
                    prepared = self.model.language_model.prepare_inputs_for_generation(
                        window,
                        past_key_values,
                        None,
                        inputs_embeds=None,
                        use_cache=True,
                        position_ids=position_ids,
                    )
                    outputs = _unwrap_lm_output(
                        _forward_language_model(
                            self.model,
                            prepared,
                            visual_features=visual_features if not blocks else None,
                        )
                    )
                    block_logits = outputs.logits[0, -block_size:, :]
                    # Production-aligned: unrestricted MTP proposal, then
                    # handle_pattern. Do NOT use sample_pbd_block here — its
                    # forced coordinate/</box> supports make error_box impossible.
                    actions, slots = sample_hybrid_mtp_block(
                        block_logits,
                        history_ids=generated[0].tolist(),
                        token_ids=self.token_ids,
                        config=self.sampling,
                        generator=generator,
                        diagnostic_context=_sampling_debug_context(
                            self.model, block_index=len(blocks), branch="PBD proposal",
                            current_sequence_length=int(prepared["input_ids"].size(1)), cache_length=cache_before,
                            lm_head_logits=block_logits,
                        ),
                    )
                    pattern = classify_hybrid_mtp_proposal(actions, self.token_ids)
                    committed = [int(x) for x in pattern["tokens"]]
                    hit_eos = bool(pattern.get("is_terminal"))
                    proposal_event = {
                        "block_index": len(blocks),
                        "prefix_length": prefix_length,
                        "cache_length_before": cache_before,
                        "cache_length_after_forward": _cache_length(outputs.past_key_values),
                        "position_ids": position_ids[0].detach().cpu().tolist(),
                        "input_window_ids": prepared["input_ids"][0].detach().cpu().tolist(),
                        "proposed_token_ids": [int(value) for value in actions],
                        "classification": str(pattern["type"]),
                        "committed_token_ids": list(committed),
                        "is_terminal": bool(pattern.get("is_terminal")),
                        "remaining_budget": int(remaining),
                        "logits": self._tensor_observation(block_logits),
                    }
                    proposal_events.append(proposal_event)
                    self._observe("pbd_proposal", **proposal_event)
                    if len(committed) > remaining:
                        budget_exhausted = True
                        stop_reason = "proposal_exceeds_remaining_token_budget"
                        break

                    if pattern["type"] == "error_box":
                        fallback_triggered = True
                        scored_slots = self._scored_slots_for_error_box(
                            slots, len(committed)
                        )
                        rejected_proposals.append(
                            {
                                "full_proposal_token_ids": list(actions),
                                "committed_prefix_token_ids": list(committed),
                                "scored_proposal_token_ids": [
                                    int(s.action_token_id) for s in scored_slots
                                ],
                                "logprob_objective": self.logprob_objective,
                                "prefix_length": prefix_length,
                            }
                        )
                        past_key_values = _truncate_legacy_cache(
                            outputs.past_key_values, prefix_length
                        )
                        generated = torch.cat(
                            [
                                generated,
                                torch.tensor(
                                    committed, device=device, dtype=generated.dtype
                                ).unsqueeze(0),
                            ],
                            dim=1,
                        )
                        blocks.append(
                            BlockTrace(
                                block_index=len(blocks),
                                prefix_length=prefix_length,
                                cache_length_before=cache_before,
                                cache_length_after=_cache_length(past_key_values),
                                block_type="error_box_prefix",
                                position_ids=position_ids[0].detach().cpu().tolist(),
                                input_window_ids=prepared["input_ids"][0]
                                .detach()
                                .cpu()
                                .tolist(),
                                # Sequence advances only by committed prefix.
                                action_token_ids=list(committed),
                                # GRPO slots follow A (full proposal) or B (prefix).
                                slots=scored_slots,
                                scored_for_grpo=True,
                                source="pbd",
                                rejected_proposal_token_ids=list(actions),
                            )
                        )
                        open_box_tokens = list(committed)
                        use_mtp = False
                        self._observe(
                            "fallback_entered",
                            block_index=len(blocks) - 1,
                            rejected_proposal_token_ids=[int(value) for value in actions],
                            committed_prefix_token_ids=list(committed),
                            completed_box_count=completed_box_count,
                        )
                        continue

                    # Accepted MTP pattern.
                    committed_slots = slots[: len(committed)]
                    if pattern["type"] == "coord_box" and len(committed) == 6:
                        committed_slots = slots
                    past_key_values = _truncate_legacy_cache(
                        outputs.past_key_values, prefix_length
                    )
                    generated = torch.cat(
                        [
                            generated,
                            torch.tensor(
                                committed, device=device, dtype=generated.dtype
                            ).unsqueeze(0),
                        ],
                        dim=1,
                    )
                    source = "pbd"
                    mapped_type = pattern["type"]
                    if mapped_type == "coord_box":
                        box = box_from_coord_box_tokens(committed, self.token_ids)
                        if box is not None:
                            completed_box_count += 1
                            if completed_box_count == 1:
                                committed_box = box
                                reward_branch = "pbd"
                            else:
                                # completed_box_count != 1 → unambiguous failure
                                committed_box = None
                                reward_branch = "none"
                        open_box_tokens = None
                    elif mapped_type == "im_end":
                        stopped = True
                        stop_reason = "im_end_token"
                    blocks.append(
                        BlockTrace(
                            block_index=len(blocks),
                            prefix_length=prefix_length,
                            cache_length_before=cache_before,
                            cache_length_after=_cache_length(past_key_values),
                            block_type=mapped_type,
                            position_ids=position_ids[0].detach().cpu().tolist(),
                            input_window_ids=prepared["input_ids"][0]
                            .detach()
                            .cpu()
                            .tolist(),
                            action_token_ids=list(committed),
                            slots=committed_slots,
                            scored_for_grpo=True,
                            source=source,
                            rejected_proposal_token_ids=None,
                        )
                    )
                    if hit_eos or pattern.get("is_terminal"):
                        stopped = True
                    continue

                # ---------------- NTP / AR fallback ----------------
                _set_rotary_debug_context(self.model, current_hybrid_block_index=len(blocks), branch="NTP fallback")
                prepared = self.model.language_model.prepare_inputs_for_generation(
                    generated,
                    past_key_values,
                    None,
                    inputs_embeds=None,
                    use_cache=True,
                    position_ids=full_positions[:, cache_before : generated.size(1)],
                )
                outputs = _unwrap_lm_output(
                    _forward_language_model(
                        self.model,
                        prepared,
                        visual_features=visual_features if not blocks else None,
                    )
                )
                logits = outputs.logits[0, -1, :]
                slot = _sample_slot(
                    logits,
                    slot_index=0,
                    support_kind="full",
                    history_ids=generated[0].tolist(),
                    token_ids=self.token_ids,
                    config=self.sampling,
                    generator=generator,
                    diagnostic_context=_sampling_debug_context(
                        self.model, block_index=len(blocks), branch="NTP fallback",
                        current_sequence_length=int(prepared["input_ids"].size(1)), cache_length=cache_before,
                        lm_head_logits=logits,
                    ),
                )
                action = int(slot.action_token_id)
                past_key_values = _truncate_legacy_cache(
                    outputs.past_key_values, prefix_length
                )
                generated = torch.cat(
                    [
                        generated,
                        torch.tensor(
                            [action], device=device, dtype=generated.dtype
                        ).unsqueeze(0),
                    ],
                    dim=1,
                )
                pos = prepared.get("position_ids")
                if pos is not None:
                    position_list = pos[0].detach().cpu().tolist()
                else:
                    position_list = (
                        full_positions[0, cache_before : generated.size(1) - 1]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                blocks.append(
                    BlockTrace(
                        block_index=len(blocks),
                        prefix_length=prefix_length,
                        cache_length_before=cache_before,
                        cache_length_after=_cache_length(past_key_values),
                        block_type="ntp",
                        position_ids=position_list,
                        input_window_ids=prepared["input_ids"][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        action_token_ids=[action],
                        slots=[slot],
                        scored_for_grpo=True,
                        source="ntp_fallback",
                        rejected_proposal_token_ids=None,
                    )
                )
                if open_box_tokens is not None:
                    open_box_tokens.append(action)
                if action == int(self.token_ids["box_end_token_id"]):
                    use_mtp = True
                    if open_box_tokens is not None:
                        box = box_from_token_span(open_box_tokens, self.token_ids)
                        if box is not None:
                            completed_box_count += 1
                            if completed_box_count == 1:
                                committed_box = box
                                reward_branch = "ntp_fallback"
                            else:
                                # completed_box_count != 1 → unambiguous failure
                                committed_box = None
                                reward_branch = "none"
                        open_box_tokens = None
                elif action == int(self.token_ids["im_end_token_id"]):
                    stopped = True
                    stop_reason = "im_end_token"
                self._observe(
                    "ntp_commit",
                    block_index=len(blocks) - 1,
                    action_token_id=action,
                    cache_length_before=cache_before,
                    cache_length_after=_cache_length(past_key_values),
                    completed_box_count=completed_box_count,
                    reward_branch=reward_branch,
                )

        generated_ids = generated[0, prompt_length:].detach().cpu().tolist()
        truncated = not stopped and (
            budget_exhausted or int(generated.size(1)) >= total_length
        )
        if stop_reason is None:
            stop_reason = "max_token_budget" if truncated else "completed"
        # Unambiguous iff exactly one completed box was committed.
        has_box = committed_box is not None and completed_box_count == 1
        trace = RolloutTrace(
            prompt_token_ids=input_ids[0].detach().cpu().tolist(),
            generated_token_ids=generated_ids,
            blocks=blocks,
            sampling=self.sampling,
            stopped_on_eos=stopped,
            truncated=truncated,
            decoded_text=self.tokenizer.decode(
                generated_ids, skip_special_tokens=False
            ),
            decoder_path="hybrid",
            reward_branch=reward_branch if has_box else "none",
            committed_final_box_norm_1000=committed_box if has_box else None,
            has_unambiguous_committed_box=has_box,
            fallback_triggered=fallback_triggered,
            rejected_pbd_proposals=rejected_proposals,
            stop_reason=stop_reason,
            max_new_tokens=int(max_new_tokens),
            max_reachable_generated_length=(
                int(max_new_tokens) - (int(max_new_tokens) % int(block_size))
                if not fallback_triggered
                else len(generated_ids)
            ),
            proposal_events=proposal_events,
        )
        self._observe(
            "generation_end",
            stop_reason=stop_reason,
            stopped_on_eos=bool(stopped),
            truncated=bool(truncated),
            generated_token_ids=list(generated_ids),
            decoded_text=trace.decoded_text,
            reward_branch=trace.reward_branch,
            committed_final_box_norm_1000=trace.committed_final_box_norm_1000,
            completed_box_count=completed_box_count,
            fallback_triggered=fallback_triggered,
            rejected_pbd_proposals=list(rejected_proposals),
        )
        return trace


class HybridRolloutReplayer:
    """Replay Hybrid GRPO log-probs under objective A or B.

    Sequence replay always appends only committed ``action_token_ids``.
    For ``error_box_prefix`` blocks under objective A, ``slots`` may include
    rejected proposal actions that are scored from the same MTP logits but
    never appended to the token stream.

    Trajectory log-probability is exactly the sum of scored block
    log-probabilities. Replay always runs under ``model.eval()`` (existing
    scientific path; LoRA dropout disabled).
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        logprob_objective: str = DEFAULT_LOGPROB_OBJECTIVE,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.token_ids = resolve_hybrid_token_ids(model)
        self.logprob_objective = resolve_logprob_objective(logprob_objective)

    def _projector_visual_features(self, pixel_values, image_grid_hws):
        pixel_values = pixel_values.to(self.model.language_model.dtype)
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(
                pixel_values.device, dtype=torch.int32
            )
        visual_features = self.model.extract_feature(pixel_values, image_grid_hws)
        if image_grid_hws is not None:
            visual_features = self.model.mlp1(torch.cat(visual_features, dim=0))
        return visual_features, image_grid_hws

    def _append_actions(self, generated: torch.Tensor, action_token_ids: Sequence[int]):
        if not action_token_ids:
            return generated
        return torch.cat(
            [
                generated,
                torch.tensor(
                    list(action_token_ids),
                    device=generated.device,
                    dtype=generated.dtype,
                ).unsqueeze(0),
            ],
            dim=1,
        )

    def _score_one_block(
        self,
        *,
        block: BlockTrace,
        generated: torch.Tensor,
        mask_tail: torch.Tensor,
        full_positions: torch.Tensor,
        past_key_values,
        visual_features,
        inject_visual: bool,
        carry_cache: bool,
        sampling: PBDSamplingConfig,
        block_size: int,
        legacy_nocache_masks: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        """Score one Hybrid block.

        ``carry_cache`` controls whether a KV cache is reused across blocks.
        Independently, the model forward always requests production inference
        MTP masks (``use_cache=True`` in ``prepare_inputs_for_generation``)
        unless ``legacy_nocache_masks`` is set for diagnostics.

        Why: Qwen2 eval SDPA chooses
        ``update_causal_mask_for_one_gen_window_2d`` when ``use_cache=True``
        and ``update_causal_mask_with_pad_non_visible_2d`` when False. The
        latter is not equivalent to cached generation and changes action
        log-probs. Full-prefix recompute must keep the production mask path.
        """
        if int(generated.size(1)) != block.prefix_length:
            raise RuntimeError("hybrid replay prefix length differs from trace")
        cache_before = _cache_length(past_key_values) if carry_cache else 0
        if carry_cache and cache_before != block.cache_length_before:
            raise RuntimeError("hybrid replay cache length differs from trace")

        # Production MTP/AR mask selection key inside modeling_qwen2.py.
        model_use_cache = False if legacy_nocache_masks else True

        if block.source == "ntp_fallback" or block.block_type == "ntp":
            if carry_cache:
                position_ids = full_positions[:, cache_before : generated.size(1)]
                past = past_key_values
            else:
                position_ids = full_positions[:, : generated.size(1)]
                past = None
            prepared = self.model.language_model.prepare_inputs_for_generation(
                generated,
                past,
                None,
                inputs_embeds=None,
                use_cache=model_use_cache,
                position_ids=position_ids,
            )
            outputs = _unwrap_lm_output(
                _forward_language_model(
                    self.model,
                    prepared,
                    visual_features=visual_features if inject_visual else None,
                )
            )
            logits = outputs.logits[0, -1, :].float()
            del outputs.logits
            if len(block.slots) != 1 or len(block.action_token_ids) != 1:
                raise RuntimeError("NTP block must contain exactly one action")
            slot = block.slots[0]
            distribution = build_filtered_categorical(
                logits,
                history_ids=generated[0].tolist(),
                config=sampling,
                allowed_token_ids=None,
            )
            value = distribution.log_prob(slot.action_token_id)
            del logits
            next_cache = (
                _truncate_legacy_cache(outputs.past_key_values, int(generated.size(1)))
                if carry_cache
                else None
            )
            return value, next_cache, [value]

        window = torch.cat((generated, generated[:, -1:].clone(), mask_tail), dim=1)
        if carry_cache:
            position_ids = full_positions[:, cache_before : window.size(1)].clone()
            past = past_key_values
        else:
            position_ids = full_positions[:, : window.size(1)].clone()
            past = None
        position_ids[0, -block_size:] -= 1
        prepared = self.model.language_model.prepare_inputs_for_generation(
            window,
            past,
            None,
            inputs_embeds=None,
            use_cache=model_use_cache,
            position_ids=position_ids,
        )
        outputs = _unwrap_lm_output(
            _forward_language_model(
                self.model,
                prepared,
                visual_features=visual_features if inject_visual else None,
            )
        )
        block_logits = outputs.logits[0, -block_size:, :].float()
        del outputs.logits
        per_slot: List[torch.Tensor] = []
        for slot in block.slots:
            distribution = build_filtered_categorical(
                block_logits[slot.slot_index],
                history_ids=generated[0].tolist(),
                config=sampling,
                allowed_token_ids=_support_ids(slot.support_kind, self.token_ids),
            )
            per_slot.append(distribution.log_prob(slot.action_token_id))
        value = torch.stack(per_slot).sum()
        del block_logits
        next_cache = (
            _truncate_legacy_cache(outputs.past_key_values, int(generated.size(1)))
            if carry_cache
            else None
        )
        return value, next_cache, per_slot

    def score(
        self,
        trace: RolloutTrace,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
        use_cache: bool = True,
        legacy_nocache_masks: bool = False,
        on_scored_block_begin=None,
        on_scored_block_end=None,
        on_scored_block_cache=None,
        on_scored_token_logps=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Replay trajectory log-probs.

        ``use_cache`` here means *carry KV across blocks* (generation-style).
        Model-side ``use_cache`` for MTP mask selection stays True unless
        ``legacy_nocache_masks`` is requested for diagnostics.

        Optional ``on_scored_block_begin`` / ``on_scored_block_end`` are
        diagnostic-only callbacks and must not alter tensors or control flow.
        """
        if input_ids[0].tolist() != trace.prompt_token_ids:
            raise RuntimeError("replay prompt does not match rollout trace")
        if getattr(trace, "decoder_path", "pbd") != "hybrid":
            raise RuntimeError("HybridRolloutReplayer requires decoder_path=hybrid")
        self.model.eval()
        block_size = trace.sampling.block_size
        device = input_ids.device
        generated = input_ids.clone()
        mask_tail = torch.full(
            (1, block_size - 1),
            int(self.token_ids["default_mask_token_id"]),
            dtype=input_ids.dtype,
            device=device,
        )
        max_length = len(trace.prompt_token_ids) + len(trace.generated_token_ids)
        full_positions = torch.arange(
            max_length + block_size, device=device
        ).unsqueeze(0)
        visual_features, _ = self._projector_visual_features(
            pixel_values, image_grid_hws
        )
        past_key_values = None
        block_logps: List[torch.Tensor] = []
        carry_cache = bool(use_cache)
        scored_index = 0

        for block in trace.blocks:
            if not block.scored_for_grpo:
                generated = self._append_actions(generated, block.action_token_ids)
                continue
            if on_scored_block_begin is not None:
                on_scored_block_begin(scored_index, block)
            inject_visual = (not carry_cache) or (not block_logps)
            prior_cache = past_key_values
            _set_rotary_debug_context(self.model, current_hybrid_block_index=block.block_index,
                                      branch="NTP fallback" if block.source == "ntp_fallback" else "PBD proposal/replay")
            value, past_key_values, per_token_logps = self._score_one_block(
                block=block,
                generated=generated,
                mask_tail=mask_tail,
                full_positions=full_positions,
                past_key_values=past_key_values,
                visual_features=visual_features,
                inject_visual=inject_visual,
                carry_cache=carry_cache,
                sampling=trace.sampling,
                block_size=block_size,
                legacy_nocache_masks=legacy_nocache_masks,
            )
            block_logps.append(value)
            if on_scored_token_logps is not None:
                on_scored_token_logps(block, per_token_logps)
            if on_scored_block_cache is not None:
                on_scored_block_cache(
                    scored_index,
                    block,
                    prior_cache,
                    past_key_values,
                )
            if on_scored_block_end is not None:
                on_scored_block_end(scored_index, block, value)
            scored_index += 1
            generated = self._append_actions(generated, block.action_token_ids)

        if generated[0, len(trace.prompt_token_ids) :].tolist() != trace.generated_token_ids:
            raise RuntimeError("hybrid replayed actions differ from emitted tokens")
        if not block_logps:
            return torch.zeros((), device=device, requires_grad=True), []
        return torch.stack(block_logps).sum(), block_logps

    def score_with_token_logprobs(
        self,
        trace: RolloutTrace,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
        use_cache: bool = True,
        legacy_nocache_masks: bool = False,
        on_scored_block_begin=None,
        on_scored_block_end=None,
        on_scored_block_cache=None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[Dict[str, Any]]]:
        """Replay and expose exactly one log-probability per scored action token.

        The flattened token stream is the Hybrid trajectory measure, not just
        emitted text: rejected PBD proposal slots are included under the
        production ``full_trajectory`` objective. Prompt/image tokens are not
        scored and therefore never enter this list.
        """
        token_logps: List[torch.Tensor] = []
        token_metadata: List[Dict[str, Any]] = []

        def collect(block: BlockTrace, values: List[torch.Tensor]) -> None:
            if len(values) != len(block.slots):
                raise RuntimeError("Hybrid token-logp count differs from scored slots")
            for slot, value in zip(block.slots, values):
                committed = bool(
                    slot.slot_index < len(block.action_token_ids)
                    and int(block.action_token_ids[slot.slot_index])
                    == int(slot.action_token_id)
                )
                token_logps.append(value)
                token_metadata.append(
                    {
                        "block_index": int(block.block_index),
                        "block_type": str(block.block_type),
                        "source": str(block.source),
                        "slot_index": int(slot.slot_index),
                        "action_token_id": int(slot.action_token_id),
                        "support_kind": str(slot.support_kind),
                        "committed_to_generated_stream": committed,
                        "rejected_pbd_proposal_token": bool(
                            not committed and block.source != "ntp_fallback"
                        ),
                        "mask": 1,
                    }
                )

        total, blocks = self.score(
            trace,
            pixel_values=pixel_values,
            input_ids=input_ids,
            image_grid_hws=image_grid_hws,
            use_cache=use_cache,
            legacy_nocache_masks=legacy_nocache_masks,
            on_scored_block_begin=on_scored_block_begin,
            on_scored_block_end=on_scored_block_end,
            on_scored_block_cache=on_scored_block_cache,
            on_scored_token_logps=collect,
        )
        if not token_logps:
            raise RuntimeError("Hybrid full trajectory contains no scored action tokens")
        stacked = torch.stack(token_logps)
        if not torch.allclose(
            stacked.sum().detach().float(),
            total.detach().float(),
            atol=1e-4,
            rtol=1e-5,
        ):
            raise RuntimeError("Hybrid per-token logps do not sum to trajectory logp")
        return total, blocks, stacked, token_metadata

    def iter_scored_block_logprobs(
        self,
        trace: RolloutTrace,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
        use_cache: bool = False,
        legacy_nocache_masks: bool = False,
    ):
        """Yield one independent scored-block log-prob graph at a time.

        Used by exact Pass-B chain-rule backward. Each yield recomputes the
        projector path so projector gradients remain live. Caller must
        ``backward`` and drop references before requesting the next block.

        Independent graphs require ``use_cache=False`` (no carried KV), but
        production MTP masks are still requested unless
        ``legacy_nocache_masks`` is set.
        """
        if input_ids[0].tolist() != trace.prompt_token_ids:
            raise RuntimeError("replay prompt does not match rollout trace")
        if getattr(trace, "decoder_path", "pbd") != "hybrid":
            raise RuntimeError("HybridRolloutReplayer requires decoder_path=hybrid")
        if use_cache:
            raise RuntimeError(
                "iter_scored_block_logprobs requires use_cache=False "
                "(no carried KV) for independent per-block graphs"
            )
        self.model.eval()
        block_size = trace.sampling.block_size
        device = input_ids.device
        generated = input_ids.clone()
        mask_tail = torch.full(
            (1, block_size - 1),
            int(self.token_ids["default_mask_token_id"]),
            dtype=input_ids.dtype,
            device=device,
        )
        max_length = len(trace.prompt_token_ids) + len(trace.generated_token_ids)
        full_positions = torch.arange(
            max_length + block_size, device=device
        ).unsqueeze(0)

        for block in trace.blocks:
            if not block.scored_for_grpo:
                generated = self._append_actions(generated, block.action_token_ids)
                continue
            # Fresh projector features each block so grads are not shared across
            # already-backward'd graphs.
            visual_features, _ = self._projector_visual_features(
                pixel_values, image_grid_hws
            )
            value, _, _per_token_logps = self._score_one_block(
                block=block,
                generated=generated,
                mask_tail=mask_tail,
                full_positions=full_positions,
                past_key_values=None,
                visual_features=visual_features,
                inject_visual=True,
                carry_cache=False,
                sampling=trace.sampling,
                block_size=block_size,
                legacy_nocache_masks=legacy_nocache_masks,
            )
            yield value
            generated = self._append_actions(generated, block.action_token_ids)

        if generated[0, len(trace.prompt_token_ids) :].tolist() != trace.generated_token_ids:
            raise RuntimeError("hybrid replayed actions differ from emitted tokens")

    def score_autograd_safe(
        self,
        trace: RolloutTrace,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Production-matching log-probs with stop-grad carried KV.

        Same windows / position_ids / ``one_gen_window`` masks / truncate
        protocol as ``score(use_cache=True)``, but every carried
        ``past_key_values`` tensor is detached before the next block so LoRA
        autograd history cannot accumulate across the trajectory.

        Forward values match production-cached replay; gradients are truncated
        BPTT through the current block only (prefix KV treated as constant).
        """
        if input_ids[0].tolist() != trace.prompt_token_ids:
            raise RuntimeError("replay prompt does not match rollout trace")
        if getattr(trace, "decoder_path", "pbd") != "hybrid":
            raise RuntimeError("HybridRolloutReplayer requires decoder_path=hybrid")
        self.model.eval()
        block_size = trace.sampling.block_size
        device = input_ids.device
        generated = input_ids.clone()
        mask_tail = torch.full(
            (1, block_size - 1),
            int(self.token_ids["default_mask_token_id"]),
            dtype=input_ids.dtype,
            device=device,
        )
        max_length = len(trace.prompt_token_ids) + len(trace.generated_token_ids)
        full_positions = torch.arange(
            max_length + block_size, device=device
        ).unsqueeze(0)
        visual_features, _ = self._projector_visual_features(
            pixel_values, image_grid_hws
        )
        past_key_values = None
        block_logps: List[torch.Tensor] = []

        for block in trace.blocks:
            if not block.scored_for_grpo:
                generated = self._append_actions(generated, block.action_token_ids)
                continue
            inject_visual = not block_logps
            past_key_values = _detach_legacy_cache(past_key_values)
            value, past_key_values, _per_token_logps = self._score_one_block(
                block=block,
                generated=generated,
                mask_tail=mask_tail,
                full_positions=full_positions,
                past_key_values=past_key_values,
                visual_features=visual_features,
                inject_visual=inject_visual,
                carry_cache=True,
                sampling=trace.sampling,
                block_size=block_size,
                legacy_nocache_masks=False,
            )
            block_logps.append(value)
            # Drop grad history from the just-produced cache before the next
            # block; keep the production truncate/advance protocol.
            past_key_values = _detach_legacy_cache(past_key_values)
            generated = self._append_actions(generated, block.action_token_ids)

        if generated[0, len(trace.prompt_token_ids) :].tolist() != trace.generated_token_ids:
            raise RuntimeError("hybrid replayed actions differ from emitted tokens")
        if not block_logps:
            return torch.zeros((), device=device, requires_grad=True), []
        return torch.stack(block_logps).sum(), block_logps

    def accumulate_autograd_safe_grads(
        self,
        trace: RolloutTrace,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
        d_loss_d_logp: float,
    ) -> List[float]:
        """Per-block production-matching truncated BP; free each block graph.

        ``d_loss_d_logp`` is ``∂ℓ/∂logπ`` (already including the ``1/G`` factor
        when used from the sequential GRPO step). For each scored block:
        stop-grad production past → score block → ``(coeff * logp_i).backward()``.
        """
        if input_ids[0].tolist() != trace.prompt_token_ids:
            raise RuntimeError("replay prompt does not match rollout trace")
        if getattr(trace, "decoder_path", "pbd") != "hybrid":
            raise RuntimeError("HybridRolloutReplayer requires decoder_path=hybrid")
        self.model.eval()
        block_size = trace.sampling.block_size
        device = input_ids.device
        generated = input_ids.clone()
        mask_tail = torch.full(
            (1, block_size - 1),
            int(self.token_ids["default_mask_token_id"]),
            dtype=input_ids.dtype,
            device=device,
        )
        max_length = len(trace.prompt_token_ids) + len(trace.generated_token_ids)
        full_positions = torch.arange(
            max_length + block_size, device=device
        ).unsqueeze(0)
        # Projector grads: only the first scored block injects visuals under
        # production carry_cache semantics (same as ``score``).
        past_key_values = None
        block_vals: List[float] = []
        scored = 0
        coeff = float(d_loss_d_logp)

        for block in trace.blocks:
            if not block.scored_for_grpo:
                generated = self._append_actions(generated, block.action_token_ids)
                continue
            inject_visual = scored == 0
            if inject_visual:
                visual_features, _ = self._projector_visual_features(
                    pixel_values, image_grid_hws
                )
            else:
                visual_features = None
            past_key_values = _detach_legacy_cache(past_key_values)
            value, past_key_values, _per_token_logps = self._score_one_block(
                block=block,
                generated=generated,
                mask_tail=mask_tail,
                full_positions=full_positions,
                past_key_values=past_key_values,
                visual_features=visual_features,
                inject_visual=inject_visual,
                carry_cache=True,
                sampling=trace.sampling,
                block_size=block_size,
                legacy_nocache_masks=False,
            )
            block_vals.append(float(value.detach().float().cpu()))
            (value * coeff).backward()
            past_key_values = _detach_legacy_cache(past_key_values)
            generated = self._append_actions(generated, block.action_token_ids)
            scored += 1
            del value, visual_features

        if generated[0, len(trace.prompt_token_ids) :].tolist() != trace.generated_token_ids:
            raise RuntimeError("hybrid replayed actions differ from emitted tokens")
        return block_vals


HYBRID_POLICY_DOC = StochasticHybridRLDecoder.__doc__
