"""Trace-exact LocateAnything Slow-Mode (pure NTP) rollout and replay.

The pinned LocateAnything ``generate(generation_mode="slow")`` path starts in
AR mode and never enables MTP.  This module mirrors that one-token generation
path while retaining the sampled token identities required by GRPO replay.
There is deliberately no PBD proposal, proposal gate, or fallback branch.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from rl.final_prediction import box_from_token_span
from rl.hybrid_rl import (
    LOGPROB_OBJECTIVE_FULL_TRAJECTORY,
    HybridRolloutReplayer,
    _sampling_debug_context,
    _set_rotary_debug_context,
    resolve_hybrid_token_ids,
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


NTP_ONLY_DECODER_PATH = "ntp_only"
NTP_ONLY_REWARD_BRANCH = "ntp_only"


def validate_ntp_only_trace(trace: RolloutTrace) -> None:
    """Fail if any non-NTP or hidden proposal action entered the trajectory."""
    if getattr(trace, "decoder_path", None) != NTP_ONLY_DECODER_PATH:
        raise RuntimeError("NTP-only trace has the wrong decoder_path")
    if bool(getattr(trace, "fallback_triggered", False)):
        raise RuntimeError("NTP-only trace unexpectedly triggered Hybrid fallback")
    if getattr(trace, "proposal_events", None):
        raise RuntimeError("NTP-only trace unexpectedly contains PBD proposal events")
    if getattr(trace, "rejected_pbd_proposals", None):
        raise RuntimeError("NTP-only trace unexpectedly contains rejected proposals")

    replayed: List[int] = []
    for block in trace.blocks:
        if block.block_type != "ntp" or block.source != NTP_ONLY_DECODER_PATH:
            raise RuntimeError("NTP-only trace contains a non-NTP block")
        if not block.scored_for_grpo:
            raise RuntimeError("every NTP completion token must be scored for GRPO")
        if len(block.action_token_ids) != 1 or len(block.slots) != 1:
            raise RuntimeError("each NTP block must contain exactly one sampled token")
        if block.rejected_proposal_token_ids is not None:
            raise RuntimeError("NTP-only block unexpectedly stores proposal tokens")
        if int(block.action_token_ids[0]) != int(block.slots[0].action_token_id):
            raise RuntimeError("NTP action/slot identity mismatch")
        replayed.extend(int(token) for token in block.action_token_ids)
    if replayed != [int(token) for token in trace.generated_token_ids]:
        raise RuntimeError("NTP replay actions differ from committed completion tokens")


def ntp_only_trajectory_diagnostic(trace: RolloutTrace, component) -> Dict[str, Any]:
    """CPU/JSON first-update diagnostic for the pure committed trajectory."""
    validate_ntp_only_trace(trace)
    generated = [int(token) for token in trace.generated_token_ids]
    coordinate_count = sum(151677 <= token <= 152677 for token in generated)
    ntp_count = sum(len(block.action_token_ids) for block in trace.blocks)
    return {
        "generated_token_count": len(generated),
        "coordinate_token_count": int(coordinate_count),
        "pbd_proposal_token_count": 0,
        "rejected_proposal_token_count": 0,
        "ntp_generated_token_count": int(ntp_count),
        "parsed_predicted_bbox_norm_1000": (
            list(component.final_box_norm_1000)
            if component.final_box_norm_1000 is not None
            else None
        ),
        "raw_iou": float(component.final_iou),
        "format_reward": float(component.format_reward),
        "spatial_reward": float(component.spatial_reward),
        "semantic_reward": float(component.semantic_reward),
    }


class StochasticNTPRLDecoder:
    """Pure AR/NTP rollout matching LocateAnything ``generation_mode=slow``."""

    decoding_mode = NTP_ONLY_DECODER_PATH
    pbd_enabled = False
    mtp_enabled = False
    hybrid_fallback_enabled = False
    rejected_proposal_path_enabled = False

    def __init__(
        self,
        model,
        tokenizer,
        sampling: Optional[PBDSamplingConfig] = None,
        *,
        diagnostic_observer: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sampling = sampling or PBDSamplingConfig()
        self.sampling.validate()
        self.token_ids = resolve_hybrid_token_ids(model)
        self.diagnostic_observer = diagnostic_observer

    def _observe(self, event: str, **payload: Any) -> None:
        if self.diagnostic_observer is not None:
            self.diagnostic_observer(
                {"event": event, "seed": getattr(self, "_diagnostic_seed", None), **payload}
            )

    def _visual_features(self, pixel_values, image_grid_hws):
        pixel_values = pixel_values.to(self.model.language_model.dtype)
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(
                pixel_values.device, dtype=torch.int32
            )
        visual_features = self.model.extract_feature(pixel_values, image_grid_hws)
        if image_grid_hws is not None:
            visual_features = self.model.mlp1(torch.cat(visual_features, dim=0))
        return visual_features

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
            raise ValueError("NTP-only GRPO requires batch size 1")
        if force_first_box_block:
            raise RuntimeError("NTP-only decoding forbids forced PBD box blocks")
        self._diagnostic_seed = int(seed)
        self.model.eval()
        device = input_ids.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generated = input_ids.clone()
        prompt_length = int(input_ids.size(1))
        total_length = min(
            int(self.tokenizer.model_max_length), prompt_length + int(max_new_tokens)
        )
        full_positions = torch.arange(total_length + 1, device=device).unsqueeze(0)
        visual_features = self._visual_features(pixel_values, image_grid_hws)
        past_key_values = None
        blocks: List[BlockTrace] = []
        stopped = False
        committed_box: Optional[Tuple[int, int, int, int]] = None
        completed_box_count = 0
        open_box_tokens: Optional[List[int]] = None
        stop_reason: Optional[str] = None

        self._observe(
            "generation_start",
            decoding_mode=NTP_ONLY_DECODER_PATH,
            prompt_length=prompt_length,
            max_new_tokens=int(max_new_tokens),
        )
        with torch.no_grad():
            while generated.size(1) < total_length and not stopped:
                prefix_length = int(generated.size(1))
                cache_before = _cache_length(past_key_values)
                _set_rotary_debug_context(
                    self.model,
                    current_hybrid_block_index=len(blocks),
                    branch="NTP only",
                )
                prepared = self.model.language_model.prepare_inputs_for_generation(
                    generated,
                    past_key_values,
                    None,
                    inputs_embeds=None,
                    use_cache=True,
                    position_ids=full_positions[:, cache_before:generated.size(1)],
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
                        self.model,
                        block_index=len(blocks),
                        branch="NTP only",
                        current_sequence_length=int(prepared["input_ids"].size(1)),
                        cache_length=cache_before,
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
                        torch.tensor([action], device=device, dtype=generated.dtype)
                        .unsqueeze(0),
                    ],
                    dim=1,
                )
                position_ids = prepared.get("position_ids")
                blocks.append(
                    BlockTrace(
                        block_index=len(blocks),
                        prefix_length=prefix_length,
                        cache_length_before=cache_before,
                        cache_length_after=_cache_length(past_key_values),
                        block_type="ntp",
                        position_ids=(
                            position_ids[0].detach().cpu().tolist()
                            if position_ids is not None
                            else full_positions[0, cache_before:prefix_length]
                            .detach()
                            .cpu()
                            .tolist()
                        ),
                        input_window_ids=prepared["input_ids"][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        action_token_ids=[action],
                        slots=[slot],
                        scored_for_grpo=True,
                        source=NTP_ONLY_DECODER_PATH,
                        rejected_proposal_token_ids=None,
                    )
                )

                if open_box_tokens is None:
                    if action == int(self.token_ids["box_start_token_id"]):
                        open_box_tokens = [action]
                else:
                    open_box_tokens.append(action)
                    if action == int(self.token_ids["box_end_token_id"]):
                        box = box_from_token_span(open_box_tokens, self.token_ids)
                        if box is not None:
                            completed_box_count += 1
                            committed_box = box if completed_box_count == 1 else None
                        open_box_tokens = None

                if action == int(self.token_ids["im_end_token_id"]):
                    stopped = True
                    stop_reason = "im_end_token"
                self._observe(
                    "ntp_commit",
                    block_index=len(blocks) - 1,
                    action_token_id=action,
                    completed_box_count=completed_box_count,
                )

        generated_ids = generated[0, prompt_length:].detach().cpu().tolist()
        truncated = not stopped and int(generated.size(1)) >= total_length
        if stop_reason is None:
            stop_reason = "max_token_budget" if truncated else "completed"
        has_box = committed_box is not None and completed_box_count == 1
        trace = RolloutTrace(
            prompt_token_ids=input_ids[0].detach().cpu().tolist(),
            generated_token_ids=generated_ids,
            blocks=blocks,
            sampling=self.sampling,
            stopped_on_eos=stopped,
            truncated=truncated,
            decoded_text=self.tokenizer.decode(generated_ids, skip_special_tokens=False),
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
        self._observe(
            "generation_end",
            decoding_mode=NTP_ONLY_DECODER_PATH,
            generated_token_count=len(generated_ids),
            stop_reason=stop_reason,
            committed_final_box_norm_1000=trace.committed_final_box_norm_1000,
        )
        return trace


class NTPRolloutReplayer(HybridRolloutReplayer):
    """Teacher-forced replay restricted to committed NTP completion tokens."""

    decoding_mode = NTP_ONLY_DECODER_PATH
    pbd_enabled = False
    hybrid_fallback_enabled = False
    rejected_proposal_path_enabled = False

    def __init__(self, model, tokenizer) -> None:
        super().__init__(
            model,
            tokenizer,
            logprob_objective=LOGPROB_OBJECTIVE_FULL_TRAJECTORY,
            expected_decoder_path=NTP_ONLY_DECODER_PATH,
        )

    def _validate_decoder_path(self, trace: RolloutTrace) -> None:
        validate_ntp_only_trace(trace)
        super()._validate_decoder_path(trace)
