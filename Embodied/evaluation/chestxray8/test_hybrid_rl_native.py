#!/usr/bin/env python3
"""Unit tests for Hybrid-aware native GRPO final-prediction + shared rewards."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml
import torch

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.final_prediction import (  # noqa: E402
    attach_pbd_final_prediction,
    box_from_coord_box_tokens,
    box_from_token_span,
)
from rl.hybrid_rl import (  # noqa: E402
    LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED,
    LOGPROB_OBJECTIVE_FULL_TRAJECTORY,
    StochasticHybridRLDecoder,
    _local_handle_pattern,
)
from rl.pbd_rl import BlockTrace, PBDSamplingConfig, RolloutTrace, SlotTrace  # noqa: E402
from rl.rewards import ProductionRewardPipeline  # noqa: E402
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    DEFAULT_NATIVE_CONFIG,
    hybrid_logprob_objective,
    is_hybrid_rollout,
    load_resolved_config,
    resolve_rollout_path,
)
from run_pbd_rl_viability import build_viability_plan  # noqa: E402


TOKEN_IDS = {
    "box_start_token_id": 151668,
    "box_end_token_id": 151669,
    "coord_start_token_id": 151677,
    "coord_end_token_id": 152677,
    "none_token_id": 4064,
    "null_token_id": 152678,
    "im_end_token_id": 151645,
    "ref_end_token_id": 151673,
}


class _MockSemanticScorer:
    def __init__(self, value: float = 0.25) -> None:
        self.value = value
        self.calls = []

    def score(self, image_path, box_norm_1000, query) -> float:
        self.calls.append(tuple(box_norm_1000))
        return float(self.value)


def _slot(token_id: int, index: int = 0) -> SlotTrace:
    return SlotTrace(
        slot_index=index,
        action_token_id=token_id,
        support_kind="full",
        log_prob_old=-0.1 * (index + 1),
        support_size=10,
        top_k=0,
        top_p=1.0,
        temperature=1.0,
    )


def _box_tokens(x1=110, y1=210, x2=320, y2=430):
    cs = TOKEN_IDS["coord_start_token_id"]
    return [
        TOKEN_IDS["box_start_token_id"],
        cs + x1,
        cs + y1,
        cs + x2,
        cs + y2,
        TOKEN_IDS["box_end_token_id"],
    ]


def test_box_token_decode() -> None:
    tokens = _box_tokens()
    assert box_from_coord_box_tokens(tokens, TOKEN_IDS) == (110, 210, 320, 430)
    assert box_from_token_span(tokens, TOKEN_IDS) == (110, 210, 320, 430)
    assert box_from_coord_box_tokens(tokens[:5], TOKEN_IDS) is None


def test_pbd_final_prediction_rule() -> None:
    tokens = _box_tokens()
    slots = [_slot(t, i) for i, t in enumerate(tokens)]
    block = BlockTrace(
        block_index=0,
        prefix_length=10,
        cache_length_before=0,
        cache_length_after=0,
        block_type="box",
        position_ids=[],
        input_window_ids=[],
        action_token_ids=tokens,
        slots=slots,
    )
    trace = RolloutTrace(
        prompt_token_ids=[1, 2],
        generated_token_ids=tokens,
        blocks=[block],
        sampling=PBDSamplingConfig(),
        stopped_on_eos=True,
        truncated=False,
        decoded_text="<ref>X</ref><box><110><210><320><430></box>",
    )
    attach_pbd_final_prediction(trace, TOKEN_IDS)
    assert trace.decoder_path == "pbd"
    assert trace.reward_branch == "pbd"
    assert trace.has_unambiguous_committed_box
    assert trace.committed_final_box_norm_1000 == (110, 210, 320, 430)

    # Multiple box blocks → no unambiguous final box.
    trace2 = RolloutTrace(
        prompt_token_ids=[1],
        generated_token_ids=tokens + tokens,
        blocks=[block, block],
        sampling=PBDSamplingConfig(),
        stopped_on_eos=True,
        truncated=False,
    )
    attach_pbd_final_prediction(trace2, TOKEN_IDS)
    assert not trace2.has_unambiguous_committed_box
    assert trace2.reward_branch == "none"


def test_format_and_spatial_are_separate() -> None:
    """Spatial uses committed box; format uses emitted text grammar."""
    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[110, 210, 320, 430]],
    }
    scorer = _MockSemanticScorer(0.4)
    pipeline = ProductionRewardPipeline(
        scorer, parser_name="native_locateanything"
    )
    # Malformed text that still has a BOX_RE match, plus a clean committed box.
    malformed = (
        "<ref>Mass</ref><null><10><20><30><40></box></box>"
        "<box><110><210><320><430></box>"
    )
    components = pipeline.score(
        malformed,
        pair,
        committed_final_box=(110, 210, 320, 430),
        has_unambiguous_committed_box=True,
        reward_branch="pbd",
    )
    assert components.format_reward == 0.0
    assert components.spatial_reward == 1.0
    assert math.isclose(components.semantic_reward, 0.4)
    assert scorer.calls == [(110, 210, 320, 430)]

    # No committed box → format/spatial zero, semantic fallback.
    none_components = pipeline.score(
        "<ref>Mass</ref><box><110><210><320><430></box>",
        pair,
        committed_final_box=None,
        has_unambiguous_committed_box=False,
        reward_branch="none",
    )
    assert none_components.format_reward == 0.0
    assert none_components.spatial_reward == 0.0
    assert none_components.semantic_reward == 0.0
    assert none_components.total_reward == 0.0


def test_score_from_trace_uses_committed_not_text_mining() -> None:
    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[50, 50, 100, 100]],
    }
    # Text contains a high-IoU box, but the decoder committed a different box.
    text = (
        "<ref>Mass</ref><box><50><50><100><100></box>"
        "<box><900><900><950><950></box>"
    )
    tokens = _box_tokens(900, 900, 950, 950)
    slots = [_slot(t, i) for i, t in enumerate(tokens)]
    block = BlockTrace(
        block_index=0,
        prefix_length=10,
        cache_length_before=0,
        cache_length_after=0,
        block_type="box",
        position_ids=[],
        input_window_ids=[],
        action_token_ids=tokens,
        slots=slots,
    )
    trace = RolloutTrace(
        prompt_token_ids=[1],
        generated_token_ids=tokens,
        blocks=[block],
        sampling=PBDSamplingConfig(),
        stopped_on_eos=True,
        truncated=False,
        decoded_text=text,
    )
    attach_pbd_final_prediction(trace, TOKEN_IDS)
    # Two boxes in text → format invalid; committed is the decoder box.
    # Wait: attach sees one box *block*, so committed is (900,...).
    assert trace.committed_final_box_norm_1000 == (900, 900, 950, 950)
    pipeline = ProductionRewardPipeline(
        _MockSemanticScorer(0.1), parser_name="native_locateanything"
    )
    components = pipeline.score_from_trace(trace, pair)
    assert components.format_reward == 0.0  # two boxes in text
    assert components.spatial_reward == 0.0  # committed far from GT
    assert components.final_box_norm_1000 == (900, 900, 950, 950)


def test_hybrid_error_box_pattern_and_rejected_not_scored() -> None:
    cs = TOKEN_IDS["coord_start_token_id"]
    # box_start + one coord + junk → error_box prefix length 2
    proposal = [
        TOKEN_IDS["box_start_token_id"],
        cs + 10,
        0,
        0,
        0,
        TOKEN_IDS["null_token_id"],
    ]
    pattern = _local_handle_pattern(proposal, TOKEN_IDS)
    assert pattern["type"] == "error_box"
    assert pattern["tokens"] == [
        TOKEN_IDS["box_start_token_id"],
        cs + 10,
    ]
    assert pattern["need_switch_to_ar"] is True

    # Accepted coord box.
    ok = _box_tokens()
    ok_pattern = _local_handle_pattern(ok, TOKEN_IDS)
    assert ok_pattern["type"] == "coord_box"

    # Objective B: rejected proposal slots must not enter GRPO old_log_prob.
    prefix = pattern["tokens"]
    all_slots = [_slot(t, i) for i, t in enumerate(proposal)]
    decoder_b = StochasticHybridRLDecoder.__new__(StochasticHybridRLDecoder)
    decoder_b.logprob_objective = LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED
    scored_b = decoder_b._scored_slots_for_error_box(all_slots, len(prefix))
    assert len(scored_b) == len(prefix)

    decoder_a = StochasticHybridRLDecoder.__new__(StochasticHybridRLDecoder)
    decoder_a.logprob_objective = LOGPROB_OBJECTIVE_FULL_TRAJECTORY
    scored_a = decoder_a._scored_slots_for_error_box(all_slots, len(prefix))
    assert len(scored_a) == len(proposal)

    block = BlockTrace(
        block_index=0,
        prefix_length=5,
        cache_length_before=0,
        cache_length_after=0,
        block_type="error_box_prefix",
        position_ids=[],
        input_window_ids=[],
        action_token_ids=list(prefix),
        slots=scored_b,
        scored_for_grpo=True,
        source="pbd",
        rejected_proposal_token_ids=proposal,
    )
    ntp_slots = [_slot(TOKEN_IDS["box_end_token_id"], 0)]
    ntp = BlockTrace(
        block_index=1,
        prefix_length=7,
        cache_length_before=0,
        cache_length_after=0,
        block_type="ntp",
        position_ids=[],
        input_window_ids=[],
        action_token_ids=[TOKEN_IDS["box_end_token_id"]],
        slots=ntp_slots,
        scored_for_grpo=True,
        source="ntp_fallback",
    )
    trace = RolloutTrace(
        prompt_token_ids=[1],
        generated_token_ids=list(prefix) + [TOKEN_IDS["box_end_token_id"]],
        blocks=[block, ntp],
        sampling=PBDSamplingConfig(),
        stopped_on_eos=False,
        truncated=False,
        decoder_path="hybrid",
        reward_branch="ntp_fallback",
        fallback_triggered=True,
        rejected_pbd_proposals=[{"full_proposal_token_ids": proposal}],
    )
    # Under B, old_log_prob sums only committed prefix slots + NTP.
    assert len(block.rejected_proposal_token_ids) == 6
    assert len(block.action_token_ids) == 2
    assert len(block.slots) == 2
    expected = block.old_log_prob + ntp.old_log_prob
    assert math.isclose(trace.old_log_prob, expected)


def test_hybrid_logprob_objective_defaults_to_full_trajectory() -> None:
    hybrid = load_resolved_config(DEFAULT_HYBRID_NATIVE_CONFIG)
    assert hybrid_logprob_objective(hybrid) == LOGPROB_OBJECTIVE_FULL_TRAJECTORY
    assert hybrid["rollout"]["hybrid"]["score_rejected_pbd_proposals"] is True
    assert hybrid["rollout"]["hybrid"]["logprob_objective"] == "full_trajectory"
    doc = (CHEST_DIR / "rl" / "hybrid_rl.py").read_text(encoding="utf-8")
    assert "primary Hybrid experiment" in doc
    assert "surrogate ablation only" in doc


def test_conditional_committed_ablation_yaml() -> None:
    ablation = load_resolved_config(
        CHEST_DIR / "rl" / "chestxray8_grpo_hybrid_native_conditional_ablation.yaml"
    )
    assert hybrid_logprob_objective(ablation) == LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED
    assert ablation["rollout"]["hybrid"]["score_rejected_pbd_proposals"] is False
    assert "ablation" in ablation["experiment"]


def test_completed_box_count_rule() -> None:
    """Unambiguous failure when completed_box_count != 1."""
    tokens = _box_tokens()
    slots = [_slot(t, i) for i, t in enumerate(tokens)]
    block = BlockTrace(
        block_index=0,
        prefix_length=10,
        cache_length_before=0,
        cache_length_after=0,
        block_type="box",
        position_ids=[],
        input_window_ids=[],
        action_token_ids=tokens,
        slots=slots,
    )
    # Two completed box blocks → not unambiguous.
    trace = RolloutTrace(
        prompt_token_ids=[1],
        generated_token_ids=tokens + tokens,
        blocks=[block, block],
        sampling=PBDSamplingConfig(),
        stopped_on_eos=True,
        truncated=False,
    )
    attach_pbd_final_prediction(trace, TOKEN_IDS)
    assert not trace.has_unambiguous_committed_box
    assert trace.reward_branch == "none"


def test_hybrid_and_pbd_yaml_share_reward_weights() -> None:
    pbd = load_resolved_config(DEFAULT_NATIVE_CONFIG)
    hybrid = load_resolved_config(DEFAULT_HYBRID_NATIVE_CONFIG)
    assert resolve_rollout_path(pbd) == "stochastic_native_pbd_rl"
    assert resolve_rollout_path(hybrid) == "stochastic_native_hybrid_rl"
    assert is_hybrid_rollout(hybrid)
    assert not is_hybrid_rollout(pbd)
    for key in ("format", "spatial", "semantic"):
        assert pbd["rewards"][key]["weight"] == 1.0
        assert hybrid["rewards"][key]["weight"] == 1.0
    assert pbd["rewards"]["spatial"]["iou_threshold"] == 0.5
    assert hybrid["rewards"]["spatial"]["iou_threshold"] == 0.5
    assert pbd["objective"]["loss_total"] == "L_GRPO"
    assert hybrid["objective"]["loss_total"] == "L_GRPO"
    assert pbd["objective"]["supervised_losses"] == []
    assert hybrid["objective"]["supervised_losses"] == []
    assert hybrid["rollout"]["hybrid"]["score_rejected_pbd_proposals"] is True
    assert hybrid["rollout"]["hybrid"]["logprob_objective"] == "full_trajectory"
    assert hybrid["rewards"]["use_decoder_committed_final_box"] is True
    assert pbd["rewards"]["use_decoder_committed_final_box"] is True
    raw = yaml.safe_load(DEFAULT_HYBRID_NATIVE_CONFIG.read_text(encoding="utf-8"))
    assert "L_NTP" not in str(raw)
    assert "L_MTP" not in str(raw)
    plan = build_viability_plan(hybrid, population_size=790)
    assert plan["optimizer_updates"] == 0


def test_same_spatial_semantic_code_path() -> None:
    """PBD and Hybrid both call ProductionRewardPipeline.score_from_trace."""
    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[110, 210, 320, 430]],
    }
    pipeline = ProductionRewardPipeline(
        _MockSemanticScorer(0.3), parser_name="native_locateanything"
    )
    pbd_trace = RolloutTrace(
        prompt_token_ids=[1],
        generated_token_ids=_box_tokens(),
        blocks=[],
        sampling=PBDSamplingConfig(),
        stopped_on_eos=True,
        truncated=False,
        decoded_text="<ref>Mass</ref><box><110><210><320><430></box>",
        decoder_path="pbd",
        reward_branch="pbd",
        committed_final_box_norm_1000=(110, 210, 320, 430),
        has_unambiguous_committed_box=True,
    )
    hybrid_trace = RolloutTrace(
        prompt_token_ids=[1],
        generated_token_ids=_box_tokens(),
        blocks=[],
        sampling=PBDSamplingConfig(),
        stopped_on_eos=True,
        truncated=False,
        decoded_text="<ref>Mass</ref><box><110><210><320><430></box>",
        decoder_path="hybrid",
        reward_branch="ntp_fallback",
        committed_final_box_norm_1000=(110, 210, 320, 430),
        has_unambiguous_committed_box=True,
        fallback_triggered=True,
    )
    pbd_c = pipeline.score_from_trace(pbd_trace, pair)
    hybrid_c = pipeline.score_from_trace(hybrid_trace, pair)
    assert pbd_c.spatial_reward == hybrid_c.spatial_reward == 1.0
    assert math.isclose(pbd_c.semantic_reward, hybrid_c.semantic_reward)
    assert pbd_c.format_reward == hybrid_c.format_reward == 1.0


def test_diagnostic_observer_is_default_off_and_cpu_metadata_only() -> None:
    decoder = StochasticHybridRLDecoder.__new__(StochasticHybridRLDecoder)
    decoder.diagnostic_observer = None
    decoder._diagnostic_seed = 4000180
    decoder._observe("ignored", token_ids=[1, 2, 3])
    events = []
    decoder.diagnostic_observer = events.append
    decoder._observe("proposal", token_ids=[1, 2, 3])
    assert events == [
        {"event": "proposal", "seed": 4000180, "token_ids": [1, 2, 3]}
    ]
    summary = decoder._tensor_observation(torch.tensor([-1.0, 2.0]))
    assert not any(isinstance(value, torch.Tensor) for value in summary.values())


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"HYBRID NATIVE GRPO TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
