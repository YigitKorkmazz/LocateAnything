"""CPU/static contract checks for the pure-NTP G=8 GRPO ablation."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_two_gpu_hybrid_grpo_checkpoints as evaluator  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_production as ablation  # noqa: E402
from rl.ntp_rl import (  # noqa: E402
    NTP_ONLY_DECODER_PATH,
    NTPRolloutReplayer,
    StochasticNTPRLDecoder,
    ntp_only_trajectory_diagnostic,
    validate_ntp_only_trace,
)
from rl.pbd_rl import BlockTrace, PBDSamplingConfig, RolloutTrace, SlotTrace  # noqa: E402
from rl.rewards import RewardComponents  # noqa: E402
from rl.runtime import (  # noqa: E402
    ROLLOUT_PATH_NTP_ONLY,
    is_ntp_only_rollout,
)
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402


def configs():
    frozen = resolve_experiment_config(ablation.PROJECTOR_FROZEN_SPEC)
    ntp = resolve_experiment_config(ablation.DEFAULT_SPEC)
    return frozen, ntp


def test_only_rollout_trajectory_mode_changes():
    frozen, ntp = configs()
    for section in (
        "prompt",
        "model",
        "objective",
        "rewards",
        "policy_state",
        "data",
        "training",
        "runtime_contract",
        "evaluation",
        "medclip",
    ):
        assert ntp[section] == frozen[section]
    assert ntp["rollout"] == ablation._expected_ntp_rollout(frozen["rollout"])
    assert ntp["rollout"]["path"] == ROLLOUT_PATH_NTP_ONLY
    assert is_ntp_only_rollout(ntp)
    assert ntp["rollout"]["bbox_format"] == "<box><x1><y1><x2><y2></box>"
    assert ntp["rollout"]["coordinate_token_id_range"] == [151677, 152677]
    ablation.validate_ntponly_contract(ntp)


def _trace(*, proposal=False):
    actions = [151672, 100, 151673, 151674, 151677, 151678, 151679, 151680, 151675]
    blocks = [
        BlockTrace(
            block_index=index,
            prefix_length=10 + index,
            cache_length_before=9 + index,
            cache_length_after=10 + index,
            block_type="ntp",
            position_ids=[9 + index],
            input_window_ids=[1],
            action_token_ids=[token],
            slots=[
                SlotTrace(
                    slot_index=0,
                    action_token_id=token,
                    support_kind="full",
                    log_prob_old=-1.0,
                    support_size=100,
                    top_k=0,
                    top_p=1.0,
                    temperature=1.0,
                )
            ],
            scored_for_grpo=True,
            source=NTP_ONLY_DECODER_PATH,
            rejected_proposal_token_ids=None,
        )
        for index, token in enumerate(actions)
    ]
    return RolloutTrace(
        prompt_token_ids=[1] * 10,
        generated_token_ids=actions,
        blocks=blocks,
        sampling=PBDSamplingConfig(),
        stopped_on_eos=False,
        truncated=False,
        decoded_text="<ref>x</ref><box><0><1><2><3></box>",
        decoder_path=NTP_ONLY_DECODER_PATH,
        reward_branch=NTP_ONLY_DECODER_PATH,
        committed_final_box_norm_1000=(0, 1, 2, 3),
        has_unambiguous_committed_box=True,
        fallback_triggered=False,
        rejected_pbd_proposals=[{"forbidden": True}] if proposal else [],
        proposal_events=[],
    )


def test_ntp_trace_is_exactly_the_committed_token_stream():
    trace = _trace()
    validate_ntp_only_trace(trace)
    component = RewardComponents(
        format_reward=1.0,
        spatial_reward=0.0,
        semantic_reward=0.25,
        total_reward=1.25,
        final_iou=0.2,
        format_valid=True,
        geometry_valid=True,
        final_box_norm_1000=(0, 1, 2, 3),
        parse_error=None,
        reward_branch=NTP_ONLY_DECODER_PATH,
        has_unambiguous_committed_box=True,
    )
    diagnostic = ntp_only_trajectory_diagnostic(trace, component)
    assert diagnostic["generated_token_count"] == len(trace.generated_token_ids)
    assert diagnostic["ntp_generated_token_count"] == len(trace.generated_token_ids)
    assert diagnostic["coordinate_token_count"] == 4
    assert diagnostic["pbd_proposal_token_count"] == 0
    assert diagnostic["rejected_proposal_token_count"] == 0
    assert diagnostic["parsed_predicted_bbox_norm_1000"] == [0, 1, 2, 3]
    assert diagnostic["raw_iou"] == 0.2


def test_ntp_trace_rejects_any_proposal_metadata():
    try:
        validate_ntp_only_trace(_trace(proposal=True))
    except RuntimeError as error:
        assert "rejected proposals" in str(error)
    else:
        raise AssertionError("proposal-bearing NTP trace did not fail")


def test_ntp_decoder_has_no_pbd_or_hybrid_control_path():
    source = inspect.getsource(StochasticNTPRLDecoder.generate)
    for forbidden in (
        "sample_hybrid_mtp_block",
        "sample_pbd_block",
        "classify_hybrid_mtp_proposal",
        "use_mtp",
        "fallback_entered",
    ):
        assert forbidden not in source
    assert StochasticNTPRLDecoder.pbd_enabled is False
    assert StochasticNTPRLDecoder.mtp_enabled is False
    assert StochasticNTPRLDecoder.hybrid_fallback_enabled is False
    assert StochasticNTPRLDecoder.rejected_proposal_path_enabled is False
    assert NTPRolloutReplayer.pbd_enabled is False
    assert NTPRolloutReplayer.hybrid_fallback_enabled is False


def test_existing_evaluator_uses_shared_rollout_mode_factory():
    source = inspect.getsource(evaluator._evaluate_condition)
    assert "build_rollout_decoder(model, tokenizer, config)" in source
    assert "StochasticHybridRLDecoder(" not in source


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("{} NTP-only ablation tests passed".format(len(tests)))
