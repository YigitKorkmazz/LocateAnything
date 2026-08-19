"""Static contracts for the clean T=1.2/top-p=0.9 NTP ablation."""

from __future__ import annotations

import copy
import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_production as ablation  # noqa: E402
from rl.ntp_rl import NTPRolloutReplayer  # noqa: E402
from rl.spatial_density_diagnostics import (  # noqa: E402
    aggregate_spatial_density_window,
    build_group_spatial_diagnostic,
)
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402

production = ablation.production


def test_only_temperature_top_p_and_diagnostics_differ_from_previous_ntp():
    previous = resolve_experiment_config(ablation.previous_ntp.DEFAULT_SPEC)
    current = resolve_experiment_config(ablation.DEFAULT_SPEC)
    for section in (
        "prompt", "model", "objective", "rewards", "policy_state", "data",
        "training", "runtime_contract", "evaluation", "medclip",
    ):
        assert current[section] == previous[section]
    expected_rollout = copy.deepcopy(previous["rollout"])
    expected_rollout["temperature"] = 1.2
    expected_rollout["top_p"] = 0.9
    assert current["rollout"] == expected_rollout
    assert current["rollout"]["top_k"] == 0
    assert current["rollout"]["repetition_penalty"] == 1.0
    assert current["objective"]["group_size"] == 8
    assert current["training"]["learning_rate"] == 2e-5
    assert current["runtime_contract"]["detach_kv"] is False
    assert current["runtime_contract"]["truncated_bptt"] is False
    assert current["runtime_contract"]["saved_tensor_offload"] is False
    assert current["rollout"]["max_new_tokens"] == 512
    ablation.validate_sampling_ablation_contract(current)


def test_cleanup_is_only_at_two_graph_free_replay_boundaries():
    source = inspect.getsource(production.main)
    assert source.count("_graph_free_cuda_cleanup()") == 2
    old_policy_cleanup = source.index(
        "if SEMANTICS_PRESERVING_REPLAY_CUDA_CLEANUP:\n"
        "                _graph_free_cuda_cleanup()"
    )
    current_replay_start = source.index("optimizer.zero_grad(set_to_none=True)")
    assert old_policy_cleanup < current_replay_start
    backward = source.index("scaled.backward()")
    recorder_clear = source.index("recorder._previous_next_cache = None", backward)
    graph_deletion = source.index(
        "del current, blocks, loss, loss_grpo, kl_value, kl_contribution, scaled, recorder",
        recorder_clear,
    )
    post_backward_cleanup = source.index("_graph_free_cuda_cleanup()", graph_deletion)
    assert backward < recorder_clear < graph_deletion < post_backward_cleanup
    assert "empty_cache" not in inspect.getsource(NTPRolloutReplayer.score)


def test_loss_scaling_and_replay_order_are_unchanged():
    source = inspect.getsource(production.main)
    assert "for group_index, trace in enumerate(traces):" in source
    assert "scaled = loss / GROUP_SIZE" in source
    assert "scaled.backward()" in source
    assert "sorted(traces" not in source
    assert "detach_kv" not in inspect.getsource(NTPRolloutReplayer.score)
    launcher_source = inspect.getsource(ablation.main)
    assert "production.GROUP_SIZE = 8" in launcher_source


def test_launcher_enables_only_requested_memory_metadata_and_checkpoint_backend():
    source = inspect.getsource(ablation.main)
    assert "production.PERSIST_PRE_REPLAY_TRACE_METADATA = True" in source
    assert "production.SEMANTICS_PRESERVING_REPLAY_CUDA_CLEANUP = True" in source
    assert "production.PER_REPLAY_PEAK_MEMORY_DIAGNOSTICS = True" in source
    assert "production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True" in source
    assert "saved_tensor_offload" not in source


def test_checkpointing_is_scoped_only_to_current_policy_replay_and_backward():
    source = inspect.getsource(production.main)
    assert source.count("with functional_kv_layer_checkpointing(") == 1
    old_policy_replay = source.index("with torch.no_grad():\n                old_logps")
    checkpoint_context = source.index("with functional_kv_layer_checkpointing(")
    current_policy_score = source.index("current, blocks = replayer.score(", checkpoint_context)
    backward = source.index("scaled.backward()", current_policy_score)
    context_exit = source.index("checkpoint_report = dict(checkpoint_report)", backward)
    assert old_policy_replay < checkpoint_context < current_policy_score < backward < context_exit
    generation = source.index("generate_rollout_group(")
    assert generation < old_policy_replay < checkpoint_context


def test_only_t1p2_launcher_enables_checkpointing():
    assert production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING is False
    enabled = "production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True"
    assert enabled in inspect.getsource(ablation.main)
    unchanged_launchers = (
        "two_gpu_g8_loraonly_projectorfrozen_ntponly_production.py",
        "two_gpu_g8_loraonly_projectorfrozen_production.py",
        "two_gpu_g8_caseb_production.py",
    )
    for filename in unchanged_launchers:
        assert enabled not in (HERE / filename).read_text(encoding="utf-8")


def test_first_replay_runtime_contract_checks_all_layers_and_live_kv():
    source = inspect.getsource(production._assert_first_checkpointed_replay)
    assert "set(range(36))" in source
    assert 'layer.get("key_requires_grad")' in source
    assert 'layer.get("value_requires_grad")' in source
    assert 'layer.get("key_grad_fn") is not None' in source
    assert 'layer.get("value_grad_fn") is not None' in source
    assert '"no_kv_tensor_detached": all_kv_differentiable' in source


def test_production_launch_requires_passed_72_token_preflight():
    source = inspect.getsource(ablation.main)
    validator = inspect.getsource(ablation.validate_checkpoint_preflight_receipt)
    assert "--checkpoint-preflight-json" in inspect.getsource(ablation.parse_args)
    assert "production launch requires --checkpoint-preflight-json" in source
    assert '"passed_checkpoint_preflight"' in validator
    assert '"generated_token_count", -1)) != 72' in validator
    assert "PREFLIGHT_GPU0_PEAK_LIMIT_BYTES" in validator
    assert '"lora_gradient_tensors_finite", -1)) != 504' in validator


def test_peak_memory_diagnostics_are_scalar_and_reset_per_trajectory():
    source = inspect.getsource(production.main)
    replay_loop = source.index("for group_index, trace in enumerate(traces):")
    reset = source.index("_reset_peaks(devices)", replay_loop)
    backward = source.index("scaled.backward()", reset)
    capture = source.index("_replay_peak_memory_diagnostics(devices)", backward)
    assert replay_loop < reset < backward < capture
    diagnostic_source = inspect.getsource(production._replay_peak_memory_diagnostics)
    for field in (
        '"allocated"',
        '"reserved"',
        '"max_memory_allocated"',
        '"max_memory_reserved"',
    ):
        assert field in diagnostic_source
    assert ".detach(" not in diagnostic_source


def test_all_trace_metadata_is_materialized_and_persisted_before_replay():
    traces = [
        SimpleNamespace(
            generated_token_ids=list(range(index + 1)),
            blocks=[object()] * (index + 1),
            truncated=index == 7,
            stop_reason="max_token_budget" if index == 7 else "im_end_token",
            max_new_tokens=512,
            committed_final_box_norm_1000=(0, 0, 10, 10),
        )
        for index in range(8)
    ]
    components = [
        SimpleNamespace(
            format_valid=True,
            geometry_valid=True,
            has_unambiguous_committed_box=True,
            parse_error=None,
            reward_branch="ntp_only",
        )
        for _ in range(8)
    ]
    metadata = production._pre_replay_trace_metadata(traces, components)
    assert len(metadata) == 8
    assert [row["group_index"] for row in metadata] == list(range(8))
    assert metadata[-1] == {
        "group_index": 7,
        "generated_token_count": 8,
        "block_count": 8,
        "truncated": True,
        "stop_reason": "max_token_budget",
        "max_new_tokens": 512,
        "format_valid": True,
        "geometry_valid": True,
        "has_unambiguous_committed_box": True,
        "valid_native_box": True,
        "parse_error": None,
        "reward_branch": "ntp_only",
    }
    source = inspect.getsource(production.main)
    persisted = source.index("_append_jsonl(\n                    pre_replay_metadata_path")
    old_policy_replay = source.index("with torch.no_grad():\n                old_logps")
    current_policy_replay = source.index("for group_index, trace in enumerate(traces):")
    assert persisted < old_policy_replay < current_policy_replay


def test_pure_ntp_guard_reachable_max_is_512():
    trace = SimpleNamespace(
        generated_token_ids=[1] * 512,
        truncated=True,
        stop_reason="max_token_budget",
        reward_branch="none",
        decoder_path=production.NTP_ONLY_DECODER_PATH,
        max_reachable_generated_length=512,
    )
    component = SimpleNamespace(parse_error="no native box", total_reward=0.0)
    result = production._fresh_start_trajectory_guard_predicates(
        trace, component, max_new_tokens=512, block_size=6
    )
    assert result["maximum_reachable_generated_length"] == 512
    assert result["reachability_mode"] == "pure_ntp"
    assert result["matches"] is True


def _component(iou, spatial, fmt, semantic, box):
    return SimpleNamespace(
        final_iou=iou,
        spatial_reward=spatial,
        format_reward=fmt,
        semantic_reward=semantic,
        total_reward=fmt + spatial + semantic,
        format_valid=bool(fmt),
        geometry_valid=box is not None,
        has_unambiguous_committed_box=box is not None,
    )


def _trace(box):
    return SimpleNamespace(committed_final_box_norm_1000=box)


def test_group_and_window_spatial_density_metrics():
    boxes = [
        (0, 0, 1000, 1000),
        (0, 0, 500, 500),
        (100, 100, 600, 600),
        None, None, None, None, None,
    ]
    ious = [0.6, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]
    components = [
        _component(iou, float(iou > 0.5), float(box is not None), index / 10.0, box)
        for index, (iou, box) in enumerate(zip(ious, boxes))
    ]
    group = build_group_spatial_diagnostic(
        [_trace(box) for box in boxes], components, expected_group_size=8
    )
    assert group["raw_ious"] == ious
    assert group["binary_spatial_rewards"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert group["group_spatial_class"] == "mixed_spatial_0_1"
    assert group["valid_box_rate"] == 3 / 8
    assert group["near_full_box_rate"] == 1 / 8
    assert group["reward_nonzero_variance"] == {
        "format": True, "spatial": True, "semantic": True, "total_reward": True,
    }
    assert group["mean_pairwise_bbox_iou_within_group"] is not None
    assert group["mean_coordinate_std_norm_1000"] is not None

    aggregate = aggregate_spatial_density_window(
        [group], optimizer_step_end=25, interval=25
    )
    density = aggregate["spatial_density"]
    assert density["fraction_mixed_spatial_0_1"] == 1.0
    assert density["rollout_p_iou_gt_0_5"] == 1 / 8
    assert density["p_group_contains_ge_1_iou_gt_0_5"] == 1.0
    assert aggregate["reference_context"]["hard_pass_threshold_applied"] is False
    assert math.isclose(
        aggregate["reward_variation"]["fraction_groups_nonzero_variance"]["semantic"],
        1.0,
    )


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("{} sampling-ablation contract tests passed".format(len(tests)))
