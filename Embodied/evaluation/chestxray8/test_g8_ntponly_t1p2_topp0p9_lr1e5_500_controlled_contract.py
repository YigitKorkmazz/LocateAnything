"""Static contracts for the validation-gated LR=1e-5 long run."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import internal_validation_lr1e5_500_guard as guard  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr1e5_50 as diagnostic  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr1e5_500_controlled as controlled  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402


def _diff(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        out = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                out.add(child)
            else:
                out.update(_diff(left[key], right[key], child))
        return out
    return set() if left == right else {prefix}


def test_only_duration_and_checkpoint_schedule_differ_scientifically():
    short = resolve_experiment_config(diagnostic.DEFAULT_SPEC)
    long = resolve_experiment_config(controlled.DEFAULT_SPEC)
    assert _diff(short, long) == {
        "_config_path",
        "experiment",
        "training.max_optimizer_steps",
        "training.checkpoint_interval",
    }
    controlled.validate_long_run_contract(long)
    assert long["training"]["learning_rate"] == 1e-5
    assert long["runtime_contract"]["detach_kv"] is False


def test_no_monolithic_500_launch_and_each_resume_requires_guard():
    source = inspect.getsource(controlled.main)
    assert 'choices=TARGET_STEPS' in inspect.getsource(controlled.parse_args)
    assert '"--max-optimizer-steps", str(target)' in source
    assert "resumed segments require checkpoint+guard" in source
    assert "validate_guard_receipt(" in source
    assert '"next_action_required": "pinned_internal_validation_then_guard"' in source


def test_trainable_lr_backend_and_diagnostics_are_locked():
    source = inspect.getsource(controlled.main)
    for required in (
        "production.GROUP_SIZE = 8",
        "production.EXPECTED_LORA_TENSORS = 504",
        "production.EXPECTED_PROJECTOR_TENSORS = 0",
        "production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True",
        "lr_ablation._optimizer_contract_validator(1e-5)",
        '"--checkpoint-interval", "100"',
    ):
        assert required in source
    diagnostic_source = inspect.getsource(
        controlled.production.aggregate_spatial_density_window
    )
    for field in (
        "fraction_mixed_spatial_0_1",
        "rollout_p_iou_gt_0_5",
        "fraction_all_spatial_0",
        "mean_group_max_raw_iou",
        "mean_within_group_raw_iou_range",
        "mean_pairwise_bbox_iou_within_group",
        "mean_coordinate_std_norm_1000",
        "valid_box_rate",
        "near_full_box_rate",
        "mean_predicted_box_area_fraction",
        "fraction_groups_nonzero_variance",
    ):
        assert field in diagnostic_source


def test_checkpoint_backend_remains_current_policy_only():
    source = inspect.getsource(controlled.production.main)
    assert source.count("with functional_kv_layer_checkpointing(") == 1
    old_policy = source.index("with torch.no_grad():\n                old_logps")
    checkpoint = source.index("with functional_kv_layer_checkpointing(")
    current = source.index("current, blocks = replayer.score(", checkpoint)
    backward = source.index("scaled.backward()", current)
    assert old_policy < checkpoint < current < backward


def test_guard_implements_exact_procedural_rules_and_no_heldout_access():
    source = inspect.getsource(guard.main)
    assert 'current["near_full_box_rate"] > 0.60' in source
    assert 'current["mean_iou"] <= previous_mean' in source
    assert source.count('< base_mean_iou') == 2
    assert 'aggregate.get("heldout_test_used") is not False' in source
    assert "test_pairs_seed42" not in source
    assert "optimizer" not in source
    assert "train(" not in source

