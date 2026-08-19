"""Static contracts for the two learning-rate-only diagnostics."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ntp_checkpoint_replay_lr_ablation_oracle as lr_oracle  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr1e5_50 as lr1  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr5e6_50 as lr5  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr_ablation as ablation  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402


def _diff(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in set(left) | set(right):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_diff(left[key], right[key], child))
        return paths
    return set() if left == right else {prefix}


def test_resolved_configs_have_exact_allowlisted_differences():
    reference = resolve_experiment_config(ablation.REFERENCE_SPEC)
    cases = ((lr1, 1e-5), (lr5, 5e-6))
    for module, expected_lr in cases:
        current = resolve_experiment_config(module.DEFAULT_SPEC)
        assert _diff(reference, current) == {
            "_config_path",
            "experiment",
            "training.learning_rate",
            "training.max_optimizer_steps",
        }
        ablation.validate_lr_ablation_contract(
            current,
            expected_lr=expected_lr,
            expected_run_name=module.RUN_NAME,
        )
    first = resolve_experiment_config(lr1.DEFAULT_SPEC)
    second = resolve_experiment_config(lr5.DEFAULT_SPEC)
    assert _diff(first, second) == {
        "_config_path", "experiment", "training.learning_rate"
    }


def test_explicit_startup_contract_and_diagnostic_schedule():
    source = inspect.getsource(ablation.main_for_run)
    for required in (
        "production.GROUP_SIZE = 8",
        "production.EXPECTED_LORA_TENSORS = 504",
        "production.EXPECTED_PROJECTOR_TENSORS = 0",
        "production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True",
        '"--max-optimizer-steps", "50"',
        '"--checkpoint-interval", "25"',
        '"--seed", "42"',
        '"--require-fresh-start"',
    ):
        assert required in source
    assert ablation._line_count(ablation.OPTIMIZATION) == 710
    assert ablation._line_count(ablation.VALIDATION) == 80


def test_optimizer_has_one_exact_lr_group_per_run():
    source = inspect.getsource(ablation._optimizer_contract_validator)
    assert 'report["optimizer_group_learning_rates"] != [expected_lr]' in source
    assert '"all_groups_exact": True' in source


def test_existing_functional_kv_backend_semantics_are_reused_unchanged():
    assert lr_oracle.oracle.__name__ == "ntp_checkpoint_replay_oracle"
    adapter_source = inspect.getsource(lr_oracle)
    assert "oracle.validate_sampling_ablation_contract = validate_known_lr_config" in adapter_source
    assert "oracle.main()" in adapter_source
    assert "functional_kv_layer_checkpointing" not in adapter_source
    oracle_source = inspect.getsource(lr_oracle.oracle._run_replay)
    assert "functional_kv_layer_checkpointing(model, enabled=True)" in oracle_source


def test_checkpointing_stays_scoped_to_current_policy_replay():
    production_source = inspect.getsource(ablation.production.main)
    assert production_source.count("with functional_kv_layer_checkpointing(") == 1
    old_policy = production_source.index(
        "with torch.no_grad():\n                old_logps"
    )
    checkpoint = production_source.index("with functional_kv_layer_checkpointing(")
    current = production_source.index("current, blocks = replayer.score(", checkpoint)
    backward = production_source.index("scaled.backward()", current)
    assert old_policy < checkpoint < current < backward


def test_required_spatial_density_fields_are_unchanged():
    source = inspect.getsource(ablation.production.aggregate_spatial_density_window)
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
        assert field in source


def test_heldout_evaluation_is_not_reachable_from_launchers():
    source = inspect.getsource(ablation.main_for_run)
    assert "test_pairs_seed42" not in source
    assert "eval_two_gpu_hybrid_grpo_checkpoints" not in source

