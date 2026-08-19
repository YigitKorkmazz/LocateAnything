"""CPU/static checks for the G=8 projector-frozen LoRA-only ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval_two_gpu_hybrid_grpo_checkpoints as evaluator  # noqa: E402
import two_gpu_g4_grpo_multistep_smoke as production  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_production as ablation  # noqa: E402
from rl.runtime import load_resolved_config  # noqa: E402
from train_chestxray8_sft import build_optimizer  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402


def configs():
    base = load_resolved_config(HERE / "rl/chestxray8_grpo_hybrid_native.yaml")
    experiment = resolve_experiment_config(ablation.DEFAULT_SPEC)
    return base, experiment


def test_only_projector_trainability_and_ablation_schedule_change():
    base, experiment = configs()
    for section in ("prompt", "rollout", "rewards", "runtime_contract", "evaluation"):
        assert experiment[section] == base[section]
    assert experiment["model"]["lora"] == base["model"]["lora"]
    assert experiment["model"]["projector_trainable"] is False
    assert experiment["objective"]["group_size"] == 8
    assert experiment["objective"]["loss_total"] == "L_GRPO"
    assert experiment["objective"]["reference_kl"] == {"enabled": False}
    assert experiment["policy_state"]["synchronized_parameters"] == ["lora"]
    assert experiment["training"]["learning_rate"] == 2e-5
    assert experiment["training"]["checkpoint_interval"] == 25
    assert experiment["training"]["max_optimizer_steps"] == 100
    ablation.validate_loraonly_projectorfrozen_contract(experiment)


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(3))
        self.mlp1 = torch.nn.Linear(3, 2)
        self.base_weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.mlp1.requires_grad_(False)


def _with_tiny_expected_contract(callback):
    previous = (
        production.EXPECTED_LORA_TENSORS,
        production.EXPECTED_PROJECTOR_TENSORS,
        production.EXPECTED_TRAINABLE_TENSORS,
    )
    try:
        production.EXPECTED_LORA_TENSORS = 1
        production.EXPECTED_PROJECTOR_TENSORS = 0
        production.EXPECTED_TRAINABLE_TENSORS = 1
        callback()
    finally:
        (
            production.EXPECTED_LORA_TENSORS,
            production.EXPECTED_PROJECTOR_TENSORS,
            production.EXPECTED_TRAINABLE_TENSORS,
        ) = previous


def test_startup_and_optimizer_contract_exclude_projector():
    def check():
        model = _TinyPolicy()
        startup = production._startup_trainable_contract_report(model)
        assert startup["total_trainable_parameter_tensors"] == 1
        assert startup["total_trainable_parameter_count"] == 3
        assert startup["lora_trainable_tensor_count"] == 1
        assert startup["projector_trainable_tensor_count"] == 0
        assert startup["unexpected_trainable_parameter_names"] == []
        optimizer = build_optimizer(
            model,
            lr=2e-5,
            projector_lr=1e-5,
            weight_decay=0.0,
            use_8bit_adam=False,
        )
        report = production._optimizer_contract_report(model, optimizer)
        assert report["optimizer_group_count"] == 1
        assert report["optimizer_parameter_tensor_count"] == 1
        assert report["optimizer_projector_tensor_count"] == 0
        assert report["optimizer_group_learning_rates"] == [2e-5]

    _with_tiny_expected_contract(check)


def test_lora_only_gradient_sanity_reports_and_fails_loudly():
    def check():
        model = _TinyPolicy()
        model.lora_A.grad = torch.tensor([1.0, 2.0, 3.0])
        report = production._grad_report(model)
        assert report["lora_tensors_with_grad"] == 1
        assert report["lora_tensors_with_finite_grad"] == 1
        assert report["projector_tensors_with_grad"] == 0
        assert report["frozen_non_lora_tensors_with_grad"] == 0
        assert report["lora_grad_norm_summary"]["min"] > 0.0
        production._assert_lora_only_gradient_sanity(report)

        model.mlp1.weight.grad = torch.ones_like(model.mlp1.weight)
        try:
            production._assert_lora_only_gradient_sanity(production._grad_report(model))
        except RuntimeError as error:
            assert "projector parameter received a gradient" in str(error)
        else:
            raise AssertionError("projector gradient did not fail loudly")

        model.mlp1.weight.grad = None
        model.base_weight.grad = torch.ones_like(model.base_weight)
        try:
            production._assert_lora_only_gradient_sanity(production._grad_report(model))
        except RuntimeError as error:
            assert "frozen non-LoRA parameter received a gradient" in str(error)
        else:
            raise AssertionError("frozen base gradient did not fail loudly")

        model.base_weight.grad = None
        model.lora_A.grad[0] = float("nan")
        try:
            production._assert_lora_only_gradient_sanity(production._grad_report(model))
        except RuntimeError as error:
            assert "LoRA gradient contains NaN/Inf" in str(error)
        else:
            raise AssertionError("nonfinite LoRA gradient did not fail loudly")

    _with_tiny_expected_contract(check)


def test_validation_reports_mean_and_median_valid_box_area():
    rows = []
    for area, valid in ((0.1, 1.0), (0.3, 1.0), (0.9, 1.0), (0.0, 0.0)):
        rows.append(
            {
                "format_valid": valid,
                "geometry_valid": valid,
                "valid_native_box": valid,
                "malformed_or_no_box": 1.0 - valid,
                "exactly_one_box": valid,
                "pbd_branch": 1.0,
                "ntp_branch": 0.0,
                "none_branch": 0.0,
                "iou": 0.2,
                "iou_at_025": 0.0,
                "iou_gt_0_5": 0.0,
                "semantic_reward": 0.1,
                "near_full_image": 0.0,
                "box_area_norm01": area,
                "total_reward": 1.1,
                "malformed_output": 1.0 - valid,
                "generated_token_count": 12.0,
                "truncated": 0.0,
                "committed_branch": "pbd",
            }
        )
    metrics = evaluator._aggregate(rows, replicates=20, seed=7)["metrics"]
    assert abs(metrics["mean_box_area_norm01_among_valid"]["point"] - (1.3 / 3.0)) < 1e-12
    assert metrics["median_box_area_norm01_among_valid"]["point"] == 0.3


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("{} projector-frozen ablation tests passed".format(len(tests)))
