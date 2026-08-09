"""CPU/static checks for G=8 Case-B experiment preparation."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g4_grpo_multistep_smoke as production  # noqa: E402
import two_gpu_g8_caseb_production as g8  # noqa: E402
from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.runtime import load_resolved_config  # noqa: E402


def configs():
    return load_resolved_config(HERE / "rl/chestxray8_grpo_hybrid_native.yaml"), g8.resolve_experiment_config(g8.DEFAULT_SPEC)


def test_scientific_contract_is_unchanged():
    base, experiment = configs()
    assert experiment["prompt"] == base["prompt"]
    assert experiment["rollout"] == base["rollout"]
    assert experiment["rewards"] == base["rewards"]
    assert experiment["runtime_contract"] == base["runtime_contract"]
    assert experiment["model"]["projector_trainable"] is True
    assert experiment["model"]["lora"] == base["model"]["lora"]
    assert experiment["objective"]["ppo_clip_epsilon"] == base["objective"]["ppo_clip_epsilon"]
    assert experiment["objective"]["old_policy_sync_interval_optimizer_steps"] == 1
    assert experiment["objective"]["loss_total"] == "L_GRPO"
    assert experiment["objective"]["reference_kl"] == {"enabled": False}


def test_trainable_and_schedule_contract():
    _, config = configs()
    g8.validate_g8_caseb_contract(config)
    assert production.EXPECTED_LORA_TENSORS == 504
    assert production.EXPECTED_PROJECTOR_TENSORS == 6
    assert production.EXPECTED_TRAINABLE_TENSORS == 510
    assert config["objective"]["group_size"] == 8
    assert config["training"]["learning_rate"] == 2e-5
    assert config["training"]["projector_learning_rate"] == 1e-5
    assert config["training"]["max_grad_norm"] == 1.0
    assert config["training"]["checkpoint_interval"] == 25


def test_per_trajectory_accumulation_equals_vector_mean_for_g4_and_g8():
    for size in (4, 8):
        rewards = torch.linspace(-0.7, 1.3, size)
        advantages = group_relative_advantages(rewards)
        old = torch.linspace(-3.0, -1.0, size)
        current = old + torch.linspace(-0.1, 0.1, size)
        vector = grpo_clipped_loss(current, old, advantages, clip_epsilon=0.2)
        accumulated = sum(
            grpo_clipped_loss(current[i:i + 1], old[i:i + 1], advantages[i:i + 1], clip_epsilon=0.2) / size
            for i in range(size)
        )
        assert torch.equal(vector, accumulated) or torch.allclose(vector, accumulated, atol=1e-7, rtol=1e-7)
        assert abs(float(advantages.mean())) < 1e-6
        assert abs(float(advantages.std(unbiased=False)) - 1.0) < 1e-6


def test_runner_has_no_fixed_four_in_objective_normalization():
    source = inspect.getsource(production.main)
    assert "scaled = loss / GROUP_SIZE" in source
    assert "mean_grpo_loss = float(sum(group_grpo_losses) / GROUP_SIZE)" in source
    assert "mean_total_loss = float(sum(group_total_losses) / GROUP_SIZE)" in source
    assert "group_relative_advantages(reward_values)" in source


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("{} G=8 Case-B static tests passed".format(len(tests)))
