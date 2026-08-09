#!/usr/bin/env python3
"""CPU equivalence: stacked G=4 GRPO vs sequential (loss_i/G).backward()."""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.grpo_train_step import (  # noqa: E402
    CHECKPOINTING_INCOMPATIBLE_MSG,
    assert_grpo_loss_depends_only_on_trajectory_logprob,
    assert_initialization_ratios,
    assert_production_cached_replay_checkpointing_disabled,
)
from rl.replay_memory import eval_mode_layer_checkpointing  # noqa: E402


ABS_TOL = 1e-5
REL_TOL = 1e-5
CLIP_EPSILON = 0.2
GROUP_SIZE = 4


@dataclass
class ToyTrace:
    block_features: List[torch.Tensor]


class ToyPolicy(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.bias = nn.Parameter(torch.zeros(()))

    def block_logprob(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(feature @ self.weight + self.bias)

    def trajectory_logprob(self, trace: ToyTrace) -> torch.Tensor:
        blocks = [self.block_logprob(feature) for feature in trace.block_features]
        return torch.stack(blocks).sum()


def _make_fixed_traces(seed: int = 0) -> Tuple[List[ToyTrace], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    traces: List[ToyTrace] = []
    for _ in range(GROUP_SIZE):
        blocks = [
            torch.randn(8, generator=generator),
            torch.randn(8, generator=generator),
            torch.randn(8, generator=generator),
        ]
        traces.append(ToyTrace(block_features=blocks))
    rewards = torch.tensor([0.0, 1.0, 0.5, 0.25], dtype=torch.float32)
    advantages = group_relative_advantages(rewards)
    old_logps = torch.tensor([-1.2, -0.8, -1.0, -1.5], dtype=torch.float32)
    return traces, advantages, old_logps


def _clone_state(module: nn.Module):
    return copy.deepcopy(module.state_dict())


def _param_vector(module: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _grad_vector(module: nn.Module) -> torch.Tensor:
    chunks = []
    for parameter in module.parameters():
        if parameter.grad is None:
            chunks.append(torch.zeros_like(parameter).reshape(-1))
        else:
            chunks.append(parameter.grad.detach().reshape(-1))
    return torch.cat(chunks)


def baseline_stacked_step(
    policy: ToyPolicy,
    traces: Sequence[ToyTrace],
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
) -> dict:
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    current = torch.stack([policy.trajectory_logprob(trace) for trace in traces])
    loss = grpo_clipped_loss(
        current,
        old_logps,
        advantages,
        clip_epsilon=CLIP_EPSILON,
    )
    ratios = torch.exp(current.detach() - old_logps)
    loss.backward()
    grads = _grad_vector(policy)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "current_logps": current.detach().tolist(),
        "ratios": ratios.tolist(),
        "grads": grads,
        "params": _param_vector(policy),
    }


def sequential_loss_over_g_step(
    policy: ToyPolicy,
    traces: Sequence[ToyTrace],
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
) -> dict:
    """Exact memory-safe baseline: (ell_i / G).backward() per trajectory."""
    assert_grpo_loss_depends_only_on_trajectory_logprob()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    current_logps: List[float] = []
    ratios: List[float] = []
    losses: List[float] = []

    for index, trace in enumerate(traces):
        current = policy.trajectory_logprob(trace)
        rollout_loss = grpo_clipped_loss(
            current.reshape(1),
            old_logps[index].reshape(1),
            advantages[index].reshape(1),
            clip_epsilon=CLIP_EPSILON,
        )
        (rollout_loss / float(GROUP_SIZE)).backward()
        current_logps.append(float(current.detach()))
        ratios.append(float(torch.exp(current.detach() - old_logps[index])))
        losses.append(float(rollout_loss.detach()))
        del current, rollout_loss

    grads = _grad_vector(policy)
    optimizer.step()
    mean_loss = float(sum(losses) / GROUP_SIZE)
    return {
        "loss": mean_loss,
        "current_logps": current_logps,
        "ratios": ratios,
        "grads": grads,
        "params": _param_vector(policy),
    }


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.clamp(a.abs().max(), min=1e-12)
    return float((a - b).abs().max() / denom)


def test_grpo_loss_has_no_hidden_or_block_logit_terms() -> None:
    assert_grpo_loss_depends_only_on_trajectory_logprob()


def test_sequential_loss_over_g_matches_stacked_baseline() -> None:
    traces, advantages, old_logps = _make_fixed_traces(seed=7)
    torch.manual_seed(123)
    baseline_policy = ToyPolicy()
    init_state = _clone_state(baseline_policy)
    baseline = baseline_stacked_step(
        baseline_policy, traces, old_logps, advantages
    )

    optimized_policy = ToyPolicy()
    optimized_policy.load_state_dict(init_state)
    optimized = sequential_loss_over_g_step(
        optimized_policy, traces, old_logps, advantages
    )

    loss_diff = abs(baseline["loss"] - optimized["loss"])
    logp_diffs = [
        abs(a - b)
        for a, b in zip(baseline["current_logps"], optimized["current_logps"])
    ]
    ratio_diffs = [
        abs(a - b) for a, b in zip(baseline["ratios"], optimized["ratios"])
    ]
    grad_max_abs = float((baseline["grads"] - optimized["grads"]).abs().max())
    grad_rel = _rel_diff(baseline["grads"], optimized["grads"])
    update_max_abs = float((baseline["params"] - optimized["params"]).abs().max())
    update_rel = _rel_diff(baseline["params"], optimized["params"])

    report = {
        "total_loss_difference": loss_diff,
        "per_rollout_logprob_max_difference": max(logp_diffs),
        "ppo_ratio_max_difference": max(ratio_diffs),
        "gradient_max_absolute_difference": grad_max_abs,
        "gradient_relative_difference": grad_rel,
        "optimizer_update_max_absolute_difference": update_max_abs,
        "optimizer_update_relative_difference": update_rel,
        "baseline_loss": baseline["loss"],
        "optimized_loss": optimized["loss"],
        "clipping_scope": "trajectory",
        "backend": "live_production_cached_autograd",
    }
    print("EQUIVALENCE_REPORT", report)

    assert loss_diff <= ABS_TOL, report
    assert max(logp_diffs) <= ABS_TOL, report
    assert max(ratio_diffs) <= ABS_TOL, report
    assert grad_max_abs <= ABS_TOL or grad_rel <= REL_TOL, report
    assert update_max_abs <= ABS_TOL or update_rel <= REL_TOL, report


def test_initialization_ratios_helper() -> None:
    assert_initialization_ratios([-1.0, -2.0], [-1.0, -2.0])
    try:
        assert_initialization_ratios([-1.1], [-1.0])
        raise AssertionError("expected initialization assert failure")
    except RuntimeError:
        pass


def test_checkpointed_cached_replay_rejected_before_forward() -> None:
    """sequential_production_cached + gradient_checkpointing must hard-fail."""
    assert_production_cached_replay_checkpointing_disabled(
        replay_backend="sequential_production_cached",
        gradient_checkpointing=False,
    )
    try:
        assert_production_cached_replay_checkpointing_disabled(
            replay_backend="sequential_production_cached",
            gradient_checkpointing=True,
        )
        raise AssertionError("expected checkpointing rejection")
    except RuntimeError as exc:
        message = str(exc)
        assert "incompatible with gradient checkpointing" in message
        assert "past_key_values" in message or "KV-cache" in message
        assert "Bfix" in message
        assert CHECKPOINTING_INCOMPATIBLE_MSG[:48] in message

    class _Dummy:
        training = False

    try:
        with eval_mode_layer_checkpointing(_Dummy(), enabled=True):
            raise AssertionError("expected layer-checkpoint refusal")
    except RuntimeError as exc:
        message = str(exc)
        assert "mutable past_key_values" in message
        assert "checkpoint-safe" in message
        assert "two-pass" in message


def test_ppo_clip_is_trajectory_level_not_per_block() -> None:
    old = torch.tensor(0.0)
    advantage = torch.tensor(1.0)
    block_a = torch.tensor(2.0, requires_grad=True)
    block_b = torch.tensor(-0.5, requires_grad=True)
    traj = block_a + block_b
    traj_loss = grpo_clipped_loss(
        traj.reshape(1), old.reshape(1), advantage.reshape(1), clip_epsilon=0.2
    )
    illegal = 0.5 * (
        grpo_clipped_loss(
            block_a.reshape(1), old.reshape(1), advantage.reshape(1), clip_epsilon=0.2
        )
        + grpo_clipped_loss(
            block_b.reshape(1), old.reshape(1), advantage.reshape(1), clip_epsilon=0.2
        )
    )
    assert not math.isclose(
        float(traj_loss.detach()), float(illegal.detach()), rel_tol=1e-5, abs_tol=1e-5
    )


def main() -> None:
    test_grpo_loss_has_no_hidden_or_block_logit_terms()
    test_initialization_ratios_helper()
    test_checkpointed_cached_replay_rejected_before_forward()
    test_ppo_clip_is_trajectory_level_not_per_block()
    test_sequential_loss_over_g_matches_stacked_baseline()
    print("ALL_CPU_EQUIVALENCE_TESTS_PASSED")


if __name__ == "__main__":
    main()
