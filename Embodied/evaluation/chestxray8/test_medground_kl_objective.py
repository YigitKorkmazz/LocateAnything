#!/usr/bin/env python3
"""Focused CPU oracles for the MedGround-compatible Hybrid KL objective."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.grpo import grpo_clipped_loss  # noqa: E402
from rl.hybrid_rl import HybridRolloutReplayer  # noqa: E402
from rl.medground_kl import (  # noqa: E402
    clipped_grpo_with_medground_kl,
    masked_token_mean,
    medground_per_token_kl,
)
from rl.pbd_rl import BlockTrace, SlotTrace  # noqa: E402


def _loss(policy, reference, *, advantage=0.7, beta=0.04):
    old = policy.detach().sum().clone()
    return clipped_grpo_with_medground_kl(
        policy.sum(),
        old,
        torch.tensor(advantage),
        policy,
        reference,
        beta=beta,
        clip_epsilon=0.2,
    )


def test_beta_zero_reproduces_accepted_kl_free_loss() -> None:
    policy = torch.tensor([-1.2, -0.8, -2.0], requires_grad=True)
    reference = torch.tensor([-1.0, -1.1, -1.8])
    result = _loss(policy, reference, beta=0.0)
    expected = grpo_clipped_loss(
        policy.sum().reshape(1),
        policy.detach().sum().reshape(1),
        torch.tensor([0.7]),
        clip_epsilon=0.2,
    )
    assert torch.equal(result.total_loss, expected)
    assert result.kl_loss_contribution.item() == 0.0


def test_policy_equals_reference_gives_zero_kl() -> None:
    policy = torch.tensor([-0.5, -1.5, -2.5], requires_grad=True)
    result = _loss(policy, policy.detach().clone())
    assert torch.equal(result.kl_value, torch.zeros_like(result.kl_value))


def test_reference_receives_no_gradients() -> None:
    policy = torch.tensor([-1.0, -1.4], requires_grad=True)
    reference = torch.tensor([-1.3, -1.1], requires_grad=True)
    result = _loss(policy, reference, advantage=0.0)
    result.total_loss.backward()
    assert policy.grad is not None
    assert reference.grad is None


def test_kl_sign_is_positive_penalty() -> None:
    policy = torch.tensor([-0.2, -2.0], requires_grad=True)
    reference = torch.tensor([-1.1, -0.7])
    result = _loss(policy, reference, advantage=0.0, beta=0.04)
    assert result.kl_value.item() > 0.0
    assert result.kl_loss_contribution.item() > 0.0
    assert torch.equal(result.total_loss, result.kl_loss_contribution)


def test_kl_mask_selects_exact_intended_trajectory_tokens() -> None:
    values = torch.tensor([1.0, 10.0, 3.0, 20.0])
    mask = torch.tensor([True, False, True, False])
    assert masked_token_mean(values, mask).item() == 2.0
    policy = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    reference = torch.tensor([-1.0, -1.5, -3.0, -2.0])
    per_token = medground_per_token_kl(policy, reference)
    assert torch.equal(masked_token_mean(per_token, mask), per_token[[0, 2]].mean())


def test_hybrid_full_trajectory_mask_includes_rejected_proposal_slots() -> None:
    slots = [
        SlotTrace(
            slot_index=index,
            action_token_id=100 + index,
            support_kind="full_vocab",
            log_prob_old=-1.0,
            support_size=200,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
        )
        for index in range(6)
    ]
    block = BlockTrace(
        block_index=0,
        prefix_length=10,
        cache_length_before=0,
        cache_length_after=10,
        block_type="error_box_prefix",
        position_ids=[],
        input_window_ids=[],
        action_token_ids=[100, 101],
        slots=slots,
        scored_for_grpo=True,
        source="pbd",
        rejected_proposal_token_ids=[100 + index for index in range(6)],
    )
    values = [torch.tensor(-float(index + 1)) for index in range(6)]
    replayer = HybridRolloutReplayer.__new__(HybridRolloutReplayer)

    def fake_score(_trace, **kwargs):
        kwargs["on_scored_token_logps"](block, values)
        total = torch.stack(values).sum()
        return total, [total]

    replayer.score = fake_score
    total, _blocks, tokens, metadata = replayer.score_with_token_logprobs(
        object(), pixel_values=None, input_ids=None, image_grid_hws=None
    )
    assert torch.equal(total, tokens.sum())
    assert len(metadata) == 6
    assert [item["mask"] for item in metadata] == [1] * 6
    assert sum(item["committed_to_generated_stream"] for item in metadata) == 2
    assert sum(item["rejected_pbd_proposal_token"] for item in metadata) == 4


def test_zero_advantage_group_still_computes_kl_when_beta_positive() -> None:
    policy = torch.tensor([-0.4, -1.7], requires_grad=True)
    reference = torch.tensor([-1.0, -1.0])
    result = _loss(policy, reference, advantage=0.0, beta=0.04)
    assert result.grpo_loss.item() == 0.0
    assert result.kl_value.item() > 0.0
    result.total_loss.backward()
    assert policy.grad is not None and torch.count_nonzero(policy.grad).item() > 0


def test_optimizer_contains_exactly_510_unique_trainable_tensors() -> None:
    parameters = [torch.nn.Parameter(torch.tensor(float(i))) for i in range(510)]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    included = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(included) == 510
    assert len({id(parameter) for parameter in included}) == 510
    assert all(parameter.requires_grad for parameter in included)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"MEDGROUND KL OBJECTIVE TESTS PASSED ({len(tests)})")
