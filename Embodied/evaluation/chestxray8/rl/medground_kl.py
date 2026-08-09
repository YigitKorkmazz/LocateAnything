"""MedGround-R1's sampled per-token KL estimator and loss composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rl.grpo import grpo_clipped_loss


MEDGROUND_KL_ESTIMATOR = "exp(ref_logp-policy_logp)-(ref_logp-policy_logp)-1"


@dataclass(frozen=True)
class MedGroundKLLoss:
    grpo_loss: torch.Tensor
    kl_value: torch.Tensor
    kl_loss_contribution: torch.Tensor
    total_loss: torch.Tensor
    ppo_ratio: torch.Tensor
    policy_logp: torch.Tensor
    reference_logp: torch.Tensor
    token_count: int


def medground_per_token_kl(
    policy_token_logps: torch.Tensor,
    reference_token_logps: torch.Tensor,
) -> torch.Tensor:
    """Return the exact nonnegative sampled KL estimator used by MedGround-R1.

    In expectation over actions sampled from the policy, this estimates
    ``KL(policy || reference)``. Reference values are always treated as
    constants; callers must additionally score the reference under no-grad.
    """
    if policy_token_logps.shape != reference_token_logps.shape:
        raise ValueError("policy/reference token log-probability shapes differ")
    if policy_token_logps.ndim != 1:
        raise ValueError("MedGround KL expects one-dimensional token log-probabilities")
    if policy_token_logps.numel() == 0:
        raise ValueError("MedGround KL requires at least one scored trajectory token")
    delta = reference_token_logps.detach().to(
        device=policy_token_logps.device,
        dtype=policy_token_logps.dtype,
    ) - policy_token_logps
    return torch.exp(delta) - delta - 1.0


def masked_token_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("token values must be one-dimensional")
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    if mask.shape != values.shape:
        raise ValueError("token mask shape differs from token values")
    mask = mask.to(device=values.device, dtype=torch.bool)
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("token mask selects no trajectory tokens")
    return values.masked_select(mask).mean()


def clipped_grpo_with_medground_kl(
    current_trajectory_logp: torch.Tensor,
    old_trajectory_logp: torch.Tensor,
    advantage: torch.Tensor,
    policy_token_logps: torch.Tensor,
    reference_token_logps: torch.Tensor,
    *,
    beta: float,
    clip_epsilon: float,
    token_mask: Optional[torch.Tensor] = None,
) -> MedGroundKLLoss:
    """Compose validated clipped trajectory GRPO with MedGround's KL penalty."""
    beta = float(beta)
    if beta < 0.0:
        raise ValueError("KL beta must be nonnegative")
    if not torch.isfinite(torch.tensor(beta)):
        raise ValueError("KL beta must be finite")
    policy_total = policy_token_logps.sum()
    if not torch.allclose(
        policy_total.detach().float(),
        current_trajectory_logp.detach().float(),
        atol=1e-4,
        rtol=1e-5,
    ):
        raise RuntimeError("token policy log-probabilities do not sum to trajectory logp")
    grpo = grpo_clipped_loss(
        current_trajectory_logp.reshape(1),
        old_trajectory_logp.detach().reshape(1).to(
            current_trajectory_logp.device, torch.float32
        ),
        advantage.detach().reshape(1),
        clip_epsilon=clip_epsilon,
    )
    token_kl = medground_per_token_kl(policy_token_logps, reference_token_logps)
    kl_value = masked_token_mean(token_kl, token_mask)
    contribution = kl_value * beta
    total = grpo + contribution
    return MedGroundKLLoss(
        grpo_loss=grpo,
        kl_value=kl_value,
        kl_loss_contribution=contribution,
        total_loss=total,
        ppo_ratio=torch.exp(
            current_trajectory_logp.detach().float()
            - old_trajectory_logp.detach().float().to(current_trajectory_logp.device)
        ),
        policy_logp=current_trajectory_logp,
        reference_logp=reference_token_logps.detach().sum(),
        token_count=int(
            policy_token_logps.numel()
            if token_mask is None
            else token_mask.to(dtype=torch.bool).sum().item()
        ),
    )
