"""GRPO-only objective for probability-consistent PBD rollouts."""

from __future__ import annotations

from typing import Sequence

import torch


def group_relative_advantages(
    rewards: Sequence[float] | torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("GRPO requires a one-dimensional reward group of size >= 2")
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def grpo_clipped_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Return only the clipped GRPO policy loss; no supervised terms exist."""
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old log-probability shapes differ")
    if advantages.shape != current_log_probs.shape:
        raise ValueError("advantages and log-probability shapes differ")
    ratios = torch.exp(current_log_probs - old_log_probs.detach())
    unclipped = ratios * advantages
    clipped = torch.clamp(
        ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * advantages
    return -torch.minimum(unclipped, clipped).mean()
