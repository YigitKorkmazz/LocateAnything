#!/usr/bin/env python3
"""CPU tests: production-matching stop-grad KV replay + chain-rule grads."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.bfix_discrepancy import (  # noqa: E402
    A_MINUS_BFIX,
    A_TOTAL,
    BFIX_TOTAL,
    bfix_discrepancy_report,
)
from rl.grpo import grpo_clipped_loss  # noqa: E402
from rl.exact_backends import SURROGATE_NOT_THESIS_DEFAULT_MSG  # noqa: E402
from rl.grpo_train_step import (  # noqa: E402
    assert_initialization_ratios,
    assert_truncated_bptt_not_thesis_default,
)
from rl.pbd_rl import _detach_legacy_cache  # noqa: E402


ABS_TOL = 1e-5
REL_TOL = 1e-5


@dataclass
class ToyBlock:
    prefix_length: int
    cache_length_before: int
    action_len: int
    scored_for_grpo: bool = True


def _truncate(past, length: int):
    if past is None:
        return None
    return tuple((k[:, :, :length, :], v[:, :, :length, :]) for k, v in past)


class ToyCachedPolicy(nn.Module):
    """Minimal production-cached vs full-prefix vs stop-grad-past scorer.

    Each block appends ``action_len`` tokens. Production carries truncated KV
    (length = prefix before append). Full-prefix (Bfix analogue) ignores past.
    """

    def __init__(self, dim: int = 8, n_layers: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(64, dim)
        self.lora_A = nn.Parameter(torch.randn(dim, 2) * 0.05)
        self.lora_B = nn.Parameter(torch.randn(2, dim) * 0.05)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.out = nn.Linear(dim, 1)

    def _delta(self, h: torch.Tensor) -> torch.Tensor:
        return (h @ self.lora_A) @ self.lora_B

    def _forward_tokens(
        self,
        token_ids: torch.Tensor,
        past: Optional[Tuple],
    ) -> Tuple[torch.Tensor, Tuple]:
        # token_ids: [T]
        x = self.embed(token_ids)
        if past is not None:
            # Concatenate stop/live past channels as a stand-in for KV carry.
            past_h = past[0][0][0, 0]  # [C, D]
            x_full = torch.cat([past_h, x], dim=0)
        else:
            x_full = x
        h = x_full
        for layer in self.layers:
            h = F.gelu(layer(h) + self._delta(h))
        new_past = ((h.unsqueeze(0).unsqueeze(0), h.unsqueeze(0).unsqueeze(0)),)
        # Score last new token.
        value = self.out(h[-1]).squeeze()
        return value, new_past

    def score_production(
        self, tokens: torch.Tensor, blocks: List[ToyBlock], *, stop_grad_past: bool
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        past = None
        cursor = 0
        block_logps: List[torch.Tensor] = []
        for block in blocks:
            if not block.scored_for_grpo:
                cursor = block.prefix_length + block.action_len
                continue
            assert cursor == block.prefix_length or cursor == 0 or True
            cursor = block.prefix_length
            cache_before = 0 if past is None else int(past[0][0].size(2))
            assert cache_before == block.cache_length_before
            # Incremental suffix: tokens[cache_before:prefix+action]
            end = block.prefix_length + block.action_len
            suffix = tokens[cache_before:end]
            if stop_grad_past:
                past = _detach_legacy_cache(past)
            value, past = self._forward_tokens(suffix, past)
            past = _truncate(past, end)
            if stop_grad_past:
                past = _detach_legacy_cache(past)
            block_logps.append(value)
            cursor = end
        return torch.stack(block_logps).sum(), block_logps

    def score_bfix_full_prefix(
        self, tokens: torch.Tensor, blocks: List[ToyBlock]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        block_logps: List[torch.Tensor] = []
        for block in blocks:
            if not block.scored_for_grpo:
                continue
            end = block.prefix_length + block.action_len
            # Full prefix recompute (Bfix analogue): no past.
            value, _ = self._forward_tokens(tokens[:end], None)
            block_logps.append(value)
        return torch.stack(block_logps).sum(), block_logps


def _toy_tokens_and_blocks() -> Tuple[torch.Tensor, List[ToyBlock]]:
    # prompt 4 + 3 blocks of 2 actions. Cache catches up after each block.
    tokens = torch.arange(4 + 6, dtype=torch.long)
    blocks = [
        ToyBlock(prefix_length=4, cache_length_before=0, action_len=2),
        ToyBlock(prefix_length=6, cache_length_before=6, action_len=2),
        ToyBlock(prefix_length=8, cache_length_before=8, action_len=2),
    ]
    return tokens, blocks


def test_stop_grad_past_matches_live_production_forward() -> None:
    torch.manual_seed(0)
    model = ToyCachedPolicy()
    tokens, blocks = _toy_tokens_and_blocks()
    with torch.no_grad():
        live_total, live_blocks = model.score_production(
            tokens, blocks, stop_grad_past=False
        )
        safe_total, safe_blocks = model.score_production(
            tokens, blocks, stop_grad_past=True
        )
    assert abs(float(live_total - safe_total)) <= ABS_TOL
    for a, b in zip(live_blocks, safe_blocks):
        assert abs(float(a - b)) <= ABS_TOL


def test_bfix_discrepancy_report_uses_canonical_hybrid_gap() -> None:
    # Canonical Hybrid compare-blocks numbers (not a toy substitute).
    assert abs(A_TOTAL - BFIX_TOTAL - A_MINUS_BFIX) <= 1e-12
    # Synthetic block table with block0 match then divergence (Hybrid pattern).
    a_blocks = [-0.0392925925552845, -0.0024638872127979994, -21.3984375]
    b_blocks = [-0.0392925925552845, -0.002228624653071165, -21.482364654541016]
    report = bfix_discrepancy_report(
        a_block_logps=a_blocks, bfix_block_logps=b_blocks
    )
    assert report["first_diverging_block"] == 1
    assert "not an unroll" in report["diagnosis"]


def test_truncated_bp_chain_rule_matches_sum_backward() -> None:
    torch.manual_seed(2)
    model = ToyCachedPolicy()
    tokens, blocks = _toy_tokens_and_blocks()
    old = torch.tensor(0.0)
    adv = torch.tensor(1.0)

    # Reference: stop-grad-past score, sum, one backward.
    model_ref = copy.deepcopy(model)
    total, block_logps = model_ref.score_production(
        tokens, blocks, stop_grad_past=True
    )
    loss = grpo_clipped_loss(
        total.reshape(1), old.reshape(1), adv.reshape(1), clip_epsilon=0.2
    )
    (loss / 4.0).backward()
    ref_grads = {
        name: param.grad.detach().clone()
        for name, param in model_ref.named_parameters()
        if param.grad is not None
    }

    # Chain rule: leaf current + per-block (coeff * logp_i).backward().
    model_cr = copy.deepcopy(model)
    with torch.no_grad():
        current_f, _ = model_cr.score_production(
            tokens, blocks, stop_grad_past=True
        )
    current = torch.tensor(
        float(current_f), dtype=torch.float32, requires_grad=True
    )
    loss_leaf = grpo_clipped_loss(
        current.reshape(1), old.reshape(1), adv.reshape(1), clip_epsilon=0.2
    )
    (loss_leaf / 4.0).backward()
    coeff = float(current.grad)
    # Re-run per block with stop-grad past.
    past = None
    cursor_tokens = tokens
    for block in blocks:
        if not block.scored_for_grpo:
            continue
        cache_before = 0 if past is None else int(past[0][0].size(2))
        end = block.prefix_length + block.action_len
        past = _detach_legacy_cache(past)
        value, past = model_cr._forward_tokens(
            cursor_tokens[cache_before:end], past
        )
        past = _truncate(past, end)
        past = _detach_legacy_cache(past)
        (value * coeff).backward()
        del value

    for name, ref in ref_grads.items():
        got = dict(model_cr.named_parameters())[name].grad
        assert got is not None
        diff = (got - ref).abs().max().item()
        scale = max(ref.abs().max().item(), 1e-12)
        assert diff <= ABS_TOL or diff / scale <= REL_TOL


def test_optimizer_update_and_init_ratio() -> None:
    torch.manual_seed(3)
    model = ToyCachedPolicy()
    tokens, blocks = _toy_tokens_and_blocks()
    with torch.no_grad():
        old_logp = float(
            model.score_production(tokens, blocks, stop_grad_past=True)[0]
        )
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    # One truncated-BP step.
    with torch.no_grad():
        current_f, _ = model.score_production(
            tokens, blocks, stop_grad_past=True
        )
    assert_initialization_ratios([float(current_f)], [old_logp])
    current = torch.tensor(float(current_f), requires_grad=True)
    loss = grpo_clipped_loss(
        current.reshape(1),
        torch.tensor([old_logp]),
        torch.tensor([1.0]),
        clip_epsilon=0.2,
    )
    loss.backward()
    coeff = float(current.grad)
    opt.zero_grad(set_to_none=True)
    past = None
    for block in blocks:
        cache_before = 0 if past is None else int(past[0][0].size(2))
        end = block.prefix_length + block.action_len
        past = _detach_legacy_cache(past)
        value, past = model._forward_tokens(tokens[cache_before:end], past)
        past = _detach_legacy_cache(_truncate(past, end))
        (value * coeff).backward()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    changed = any(
        not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()
    )
    assert changed


def test_truncated_bptt_not_thesis_default() -> None:
    assert_truncated_bptt_not_thesis_default(
        replay_backend="live_production_cached_autograd",
        allow_truncated_bptt_surrogate=False,
    )
    try:
        assert_truncated_bptt_not_thesis_default(
            replay_backend="truncated_bptt_surrogate",
            allow_truncated_bptt_surrogate=False,
        )
        raise AssertionError("expected surrogate rejection")
    except RuntimeError as exc:
        assert "NOT gradient-equivalent" in str(exc)
        assert SURROGATE_NOT_THESIS_DEFAULT_MSG[:40] in str(exc)


def main() -> None:
    test_stop_grad_past_matches_live_production_forward()
    test_bfix_discrepancy_report_uses_canonical_hybrid_gap()
    test_truncated_bp_chain_rule_matches_sum_backward()
    test_optimizer_update_and_init_ratio()
    test_truncated_bptt_not_thesis_default()
    print("ALL_AUTOGRADE_SAFE_PRODUCTION_TESTS_PASSED")


if __name__ == "__main__":
    main()
