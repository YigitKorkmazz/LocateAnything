#!/usr/bin/env python3
"""CPU tiny-model: live production-cached autograd vs stop-grad-past gradients.

This establishes whether stop-grad past_key_values replay is
exact-gradient-equivalent to live cached autograd, or only a truncated-BPTT
surrogate that matches forwards.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.pbd_rl import _detach_legacy_cache  # noqa: E402


ABS_TOL = 1e-5
REL_TOL = 1e-5
COS_EXACT = 1.0 - 1e-6


@dataclass
class ToyBlock:
    prefix_length: int
    cache_length_before: int
    action_len: int
    scored_for_grpo: bool = True


def _truncate(past, length: int):
    if past is None:
        return None
    return tuple(
        (k[:, :, :length, :], v[:, :, :length, :]) for k, v in past
    )


class LoRALinear(nn.Module):
    def __init__(self, din: int, dout: int, rank: int = 2) -> None:
        super().__init__()
        self.base = nn.Linear(din, dout, bias=False)
        self.lora_A = nn.Parameter(torch.randn(rank, din) * 0.05)
        self.lora_B = nn.Parameter(torch.zeros(dout, rank))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B)


class TinyAttnLoRALayer(nn.Module):
    """One attention layer with LoRA on q/k/v (and frozen o_proj)."""

    def __init__(self, dim: int, rank: int = 2) -> None:
        super().__init__()
        self.q_proj = LoRALinear(dim, dim, rank=rank)
        self.k_proj = LoRALinear(dim, dim, rank=rank)
        self.v_proj = LoRALinear(dim, dim, rank=rank)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        for parameter in self.o_proj.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        x_new: torch.Tensor,
        past_k: Optional[torch.Tensor],
        past_v: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_new: [T_new, D]
        q = self.q_proj(x_new)
        k_new = self.k_proj(x_new)
        v_new = self.v_proj(x_new)
        if past_k is None:
            k = k_new
            v = v_new
        else:
            k = torch.cat([past_k, k_new], dim=0)
            v = torch.cat([past_v, v_new], dim=0)
        # Causal attention over full K/V, queries = new tokens only.
        scale = 1.0 / math.sqrt(q.size(-1))
        scores = (q @ k.transpose(0, 1)) * scale
        # Mask: query i (global index = past_len + i) cannot see future keys.
        past_len = 0 if past_k is None else int(past_k.size(0))
        t_new = int(q.size(0))
        t_total = past_len + t_new
        q_idx = torch.arange(past_len, t_total, device=q.device).unsqueeze(1)
        k_idx = torch.arange(t_total, device=q.device).unsqueeze(0)
        scores = scores.masked_fill(k_idx > q_idx, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = self.o_proj(attn @ v)
        return out, k, v


class TinyProductionCachedLM(nn.Module):
    """Tiny LM whose carried K/V are produced by LoRA q/k/v projections."""

    def __init__(self, dim: int = 16, n_layers: int = 2, rank: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(128, dim)
        for parameter in self.embed.parameters():
            parameter.requires_grad_(False)
        self.layers = nn.ModuleList(
            [TinyAttnLoRALayer(dim, rank=rank) for _ in range(n_layers)]
        )
        self.out = nn.Linear(dim, 1, bias=False)
        for parameter in self.out.parameters():
            parameter.requires_grad_(False)

    def lora_parameter_dict(self) -> Dict[str, nn.Parameter]:
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and "lora_" in name
        }

    def _forward_suffix(
        self,
        token_ids: torch.Tensor,
        past: Optional[Tuple],
    ) -> Tuple[torch.Tensor, Tuple]:
        x = self.embed(token_ids)  # [T, D]
        new_past_layers = []
        h = x
        for layer_index, layer in enumerate(self.layers):
            past_k = None if past is None else past[layer_index][0][0, 0]
            past_v = None if past is None else past[layer_index][1][0, 0]
            h, k, v = layer(h, past_k, past_v)
            new_past_layers.append(
                (k.unsqueeze(0).unsqueeze(0), v.unsqueeze(0).unsqueeze(0))
            )
        value = self.out(h[-1]).squeeze()
        return value, tuple(new_past_layers)

    def score(
        self,
        tokens: torch.Tensor,
        blocks: Sequence[ToyBlock],
        *,
        stop_grad_past: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        past = None
        block_logps: List[torch.Tensor] = []
        for block in blocks:
            if not block.scored_for_grpo:
                continue
            cache_before = 0 if past is None else int(past[0][0].size(2))
            if cache_before != block.cache_length_before:
                raise RuntimeError(
                    f"cache length {cache_before} != {block.cache_length_before}"
                )
            end = block.prefix_length + block.action_len
            suffix = tokens[cache_before:end]
            if stop_grad_past:
                past = _detach_legacy_cache(past)
            value, past = self._forward_suffix(suffix, past)
            past = _truncate(past, end)
            if stop_grad_past:
                past = _detach_legacy_cache(past)
            block_logps.append(value)
        return torch.stack(block_logps).sum(), block_logps


def _toy_setup() -> Tuple[TinyProductionCachedLM, torch.Tensor, List[ToyBlock]]:
    torch.manual_seed(0)
    model = TinyProductionCachedLM(dim=16, n_layers=2, rank=2)
    # prompt 6 + 3 scored blocks of 3 actions => lengths 6,9,12 prefix starts
    tokens = torch.arange(6 + 9, dtype=torch.long)
    blocks = [
        ToyBlock(prefix_length=6, cache_length_before=0, action_len=3),
        ToyBlock(prefix_length=9, cache_length_before=9, action_len=3),
        ToyBlock(prefix_length=12, cache_length_before=12, action_len=3),
    ]
    return model, tokens, blocks


def _grad_metrics(
    g_live: torch.Tensor, g_stop: torch.Tensor
) -> Dict[str, float]:
    live = g_live.detach().float().reshape(-1)
    stop = g_stop.detach().float().reshape(-1)
    diff = live - stop
    live_l2 = float(torch.linalg.vector_norm(live))
    stop_l2 = float(torch.linalg.vector_norm(stop))
    diff_l2 = float(torch.linalg.vector_norm(diff))
    denom = max(live_l2, 1e-12)
    if live_l2 == 0.0 and stop_l2 == 0.0:
        cos = 1.0
        rel = 0.0
    else:
        cos = float(F.cosine_similarity(live.unsqueeze(0), stop.unsqueeze(0)))
        rel = diff_l2 / denom
    return {
        "max_abs_diff": float(diff.abs().max()) if diff.numel() else 0.0,
        "relative_l2_diff": rel,
        "cosine_similarity": cos,
        "live_l2": live_l2,
        "stop_l2": stop_l2,
        "live_grad_nonzero": bool(live_l2 > 0.0),
        "stop_grad_nonzero": bool(stop_l2 > 0.0),
    }


def _collect_lora_grads(model: TinyProductionCachedLM) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.lora_parameter_dict().items()
        if parameter.grad is not None
    }


def _zero_grads(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad = None


def _backward_objective(
    model: TinyProductionCachedLM,
    tokens: torch.Tensor,
    blocks: Sequence[ToyBlock],
    *,
    stop_grad_past: bool,
    objective: str,
) -> Dict[str, torch.Tensor]:
    _zero_grads(model)
    total, block_logps = model.score(
        tokens, blocks, stop_grad_past=stop_grad_past
    )
    if objective == "sum":
        total.backward()
    elif objective.startswith("block_"):
        index = int(objective.split("_", 1)[1])
        block_logps[index].backward()
    else:
        raise RuntimeError(f"unknown objective {objective!r}")
    return _collect_lora_grads(model)


def _compare_grad_dicts(
    live: Dict[str, torch.Tensor],
    stop: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    names = sorted(set(live) | set(stop))
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        g0 = live.get(name)
        g1 = stop.get(name)
        if g0 is None:
            g0 = torch.zeros_like(g1)
        if g1 is None:
            g1 = torch.zeros_like(g0)
        out[name] = _grad_metrics(g0, g1)
    return out


def _aggregate(metrics_by_name: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics_by_name:
        return {
            "max_abs_diff": 0.0,
            "relative_l2_diff": 0.0,
            "cosine_similarity": 1.0,
        }
    return {
        "max_abs_diff": max(m["max_abs_diff"] for m in metrics_by_name.values()),
        "relative_l2_diff": max(
            m["relative_l2_diff"] for m in metrics_by_name.values()
        ),
        "cosine_similarity": min(
            m["cosine_similarity"] for m in metrics_by_name.values()
        ),
    }


def _is_exact(agg: Dict[str, float]) -> bool:
    return (
        agg["max_abs_diff"] <= ABS_TOL
        and agg["relative_l2_diff"] <= REL_TOL
        and agg["cosine_similarity"] >= COS_EXACT
    )


def run_gradient_equivalence_suite() -> Dict[str, object]:
    model_live, tokens, blocks = _toy_setup()
    model_stop = copy.deepcopy(model_live)
    assert len(blocks) >= 3

    # Forward match first.
    with torch.no_grad():
        live_total, live_blocks = model_live.score(
            tokens, blocks, stop_grad_past=False
        )
        stop_total, stop_blocks = model_stop.score(
            tokens, blocks, stop_grad_past=True
        )
    forward_block_diffs = [
        abs(float(a - b)) for a, b in zip(live_blocks, stop_blocks)
    ]
    forward_exact = (
        abs(float(live_total - stop_total)) <= ABS_TOL
        and max(forward_block_diffs) <= ABS_TOL
    )

    objectives = ["block_0", "block_1", "block_2", "sum"]
    per_objective = {}
    for objective in objectives:
        live_grads = _backward_objective(
            model_live, tokens, blocks, stop_grad_past=False, objective=objective
        )
        stop_grads = _backward_objective(
            model_stop, tokens, blocks, stop_grad_past=True, objective=objective
        )
        by_name = _compare_grad_dicts(live_grads, stop_grads)
        agg = _aggregate(by_name)
        per_objective[objective] = {
            "aggregate": agg,
            "exact_gradient_equivalent": _is_exact(agg),
            "by_parameter": by_name,
            "live_nonzero_param_count": sum(
                1 for m in by_name.values() if m["live_grad_nonzero"]
            ),
            "stop_nonzero_param_count": sum(
                1 for m in by_name.values() if m["stop_grad_nonzero"]
            ),
        }

    # Explicit early-KV probe: for block_1/2, compare live vs stop on params that
    # receive extra live gradient through retained past K/V (typically v_proj).
    def _early_kv_signal(objective: str) -> Dict[str, object]:
        by_name = per_objective[objective]["by_parameter"]
        # Prefer v_proj.lora_B where the live/stop gap is largest in this toy.
        focus = {
            name: metrics
            for name, metrics in by_name.items()
            if "v_proj.lora_B" in name
        }
        return {
            "focus_parameters": focus,
            "live_focus_l2_sum": sum(m["live_l2"] for m in focus.values()),
            "stop_focus_l2_sum": sum(m["stop_l2"] for m in focus.values()),
            "live_larger_than_stop_on_v_proj": any(
                m["live_l2"] > m["stop_l2"] + 1e-12 for m in focus.values()
            ),
            "not_exact": not per_objective[objective]["exact_gradient_equivalent"],
        }

    early_kv_probe = {
        "block_1": _early_kv_signal("block_1"),
        "block_2": _early_kv_signal("block_2"),
        "interpretation": (
            "Live block_1/block_2 losses send additional gradient into LoRA "
            "parameters that built earlier-block V (and K) via retained past "
            "autograd. Stop-grad past removes that path (truncated BPTT)."
        ),
    }

    # Optimizer update from identical weights.
    model_a = copy.deepcopy(model_live)
    model_b = copy.deepcopy(model_live)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)
    opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)
    _zero_grads(model_a)
    _zero_grads(model_b)
    total_a, _ = model_a.score(tokens, blocks, stop_grad_past=False)
    total_b, _ = model_b.score(tokens, blocks, stop_grad_past=True)
    total_a.backward()
    total_b.backward()
    opt_a.step()
    opt_b.step()
    update_metrics = {}
    for name, p_a in model_a.lora_parameter_dict().items():
        p_b = dict(model_b.named_parameters())[name]
        # Compare deltas from the shared init (model_live).
        init_p = dict(model_live.named_parameters())[name]
        d_a = p_a.detach() - init_p.detach()
        d_b = p_b.detach() - init_p.detach()
        update_metrics[name] = _grad_metrics(d_a, d_b)
    update_agg = _aggregate(update_metrics)

    sum_exact = per_objective["sum"]["exact_gradient_equivalent"]
    block0_exact = per_objective["block_0"]["exact_gradient_equivalent"]
    # Block 0 has no past, so live == stop there. Later blocks expose truncation.
    if sum_exact and all(
        per_objective[o]["exact_gradient_equivalent"] for o in objectives
    ):
        verdict = "exact_gradient_equivalent"
    else:
        verdict = "truncated_bptt_surrogate"

    report = {
        "num_scored_blocks": len(blocks),
        "lora_param_names": sorted(model_live.lora_parameter_dict()),
        "forward_exact_match": forward_exact,
        "forward_total_abs_diff": abs(float(live_total - stop_total)),
        "forward_max_block_abs_diff": max(forward_block_diffs),
        "objectives": per_objective,
        "early_kv_gradient_probe": early_kv_probe,
        "optimizer_update": {
            "aggregate": update_agg,
            "exact_equivalent": _is_exact(update_agg),
            "by_parameter": update_metrics,
        },
        "block0_exact_as_expected": block0_exact,
        "verdict": verdict,
        "interpretation": (
            "Stop-grad past_key_values matches production forwards, but cuts "
            "autograd edges from later-block losses into LoRA parameters that "
            "constructed earlier-block KV. Therefore it is truncated BPTT, not "
            "exact live production-cached autograd."
            if verdict == "truncated_bptt_surrogate"
            else "Stop-grad past matched live cached autograd on this toy."
        ),
    }
    return report


def main() -> None:
    report = run_gradient_equivalence_suite()
    print(json.dumps(report, indent=2))
    assert report["forward_exact_match"] is True
    assert report["num_scored_blocks"] >= 3
    assert any("q_proj.lora_A" in n for n in report["lora_param_names"])
    assert any("k_proj.lora_A" in n for n in report["lora_param_names"])
    assert any("v_proj.lora_A" in n for n in report["lora_param_names"])
    assert report["block0_exact_as_expected"] is True
    # The scientific point of this suite: later blocks differ under stop-grad.
    assert report["verdict"] == "truncated_bptt_surrogate"
    assert report["early_kv_gradient_probe"]["block_1"]["not_exact"] is True
    assert report["early_kv_gradient_probe"]["block_2"]["not_exact"] is True
    assert report["early_kv_gradient_probe"]["block_1"][
        "live_larger_than_stop_on_v_proj"
    ]
    assert report["early_kv_gradient_probe"]["block_2"][
        "live_larger_than_stop_on_v_proj"
    ]
    assert report["objectives"]["sum"]["exact_gradient_equivalent"] is False
    assert report["optimizer_update"]["exact_equivalent"] is False
    print("LIVE_VS_STOPGRAD_GRADIENT_SUITE_PASSED")
    print(f"VERDICT={report['verdict']}")


if __name__ == "__main__":
    main()
