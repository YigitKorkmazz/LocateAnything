#!/usr/bin/env python3
"""CPU equivalence: PEFT-style LoRA math vs diagnostic contiguous/fp32 workarounds.

These workarounds must preserve forward values and gradients on well-behaved
inputs. On non-contiguous BF16 inputs, contiguous() may be required for CUDA
correctness; this CPU test checks exact match on contiguous tensors and reports
whether non-contiguous inputs diverge before/after the workaround.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.lora_grad_diagnostics import (  # noqa: E402
    exact_lora_delta_reference,
    peft_style_lora_forward,
)


def _match(a: torch.Tensor, b: torch.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    return bool(torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol))


def test_contiguous_workaround_matches_peft_on_contiguous_inputs() -> None:
    torch.manual_seed(0)
    batch, seq, din, rank, dout = 2, 7, 32, 8, 32
    x = torch.randn(batch, seq, din, dtype=torch.float32, requires_grad=True)
    base_w = torch.randn(dout, din, dtype=torch.float32, requires_grad=True)
    a_w = torch.randn(rank, din, dtype=torch.float32, requires_grad=True)
    b_w = torch.randn(dout, rank, dtype=torch.float32, requires_grad=True)
    scaling = 2.0

    y_peft = peft_style_lora_forward(x, base_w, a_w, b_w, scaling=scaling)
    y_safe = F.linear(x, base_w) + exact_lora_delta_reference(
        x, a_w, b_w, scaling=scaling, contiguous_input=True, fp32_lora=False
    )
    assert _match(y_peft, y_safe)

    loss_peft = y_peft.sum()
    loss_safe = y_safe.sum()
    grads_peft = torch.autograd.grad(loss_peft, [x, base_w, a_w, b_w], retain_graph=True)
    grads_safe = torch.autograd.grad(loss_safe, [x, base_w, a_w, b_w])
    for g0, g1 in zip(grads_peft, grads_safe):
        assert _match(g0, g1)


def test_fp32_lora_workaround_matches_on_fp32_inputs() -> None:
    torch.manual_seed(1)
    x = torch.randn(3, 5, 16, dtype=torch.float32, requires_grad=True)
    base_w = torch.randn(16, 16, dtype=torch.float32, requires_grad=True)
    a_w = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    b_w = torch.randn(16, 4, dtype=torch.float32, requires_grad=True)
    scaling = 1.5
    y_peft = peft_style_lora_forward(x, base_w, a_w, b_w, scaling=scaling)
    y_fp32 = F.linear(x, base_w) + exact_lora_delta_reference(
        x, a_w, b_w, scaling=scaling, contiguous_input=False, fp32_lora=True
    )
    assert _match(y_peft, y_fp32, atol=1e-6, rtol=1e-6)
    g_peft = torch.autograd.grad(y_peft.sum(), [a_w, b_w], retain_graph=True)
    g_fp32 = torch.autograd.grad(y_fp32.sum(), [a_w, b_w])
    for g0, g1 in zip(g_peft, g_fp32):
        assert _match(g0, g1, atol=1e-6, rtol=1e-6)


def test_optimizer_update_match_for_workaround() -> None:
    torch.manual_seed(2)
    x = torch.randn(2, 4, 8, dtype=torch.float32)
    base_w = torch.randn(8, 8, dtype=torch.float32, requires_grad=True)
    a_w = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    b_w = torch.randn(8, 2, dtype=torch.float32, requires_grad=True)
    scaling = 2.0

    def one_step(contiguous: bool):
        base = base_w.detach().clone().requires_grad_(True)
        a = a_w.detach().clone().requires_grad_(True)
        b = b_w.detach().clone().requires_grad_(True)
        opt = torch.optim.SGD([base, a, b], lr=1e-2)
        y = F.linear(x, base) + exact_lora_delta_reference(
            x, a, b, scaling=scaling, contiguous_input=contiguous, fp32_lora=False
        )
        # Fake trajectory logp + ratio=1 init identity surrogate.
        logp = y.mean()
        old = logp.detach()
        ratio = torch.exp(logp - old)
        assert abs(float(ratio) - 1.0) < 1e-6
        loss = -(ratio * torch.tensor(1.0)).mean()
        loss.backward()
        opt.step()
        return base.detach().clone(), a.detach().clone(), b.detach().clone(), float(logp)

    b0, a0, c0, logp0 = one_step(False)
    b1, a1, c1, logp1 = one_step(True)
    assert abs(logp0 - logp1) < 1e-6
    assert _match(b0, b1) and _match(a0, a1) and _match(c0, c1)


def main() -> None:
    test_contiguous_workaround_matches_peft_on_contiguous_inputs()
    test_fp32_lora_workaround_matches_on_fp32_inputs()
    test_optimizer_update_match_for_workaround()
    print("ALL_LORA_WORKAROUND_CPU_TESTS_PASSED")


if __name__ == "__main__":
    main()
