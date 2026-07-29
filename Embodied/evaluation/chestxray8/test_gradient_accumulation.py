#!/usr/bin/env python3
"""Focused regression tests for epoch-local gradient accumulation."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from train_chestxray8_sft import (  # noqa: E402
    _assert_output_dir_available,
    accumulation_group_size,
    finish_gradient_accumulation,
)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.001)
        self.step_count = 0
        self.gradients = []

    def step(self, closure=None):
        parameter = self.param_groups[0]["params"][0]
        self.gradients.append(float(parameter.grad.detach().cpu()))
        self.step_count += 1
        return super().step(closure)


class CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


def run_new(num_batches: int, accum: int, epochs: int):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = CountingSGD([parameter])
    scheduler = CountingScheduler()
    optimizer.zero_grad(set_to_none=True)
    epoch_end_grads = []
    for _epoch in range(epochs):
        for batch_idx in range(num_batches):
            loss = parameter * float(batch_idx + 1)
            (loss / accum).backward()
            group_size = accumulation_group_size(batch_idx, num_batches, accum)
            if group_size:
                finish_gradient_accumulation(
                    torch.nn.ParameterList([parameter]),
                    optimizer,
                    scheduler,
                    accumulation_steps=accum,
                    group_size=group_size,
                    max_grad_norm=0.0,
                )
        epoch_end_grads.append(parameter.grad)
    return parameter, optimizer, scheduler, epoch_end_grads


def run_legacy_divisible(num_batches: int, accum: int, epochs: int):
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = CountingSGD([parameter])
    scheduler = CountingScheduler()
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(epochs):
        for batch_idx in range(num_batches):
            loss = parameter * float(batch_idx + 1)
            (loss / accum).backward()
            if (batch_idx + 1) % accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
    return parameter, optimizer, scheduler


def test_divisible_epoch_is_unchanged():
    new_parameter, new_optimizer, new_scheduler, end_grads = run_new(16, 8, 2)
    old_parameter, old_optimizer, old_scheduler = run_legacy_divisible(16, 8, 2)
    assert new_optimizer.step_count == old_optimizer.step_count == 4
    assert new_scheduler.step_count == old_scheduler.step_count == 4
    assert new_optimizer.gradients == old_optimizer.gradients
    assert torch.equal(new_parameter.detach(), old_parameter.detach())
    assert all(grad is None for grad in end_grads)


def test_non_divisible_epoch_flushes_without_gradient_carry():
    _parameter, optimizer, scheduler, end_grads = run_new(790, 8, 2)
    assert optimizer.step_count == 99 * 2
    assert scheduler.step_count == 99 * 2
    assert all(grad is None for grad in end_grads)
    # Final six gradients are rescaled from sum/8 to their true mean.
    assert abs(optimizer.gradients[98] - 787.5) < 1e-6
    # The first full group of epoch two starts clean: mean(1..8) == 4.5.
    assert abs(optimizer.gradients[99] - 4.5) < 1e-6


def test_occupied_training_output_is_refused_without_resume():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "train"
        output.mkdir()
        _assert_output_dir_available(output, None)
        (output / "old.log").write_text("occupied\n")
        try:
            _assert_output_dir_available(output, None)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Occupied training directory was not refused")
        _assert_output_dir_available(output, str(output / "checkpoint-1"))


def main() -> None:
    test_divisible_epoch_is_unchanged()
    test_non_divisible_epoch_flushes_without_gradient_carry()
    test_occupied_training_output_is_refused_without_resume()
    print("GRADIENT ACCUMULATION TESTS PASSED")


if __name__ == "__main__":
    main()
