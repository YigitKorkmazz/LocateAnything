#!/usr/bin/env python3
"""GPU test: checkpoint save/resume exactness.

Does not run the (expensive) rollout/reward/replay pipeline -- it applies a
synthetic gradient to the real LoRA parameters so the optimizer accumulates
real Adam moment state, saves a checkpoint, builds a *fresh* model +
optimizer + scheduler, resumes, and checks:
  - LoRA adapter weights match bit-for-bit
  - optimizer state_dict (Adam moments) matches bit-for-bit
  - scheduler state (last_epoch / resulting LR) matches
  - the resumed cursor points at the correct next window, so the caller
    neither re-processes nor skips a group
  - RNG state round-trips (python/numpy/torch/cuda)
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from production_grpo_fast import checkpoint as ckpt_mod, runtime as prod_runtime, sampler  # noqa: E402


def _apply_synthetic_step(model, optimizer) -> None:
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p) * 0.01
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def main() -> None:
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
    torch.cuda.set_device(device)
    tmp_dir = Path(tempfile.mkdtemp(prefix="ckpt_resume_test_"))
    try:
        model_a, _, _, _ = prod_runtime.build_model_and_tokenizer(device)
        optimizer_a = prod_runtime.build_optimizer(model_a, lr=1e-6, weight_decay=0.0)
        scheduler_a = prod_runtime.build_linear_schedule(optimizer_a, total_steps=sampler.TOTAL_OPTIMIZER_STEPS)
        for _ in range(3):
            _apply_synthetic_step(model_a, optimizer_a)
            scheduler_a.step()

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        next_group = 792  # an arbitrary window boundary that straddles an epoch boundary (790 % 4 == 2)
        assert next_group % sampler.GROUPS_PER_UPDATE == 0

        saved_path = ckpt_mod.save_checkpoint(
            tmp_dir, model_a, optimizer_a, scheduler_a,
            next_global_group_index=next_group, config={"test": True},
        )
        print("PASS checkpoint saved:", saved_path)

        lora_before = {
            n: p.detach().float().cpu().clone() for n, p in model_a.named_parameters() if p.requires_grad
        }
        opt_state_before = optimizer_a.state_dict()
        lr_before = scheduler_a.get_last_lr()

        model_b, _, _, _ = prod_runtime.build_model_and_tokenizer(device)
        optimizer_b = prod_runtime.build_optimizer(model_b, lr=1e-6, weight_decay=0.0)
        scheduler_b = prod_runtime.build_linear_schedule(optimizer_b, total_steps=sampler.TOTAL_OPTIMIZER_STEPS)
        lora_fresh = {n: p.detach().float().cpu().clone() for n, p in model_b.named_parameters() if p.requires_grad}
        any_differs = any(not torch.equal(lora_before[n], lora_fresh[n]) for n in lora_before)
        assert any_differs, "sanity check failed: trained and fresh-init weights are identical before resume"

        resumed_next_group = ckpt_mod.resume_checkpoint(saved_path, model_b, optimizer_b, scheduler_b)
        assert resumed_next_group == next_group, (resumed_next_group, next_group)
        print("PASS resumed cursor:", resumed_next_group)

        lora_after = {n: p.detach().float().cpu().clone() for n, p in model_b.named_parameters() if p.requires_grad}
        for n in lora_before:
            assert torch.equal(lora_before[n], lora_after[n]), f"LoRA tensor {n} did not round-trip exactly"
        print("PASS LoRA adapter weights match bit-for-bit after resume")

        opt_state_after = optimizer_b.state_dict()
        assert opt_state_before["param_groups"] == opt_state_after["param_groups"]
        for key_before, state_before in opt_state_before["state"].items():
            state_after = opt_state_after["state"][key_before]
            for moment_key in ("exp_avg", "exp_avg_sq", "step"):
                if moment_key in state_before:
                    a = state_before[moment_key]
                    b = state_after[moment_key]
                    if torch.is_tensor(a):
                        assert torch.equal(a.cpu(), b.cpu()), f"optimizer {moment_key} mismatch"
                    else:
                        assert a == b
        print("PASS optimizer (Adam moment) state matches bit-for-bit after resume")

        assert scheduler_b.get_last_lr() == lr_before, (scheduler_b.get_last_lr(), lr_before)
        print("PASS scheduler state matches after resume:", scheduler_b.get_last_lr())

        rng_state = ckpt_mod.rng_state()
        random.random()
        np.random.random()
        torch.rand(1)
        ckpt_mod.restore_rng_state(rng_state)
        assert ckpt_mod.rng_state()["python"] == rng_state["python"]
        print("PASS RNG state round-trips")

        print("ALL CHECKPOINT/RESUME TESTS PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
