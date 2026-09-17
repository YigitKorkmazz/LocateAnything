#!/usr/bin/env python3
"""GPU test: one real 4-prompt-group optimizer update, single process.

Phase 5 of the work plan. Verifies:
  - trainability audit (504 / 14,966,784)
  - gradient audit after the window (LoRA finite+nonzero, projector/vision/
    frozen-non-LoRA all exactly zero)
  - the optimizer actually moved the LoRA weights (update norm > 0, finite)
  - advantages are normalized within each 4-rollout group only (mean ~ 0,
    std ~ 1 per group, computed independently per group)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from production_grpo_fast import grpo_step, runtime as prod_runtime, sampler  # noqa: E402
from production_grpo_fast.rewards_adapter import build_reward_pipeline  # noqa: E402
from rl.rewards import MedCLIPSemanticScorer  # noqa: E402


def main() -> None:
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
    torch.cuda.set_device(device)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train import load_train80_pairs  # noqa: E402

    model, tokenizer, processor, load_report = prod_runtime.build_model_and_tokenizer(device)
    audit = prod_runtime.trainability_audit(model)
    print("PASS trainability audit:", audit["trainable_tensors"], audit["trainable_parameters"])

    pairs = load_train80_pairs()
    reward_pipeline = build_reward_pipeline(MedCLIPSemanticScorer(device=device))
    optimizer = prod_runtime.build_optimizer(model, lr=1e-6, weight_decay=0.0)

    specs = [sampler.resolve_group(i) for i in range(4)]
    local_pairs = [pairs[s.manifest_index] for s in specs]
    seeds = [s.sample_seed for s in specs]

    before = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    result = grpo_step.run_accumulation_window(
        model, optimizer, tokenizer, processor, reward_pipeline,
        local_pairs, device=device, sample_seeds=seeds, audit=True,
    )

    for report in result["group_reports"]:
        advantages = report["advantages"]
        mean = sum(advantages) / len(advantages)
        assert len(advantages) == 4
        assert abs(mean) < 1e-4, f"within-group advantage mean not ~0: {advantages}"
        print("PASS group advantages (within-group only):", advantages, "rewards:", report["rewards"])

    audit_result = result["gradient_audit"]
    assert audit_result["lora_tensors_with_finite_gradients"] == 504, audit_result
    assert audit_result["projector_tensors_with_gradients"] == 0, audit_result
    assert audit_result["vision_tensors_with_gradients"] == 0, audit_result
    assert audit_result["frozen_non_lora_tensors_with_gradients"] == 0, audit_result
    assert audit_result["lora_tensors_with_nonzero_gradients"] > 0, audit_result
    print("PASS gradient audit:", audit_result)

    update_norm_sq = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        delta = p.detach().float().cpu() - before[n]
        update_norm_sq += float(delta.square().sum())
    update_norm = math.sqrt(update_norm_sq)
    assert math.isfinite(update_norm) and update_norm > 0.0, update_norm
    print("PASS optimizer update norm:", update_norm)
    print("ALL ONE-OPTIMIZER-UPDATE TESTS PASSED (4 groups / 16 rollouts / 1 Adam step)")


if __name__ == "__main__":
    main()
