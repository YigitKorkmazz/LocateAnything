#!/usr/bin/env python3
"""GPU test: reference-policy KL penalty correctness.

Key check: PEFT LoRA's default init has B=0, so the LoRA delta (B @ A) is
EXACTLY zero at a freshly built checkpoint -- meaning the current policy and
the disable_adapter() reference policy are LITERALLY THE SAME MODEL before
any optimizer step. This gives a strong, almost-known-answer correctness
check for the whole reference/KL pipeline (disable_adapter actually zeroing
the LoRA delta, token-for-token alignment between the two teacher-forced
passes, and the k3 estimator's arithmetic): KL must be ~0 at init.

After one synthetic gradient step moves the LoRA weights away from zero,
the SAME rollout's completion tokens must show a clearly positive, finite
KL -- and kl_beta=0.0 must skip KL entirely (no extra key, unchanged
behavior), preserving this backend's prior default exactly.
"""

from __future__ import annotations

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

    model, tokenizer, processor, _ = prod_runtime.build_model_and_tokenizer(device)
    prod_runtime.trainability_audit(model)
    pairs = load_train80_pairs()
    reward_pipeline = build_reward_pipeline(MedCLIPSemanticScorer(device=device))
    optimizer = prod_runtime.build_optimizer(model, lr=1e-6, weight_decay=0.0)

    spec = sampler.resolve_group(0)
    pair = pairs[spec.manifest_index]

    # --- kl_beta=0.0 must skip KL entirely (unchanged prior behavior) ---
    optimizer.zero_grad(set_to_none=True)
    report0 = grpo_step.execute_prompt_group(
        model, tokenizer, processor, reward_pipeline, pair, optimizer,
        device=device, sample_seed=spec.sample_seed, loss_divisor=16.0, kl_beta=0.0,
    )
    for r in report0["rollout_reports"]:
        assert "kl_penalty" not in r, "kl_beta=0.0 must not compute a KL penalty"
    print("PASS kl_beta=0.0 skips KL entirely (no kl_penalty key)")

    # --- fresh (untrained) LoRA: B=0 -> current policy == reference policy ---
    optimizer.zero_grad(set_to_none=True)
    report_fresh = grpo_step.execute_prompt_group(
        model, tokenizer, processor, reward_pipeline, pair, optimizer,
        device=device, sample_seed=spec.sample_seed, loss_divisor=16.0, kl_beta=0.04,
    )
    fresh_kls = [r["kl_penalty"] for r in report_fresh["rollout_reports"]]
    print("fresh-model KL penalties (expect ~0):", fresh_kls)
    for kl in fresh_kls:
        assert kl == kl and kl < 1e-3, f"KL should be ~0 at a fresh (B=0) LoRA init, got {kl}"
    print("PASS KL ~ 0 at fresh LoRA init (disable_adapter() correctly reproduces the current policy)")
    optimizer.zero_grad(set_to_none=True)

    # --- move LoRA weights away from zero with a synthetic step ---
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p) * 0.05
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    report_trained = grpo_step.execute_prompt_group(
        model, tokenizer, processor, reward_pipeline, pair, optimizer,
        device=device, sample_seed=spec.sample_seed, loss_divisor=16.0, kl_beta=0.04,
    )
    trained_kls = [r["kl_penalty"] for r in report_trained["rollout_reports"]]
    print("post-synthetic-step KL penalties (expect clearly > 0):", trained_kls)
    for kl in trained_kls:
        assert kl == kl and 0.0 < kl < 1e6, f"non-finite or non-positive KL after weight update: {kl}"
    assert sum(trained_kls) > sum(fresh_kls), "KL should have grown after moving the LoRA weights"
    print("PASS KL becomes clearly positive and finite once the policy diverges from the reference")

    grad_audit = prod_runtime.gradient_audit(model)
    prod_runtime.assert_gradient_audit_ok(grad_audit)
    print("PASS gradient audit still clean with kl_beta>0:", grad_audit)

    print("ALL KL-PENALTY TESTS PASSED")


if __name__ == "__main__":
    main()
