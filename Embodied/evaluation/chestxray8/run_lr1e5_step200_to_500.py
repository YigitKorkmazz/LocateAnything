#!/usr/bin/env python3
"""Resume the controlled LR=1e-5 run exactly from step 200 to step 500."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr1e5_500_controlled as c  # noqa: E402


RESUME_CHECKPOINT = Path(
    "/auto/k2/ykorkmaz/LocateAnything-playground/Embodied/results/"
    "chestxray8_hybrid_grpo_native/"
    "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_500_"
    "CONTROLLED_FUNCTIONAL_KV_CKPT/segment_100_200/"
    "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_step_200.pt"
)
OUTPUT_DIR = Path(
    "/auto/k2/ykorkmaz/LocateAnything-playground/Embodied/results/"
    "chestxray8_hybrid_grpo_native/"
    "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_500_"
    "CONTROLLED_FUNCTIONAL_KV_CKPT/segment_200_500"
)
PREFLIGHT_RECEIPT = Path(
    "/auto/k2/ykorkmaz/LocateAnything-playground/Embodied/results/"
    "chestxray8_hybrid_grpo_native/preflight_receipts/"
    "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_50_"
    "checkpoint_preflight.json"
)


def main() -> None:
    p = c.production
    a = c.lr_ablation
    cfg = c.resolve_experiment_config(c.DEFAULT_SPEC)
    c.validate_long_run_contract(cfg)
    a.validate_checkpoint_preflight_receipt(PREFLIGHT_RECEIPT, expected_lr=1e-5)

    resume = RESUME_CHECKPOINT.resolve()
    meta = c._checkpoint_metadata(resume)
    assert meta["global_step"] == 200
    assert meta["seed"] == 42
    assert meta["lora_tensors"] == 504
    assert meta["projector_tensors"] == 0
    assert meta["trainable_tensors"] == 504
    assert meta["runtime_contract"] == p.EXPECTED_RUNTIME_CONTRACT
    assert meta["kl_contract"] == {"enabled": False, "beta": 0.0}

    p.GROUP_SIZE = 8
    p.EXPECTED_LORA_TENSORS = 504
    p.EXPECTED_PROJECTOR_TENSORS = 0
    p.EXPECTED_TRAINABLE_TENSORS = 504
    p.METRICS_FILENAME = (
        "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_metrics.jsonl"
    )
    p.SUMMARY_FILENAME = (
        "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_summary.json"
    )
    p.CHECKPOINT_PREFIX = "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_step_"
    p.PRE_REPLAY_TRACE_METADATA_FILENAME = (
        "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_"
        "pre_replay_trace_metadata.jsonl"
    )
    p.PERSIST_PRE_REPLAY_TRACE_METADATA = True
    p.SEMANTICS_PRESERVING_REPLAY_CUDA_CLEANUP = True
    p.PER_REPLAY_PEAK_MEMORY_DIAGNOSTICS = True
    p.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True
    p.load_resolved_config = lambda unused: cfg
    p._validate_config = c.validate_long_run_contract
    p._optimizer_contract_report = a._optimizer_contract_validator(1e-5)

    sys.argv = [
        sys.argv[0],
        "--config",
        str(Path(c.DEFAULT_SPEC).resolve()),
        "--output-dir",
        str(OUTPUT_DIR.resolve()),
        "--max-optimizer-steps",
        "500",
        "--max-attempted-groups",
        "4000",
        "--checkpoint-interval",
        "100",
        "--optimization-manifest",
        str(a.OPTIMIZATION.resolve()),
        "--optimization-manifest-sha256",
        a.OPTIMIZATION_SHA256,
        "--validation-manifest",
        str(a.VALIDATION.resolve()),
        "--validation-manifest-sha256",
        a.VALIDATION_SHA256,
        "--seed",
        "42",
        "--resume-checkpoint",
        str(resume),
    ]
    p.main()


if __name__ == "__main__":
    main()
