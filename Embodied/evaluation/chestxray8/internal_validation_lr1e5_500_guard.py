#!/usr/bin/env python3
"""Apply sealed internal-validation early-stop rules at one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr1e5_500_controlled import (
    INTERNAL_VALIDATION_SHA256,
    RUN_NAME,
    TARGET_STEPS,
)

FORMAT = "lr1e5_500_internal_validation_guard_v1"
DEFAULT_BASE = Path(__file__).resolve().parents[2] / "results/chestxray8_hybrid_grpo_native/G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_100_FUNCTIONAL_KV_CKPT/internal_validation80_step000_025_050_075_100/frozen_base_per_sample.jsonl"


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-step", type=int, choices=TARGET_STEPS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-aggregate", required=True)
    parser.add_argument("--frozen-base-jsonl", default=str(DEFAULT_BASE))
    parser.add_argument("--previous-guard-json")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step = int(args.checkpoint_step)
    checkpoint = Path(args.checkpoint).resolve()
    aggregate_path = Path(args.evaluation_aggregate).resolve()
    base_path = Path(args.frozen_base_jsonl).resolve()
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("split_label") != "internal_validation":
        raise RuntimeError("evaluation is not labeled internal_validation")
    if aggregate.get("manifest_sha256") != INTERNAL_VALIDATION_SHA256:
        raise RuntimeError("evaluation used the wrong manifest")
    if aggregate.get("heldout_test_used") is not False:
        raise RuntimeError("evaluation does not prove held-out exclusion")
    label = f"step_{step:03d}"
    if set(aggregate.get("conditions") or {}) != {label}:
        raise RuntimeError("aggregate must contain exactly the current checkpoint")
    condition = aggregate["conditions"][label]
    if Path(condition["checkpoint"]).resolve() != checkpoint:
        raise RuntimeError("aggregate checkpoint path mismatch")
    if int(condition["aggregate"]["n_samples"]) != 80:
        raise RuntimeError("evaluation does not contain exactly 80 samples")

    current_rows = _read_jsonl(Path(condition["per_sample_jsonl"]).resolve())
    base_rows = _read_jsonl(base_path)
    if len(current_rows) != 80 or len(base_rows) != 80:
        raise RuntimeError("base/current per-sample files must each have 80 rows")
    identity = lambda row: (
        row["sample_index"], row["image_index"], row["disease"], row["seed"]
    )
    if any(identity(left) != identity(right) for left, right in zip(base_rows, current_rows)):
        raise RuntimeError("current evaluation is not aligned with frozen base")

    base_mean_iou = float(statistics.fmean(row["iou"] for row in base_rows))
    metrics = condition["aggregate"]["metrics"]
    current = {
        "checkpoint_step": step,
        "mean_iou": float(metrics["mean_iou"]["point"]),
        "near_full_box_rate": float(metrics["near_full_image_box_rate"]["point"]),
    }
    history = []
    previous_receipt = None
    if args.previous_guard_json:
        previous_path = Path(args.previous_guard_json).resolve()
        previous_receipt = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_receipt.get("format") != FORMAT or previous_receipt.get("run_name") != RUN_NAME:
            raise RuntimeError("previous guard receipt contract mismatch")
        if int(previous_receipt.get("evaluated_checkpoint_step", -1)) != step - 100:
            raise RuntimeError("previous guard is not the immediately preceding checkpoint")
        if previous_receipt.get("decision") != "continue":
            raise RuntimeError("cannot extend a stopped/completed guard history")
        history = list(previous_receipt.get("history") or [])
    elif step != 100:
        raise RuntimeError("step>100 requires the immediately previous guard receipt")
    history.append(current)

    previous_mean = base_mean_iou if len(history) == 1 else float(history[-2]["mean_iou"])
    near_full_nonimproving = bool(
        current["near_full_box_rate"] > 0.60
        and current["mean_iou"] <= previous_mean
    )
    two_below_base = bool(
        len(history) >= 2
        and float(history[-1]["mean_iou"]) < base_mean_iou
        and float(history[-2]["mean_iou"]) < base_mean_iou
    )
    triggered = []
    if near_full_nonimproving:
        triggered.append("near_full_gt_60pct_and_mean_iou_not_improving")
    if two_below_base:
        triggered.append("mean_iou_below_frozen_base_two_consecutive_checkpoints")
    decision = "stop" if triggered else ("complete" if step == 500 else "continue")
    next_target = step + 100 if decision == "continue" else None
    receipt = {
        "format": FORMAT,
        "run_name": RUN_NAME,
        "decision": decision,
        "triggered_guards": triggered,
        "evaluated_checkpoint_step": step,
        "authorized_next_target_step": next_target,
        "internal_validation_sha256": INTERNAL_VALIDATION_SHA256,
        "heldout_test_used": False,
        "frozen_base": {
            "per_sample_jsonl": str(base_path),
            "sha256": _sha256(base_path),
            "mean_iou": base_mean_iou,
        },
        "evaluated_checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "evaluation_aggregate": {
            "path": str(aggregate_path),
            "sha256": _sha256(aggregate_path),
        },
        "current_metrics": current,
        "previous_checkpoint_mean_iou": previous_mean,
        "guards": {
            "near_full_gt_60pct_and_nonimproving": near_full_nonimproving,
            "two_consecutive_mean_iou_below_frozen_base": two_below_base,
        },
        "history": history,
        "procedural_only_no_training_or_reward_mutation": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

