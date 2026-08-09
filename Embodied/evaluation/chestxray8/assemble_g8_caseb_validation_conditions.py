#!/usr/bin/env python3
"""Assemble complete per-condition internal-validation outputs after isolated runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_two_gpu_hybrid_grpo_checkpoints import _aggregate  # noqa: E402

MANIFEST = HERE / "splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl"
MANIFEST_SHA256 = "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838"
TRAINING = HERE / "results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42"
ORDER = ["frozen_base", "step_025", "step_050", "step_075", "step_100", "step_125", "step_150"]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial-output-dir", required=True)
    parser.add_argument("--step100-aggregate", required=True)
    parser.add_argument("--step125-aggregate", required=True)
    parser.add_argument("--step150-aggregate", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()
    output = Path(args.partial_output_dir).resolve()
    destination = output / "aggregate_metrics.json"
    if destination.exists():
        raise FileExistsError(destination)

    conditions = {}
    for index, label in enumerate(ORDER[:4]):
        path = output / (label + "_per_sample.jsonl")
        rows = read_jsonl(path)
        if len(rows) != 80:
            raise RuntimeError("{} does not have 80 rows".format(label))
        checkpoint = None if label == "frozen_base" else str((TRAINING / ("two_gpu_g8_caseb_{}.pt".format(label.replace("step_", "step_")))).resolve())
        conditions[label] = {
            "checkpoint": checkpoint,
            "model_revision": "c32291ca5e996f5a7a485845b4f57a233936bba0",
            "per_sample_jsonl": str(path.resolve()),
            "aggregate": _aggregate(rows, args.bootstrap_replicates, args.bootstrap_seed + index),
        }

    evaluation_commands = [
        "CUDA_VISIBLE_DEVICES=0,1 initial seven-condition command; completed frozen_base/step_025/step_050/step_075 before cross-condition allocator OOM"
    ]
    for expected, aggregate_path in zip(
        ORDER[4:],
        (args.step100_aggregate, args.step125_aggregate, args.step150_aggregate),
    ):
        isolated = json.loads(Path(aggregate_path).read_text())
        if isolated["manifest_sha256"] != MANIFEST_SHA256 or isolated["split_label"] != "internal_validation":
            raise RuntimeError("isolated condition used wrong manifest")
        if list(isolated["conditions"]) != [expected]:
            raise RuntimeError("isolated condition label mismatch")
        item = isolated["conditions"][expected]
        if len(read_jsonl(Path(item["per_sample_jsonl"]))) != 80:
            raise RuntimeError("isolated condition does not have 80 rows")
        conditions[expected] = item
        evaluation_commands.append(isolated["command"])

    report = {
        "evaluation": "deterministic_seeded_native_hybrid_commit",
        "assembly": "four completed conditions from initial process plus three fresh isolated processes to prevent cross-model CUDA allocator accumulation",
        "config_path": str((HERE / "rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml").resolve()),
        "split_label": "internal_validation",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": MANIFEST_SHA256,
        "same_samples_seeds_decoding_reward_code": True,
        "bootstrap_replicates": args.bootstrap_replicates,
        "heldout_test_used": False,
        "command": evaluation_commands[0],
        "evaluation_commands": evaluation_commands,
        "assembly_command": " ".join(sys.argv),
        "conditions": conditions,
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"aggregate": str(destination), "conditions": list(conditions)}, indent=2))


if __name__ == "__main__":
    main()
