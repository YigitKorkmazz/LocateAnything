#!/usr/bin/env python3
"""Create immutable, patient-disjoint optimization/validation manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _patient_disease_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["patient_id"])].append(row)
    return groups


def deterministic_stratified_patient_split(rows: list[dict[str, Any]], seed: int, validation_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Patient-level deterministic allocation, stratified as closely as possible by disease.

    Each patient's records stay together.  A seeded shuffle makes ties stable;
    patients are greedily placed where they best reduce disease-count deficit.
    """
    groups = _patient_disease_groups(rows)
    all_counts = Counter(str(row["disease"]) for row in rows)
    targets = {d: max(1, round(n * validation_fraction)) for d, n in all_counts.items()}
    items = []
    for patient, patient_rows in groups.items():
        counts = Counter(str(row["disease"]) for row in patient_rows)
        items.append((patient, patient_rows, counts))
    rng = random.Random(seed)
    rng.shuffle(items)
    # Large/multi-label patients first avoids overshooting a small stratum.
    items.sort(key=lambda item: (-len(item[1]), -len(item[2])))
    validation_patients: set[str] = set(); validation_counts: Counter[str] = Counter()
    target_samples = max(1, round(len(rows) * validation_fraction))
    for patient, patient_rows, counts in items:
        deficit_gain = sum(max(0, targets[d] - validation_counts[d]) for d in counts)
        overshoot = sum(max(0, validation_counts[d] + counts[d] - targets[d]) for d in counts)
        current_samples = sum(validation_counts.values())
        candidate_samples = current_samples + len(patient_rows)
        # Do not fill a disease stratum by substantially overshooting the
        # requested 10% sample allocation merely because a patient has many
        # records.  Patient-disjointness takes priority over exact 79/790.
        choose_validation = candidate_samples <= target_samples and deficit_gain > overshoot
        if current_samples < target_samples and deficit_gain > 0 and abs(candidate_samples - target_samples) < abs(current_samples - target_samples):
            choose_validation = True
        if choose_validation:
            validation_patients.add(patient); validation_counts.update(counts)
    # Protect against unusual tiny strata where greedy allocation is empty.
    if not validation_patients:
        validation_patients.add(items[0][0])
    validation = [row for row in rows if str(row["patient_id"]) in validation_patients]
    optimization = [row for row in rows if str(row["patient_id"]) not in validation_patients]
    return optimization, validation


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"sample_count": len(rows), "patient_count": len({str(r["patient_id"]) for r in rows}),
            "disease_counts": dict(sorted(Counter(str(r["disease"]) for r in rows).items()))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-manifest", required=True); p.add_argument("--test-manifest", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=.10)
    args = p.parse_args()
    if not 0 < args.validation_fraction < 1: raise ValueError("validation fraction must be in (0,1)")
    train, test = read_jsonl(Path(args.train_manifest)), read_jsonl(Path(args.test_manifest))
    train_patients, test_patients = {str(r["patient_id"]) for r in train}, {str(r["patient_id"]) for r in test}
    if train_patients & test_patients: raise RuntimeError("existing train80/test20 split has patient overlap")
    optimization, validation = deterministic_stratified_patient_split(train, args.seed, args.validation_fraction)
    if {str(r["patient_id"]) for r in optimization} & {str(r["patient_id"]) for r in validation}: raise RuntimeError("optimization/validation patient overlap")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"refusing nonempty manifest directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    optimization_path, validation_path = output / "optimization90_of_train80_seed42.jsonl", output / "validation10_of_train80_seed42.jsonl"
    write_jsonl(optimization_path, optimization); write_jsonl(validation_path, validation)
    report = {"format": "chestxray8_patient_disjoint_internal_validation_v1", "seed": args.seed,
              "validation_fraction": args.validation_fraction, "source_train": str(Path(args.train_manifest).resolve()),
              "source_train_sha256": sha256(Path(args.train_manifest)), "heldout_test": str(Path(args.test_manifest).resolve()),
              "heldout_test_sha256": sha256(Path(args.test_manifest)),
              "optimization_manifest": str(optimization_path), "optimization_sha256": sha256(optimization_path),
              "validation_manifest": str(validation_path), "validation_sha256": sha256(validation_path),
              "train80": _stats(train), "optimization": _stats(optimization), "validation": _stats(validation),
              "heldout_test": _stats(test), "patient_overlap": {"optimization_validation": False, "optimization_test": False, "validation_test": False}}
    manifest = output / "split_manifest.json"; manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "optimization_sha256": report["optimization_sha256"], "validation_sha256": report["validation_sha256"], "counts": {k: report[k] for k in ("train80", "optimization", "validation", "heldout_test")}}, sort_keys=True))


if __name__ == "__main__": main()
