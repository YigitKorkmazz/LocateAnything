#!/usr/bin/env python3
"""Regression test for the deterministic seed-42 train80 manifest."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from prepare_sft_dataset import write_validated_train80_manifest  # noqa: E402
from sft_common import read_jsonl  # noqa: E402

EXPECTED_SHA256 = "a2b1c25f652f90e8d93e58b33ee74aa5eb09db4df7c6b37e760f5e5238c2e3b0"


def test_seed42_train80_manifest():
    split_dir = CHEST_DIR / "splits"
    train = read_jsonl(split_dir / "train_pairs_seed42.jsonl")
    val = read_jsonl(split_dir / "val_pairs_seed42.jsonl")
    test = read_jsonl(split_dir / "test_pairs_seed42.jsonl")
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "train80_pairs_seed42.jsonl"
        report = write_validated_train80_manifest(
            train, val, test, output, seed=42
        )
        assert report["n_pairs"] == 790
        assert report["n_patients"] == 581
        assert report["n_images"] == 707
        assert report["n_duplicate_pairs"] == 0
        assert report["train_test_patient_overlap"] == []
        assert report["train_test_image_overlap"] == []
        assert report["sha256"] == EXPECTED_SHA256
        assert hashlib.sha256(output.read_bytes()).hexdigest() == EXPECTED_SHA256

        rows = [
            json.loads(line)
            for line in output.read_text().splitlines()
            if line.strip()
        ]
        keys = [(row["image_index"], row["disease"]) for row in rows]
        assert keys == sorted(keys)


def main() -> None:
    test_seed42_train80_manifest()
    print("TRAIN80 SPLIT TEST PASSED")


if __name__ == "__main__":
    main()
