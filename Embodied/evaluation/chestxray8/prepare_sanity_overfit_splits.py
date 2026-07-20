#!/usr/bin/env python3
"""Build and verify ChestX-ray8 sanity-overfit manifests (20% / 50% / 100%).

Stage A uses the existing held-out test split unchanged as the 20% memorization set.
Stage B = that 20% plus a seed-42 patient-level additional ~30% of the full bbox set.
Stage C = every valid bbox image-disease pair for the eight ChestX-ray8 classes.

Does not run training or evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from PIL import Image

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from eval_locateanything_bbox import (  # noqa: E402
    CHESTXRAY8_DISEASES,
    build_image_index,
    load_bbox_annotations,
)
from prepare_sft_dataset import build_pair_records, load_patient_id_map  # noqa: E402
from sft_common import (  # noqa: E402
    DEFAULT_BBOX_CSV,
    DEFAULT_DATASET_PATH,
    DEFAULT_SEED,
    PROMPT_STRATEGY,
    class_distribution,
    read_jsonl,
    refresh_pair_prompt_fields,
    write_jsonl,
)

logger = logging.getLogger("sanity_overfit_splits")

PAIR_KEY = Tuple[str, str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_key(p: Dict[str, Any]) -> PAIR_KEY:
    return (str(p["image_index"]), str(p["disease"]))


def refresh_all(pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in pairs:
        r = refresh_pair_prompt_fields(dict(p), PROMPT_STRATEGY)
        # Force disease-consistent prompt bookkeeping.
        if r["disease"] not in r["user_query"]:
            raise RuntimeError(
                f"Prompt disease mismatch for {pair_key(r)}: query={r['user_query']!r}"
            )
        out.append(r)
    return out


def summarize(pairs: Sequence[Dict[str, Any]], label: str) -> Dict[str, Any]:
    images = {p["image_index"] for p in pairs}
    patients = {str(p.get("patient_id")) for p in pairs if p.get("patient_id") is not None}
    return {
        "label": label,
        "n_pairs": len(pairs),
        "n_unique_images": len(images),
        "n_unique_patients": len(patients),
        "class_distribution": class_distribution(pairs),
        "prompt_strategy": PROMPT_STRATEGY,
    }


def assert_no_dup_pairs(pairs: Sequence[Dict[str, Any]], label: str) -> None:
    keys = [pair_key(p) for p in pairs]
    counts = Counter(keys)
    dups = [k for k, c in counts.items() if c > 1]
    if dups:
        raise RuntimeError(f"{label}: duplicate image-disease pairs: {dups[:10]}")


def select_additional_30pct_patient_level(
    pool_pairs: Sequence[Dict[str, Any]],
    exclude_keys: Set[PAIR_KEY],
    target_n_pairs: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Patient-level draw from pool until ~target_n_pairs; never reuse exclude_keys."""
    import random

    rng = random.Random(seed)
    by_patient: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pool_pairs:
        if pair_key(p) in exclude_keys:
            continue
        by_patient[str(p["patient_id"])].append(p)

    patients = sorted(by_patient.keys())
    rng.shuffle(patients)

    selected: List[Dict[str, Any]] = []
    selected_images: Set[str] = set()
    for pid in patients:
        group = by_patient[pid]
        # Prefer whole-patient inclusion (no image-level leakage within addition).
        imgs = {g["image_index"] for g in group}
        if imgs & selected_images:
            # Should not happen with patient grouping; skip defensively.
            continue
        selected.extend(group)
        selected_images |= imgs
        if len(selected) >= target_n_pairs:
            break

    if len(selected) < target_n_pairs:
        raise RuntimeError(
            f"Could only select {len(selected)} additional pairs "
            f"(target {target_n_pairs}) from remaining patients"
        )
    return selected


def build_all_bbox_pairs(
    dataset_path: Path,
    bbox_csv: Path,
    logger_: logging.Logger,
) -> List[Dict[str, Any]]:
    annotations, _, _ = load_bbox_annotations(bbox_csv, CHESTXRAY8_DISEASES, logger_)
    needed = {img for img, _ in annotations}
    image_index = build_image_index(dataset_path, logger_, needed_names=needed)
    patient_ids = load_patient_id_map(dataset_path, needed, logger_)
    pairs = build_pair_records(annotations, image_index, patient_ids, logger_)
    return refresh_all(pairs)


def write_manifest(path: Path, pairs: Sequence[Dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, pairs)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--bbox-csv", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--split-dir",
        type=str,
        default=str(CHEST_DIR / "splits"),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "sanity_overfit"),
    )
    parser.add_argument(
        "--expected-20-sha256",
        type=str,
        default=None,
        help="Optional expected SHA256 of test_pairs_seed42.jsonl; fail if mismatch.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    split_dir = Path(args.split_dir)
    results_dir = Path(args.results_dir)
    dataset_path = Path(args.dataset_path)
    bbox_csv = Path(args.bbox_csv) if args.bbox_csv else dataset_path / DEFAULT_BBOX_CSV

    test_path = split_dir / f"test_pairs_seed{args.seed}.jsonl"
    train_path = split_dir / f"train_pairs_seed{args.seed}.jsonl"
    val_path = split_dir / f"val_pairs_seed{args.seed}.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(f"Missing 20% test split: {test_path}")

    sha20_before = sha256_file(test_path)
    logger.info("Original 20%% manifest: %s", test_path)
    logger.info("SHA256(20%% before)=%s", sha20_before)
    if args.expected_20_sha256 and args.expected_20_sha256 != sha20_before:
        raise RuntimeError(
            f"20% checksum mismatch: expected {args.expected_20_sha256}, got {sha20_before}"
        )

    test_pairs_raw = read_jsonl(test_path)
    test_pairs = refresh_all(test_pairs_raw)
    assert_no_dup_pairs(test_pairs, "20%")
    # Never rewrite the original test file; stage A only uses it as-is.
    sha20_after = sha256_file(test_path)
    if sha20_after != sha20_before:
        raise RuntimeError("CRITICAL: test_pairs_seed42.jsonl was modified")
    logger.info("Confirmed original 20%% file unchanged (checksum match)")

    # Full bbox universe for 50%/100%.
    logger.info("Building full bbox pair universe from %s", bbox_csv)
    all_pairs = build_all_bbox_pairs(dataset_path, bbox_csv, logger)
    assert_no_dup_pairs(all_pairs, "100%")
    n_all = len(all_pairs)
    logger.info("Full bbox universe size: %d", n_all)

    test_keys = {pair_key(p) for p in test_pairs}
    all_by_key = {pair_key(p): p for p in all_pairs}
    missing_in_universe = sorted(test_keys - set(all_by_key))
    if missing_in_universe:
        raise RuntimeError(
            f"20% pairs missing from rebuilt universe: {missing_in_universe[:10]}"
        )

    # Universe rows for the original 20% keys (path-consistent with 50%/100%).
    order = {pair_key(p): i for i, p in enumerate(test_pairs)}
    stage20 = sorted(
        [all_by_key[k] for k in test_keys],
        key=lambda p: order[pair_key(p)],
    )

    # Stage A working copy under results: byte-identical to the original test split.
    # Do not rewrite the original held-out file. Train/eval still rebuild prompts
    # at runtime via refresh_pair_prompt_fields / build_final_user_query.
    overfit20_dir = results_dir / "overfit_20"
    overfit20_dir.mkdir(parents=True, exist_ok=True)
    stage20_path = overfit20_dir / "train_eval_pairs.jsonl"
    stage20_path.write_bytes(test_path.read_bytes())
    sha20_copy = sha256_file(stage20_path)
    if sha20_copy != sha20_before:
        raise RuntimeError("Stage-A copy is not byte-identical to original 20%")
    (overfit20_dir / "source_test_split.txt").write_text(str(test_path.resolve()) + "\n")
    (overfit20_dir / "checksums.json").write_text(
        json.dumps(
            {
                "original_test_pairs_path": str(test_path.resolve()),
                "original_test_pairs_sha256": sha20_before,
                "stage20_copy_path": str(stage20_path.resolve()),
                "stage20_copy_sha256": sha20_copy,
                "byte_identical_to_original": True,
                "note": "Stage A train/eval must use original_test_pairs_path "
                "(or the byte-identical stage20 copy).",
            },
            indent=2,
        )
        + "\n"
    )

    # Stage B: 20% + additional ~30% of full dataset (patient-level from remainder).
    target_extra = int(round(0.30 * n_all))
    pool = [p for p in all_pairs if pair_key(p) not in test_keys]
    # Also ensure pool does not share images with the 20% set (stricter than pairs).
    test_images = {p["image_index"] for p in stage20}
    pool = [p for p in pool if p["image_index"] not in test_images]
    extra = select_additional_30pct_patient_level(
        pool, exclude_keys=test_keys, target_n_pairs=target_extra, seed=args.seed
    )
    extra_images = {p["image_index"] for p in extra}
    if extra_images & test_images:
        raise RuntimeError("Image leakage between 20% and additional 30%")
    stage50 = stage20 + extra
    assert_no_dup_pairs(stage50, "50%")
    if not test_keys.issubset({pair_key(p) for p in stage50}):
        raise RuntimeError("Original 20% is NOT a strict subset of the 50% set")

    stage50_path = split_dir / f"sanity_overfit_50_seed{args.seed}.jsonl"
    sha50 = write_manifest(stage50_path, stage50)
    overfit50_dir = results_dir / "overfit_50"
    overfit50_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(overfit50_dir / "train_eval_pairs.jsonl", stage50)
    write_manifest(overfit50_dir / "original_20_subset.jsonl", stage20)

    # Stage C: 100%
    stage100_path = split_dir / f"sanity_overfit_100_seed{args.seed}.jsonl"
    sha100 = write_manifest(stage100_path, all_pairs)
    overfit100_dir = results_dir / "overfit_100"
    overfit100_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(overfit100_dir / "train_eval_pairs.jsonl", all_pairs)
    write_manifest(overfit100_dir / "original_20_subset.jsonl", stage20)

    # Prefer existing train+val+test for reporting "existing split coverage"
    existing = []
    for pth in (train_path, val_path, test_path):
        if pth.exists():
            existing.extend(read_jsonl(pth))
    existing_keys = {pair_key(p) for p in existing}

    report = {
        "seed": args.seed,
        "prompt_strategy": PROMPT_STRATEGY,
        "classes": list(CHESTXRAY8_DISEASES),
        "checksums": {
            "overfit_20_original_test_pairs": sha20_before,
            "overfit_20_stage_copy": sha20_copy,
            "overfit_50": sha50,
            "overfit_100": sha100,
        },
        "paths": {
            "overfit_20_source": str(test_path.resolve()),
            "overfit_20_copy": str(stage20_path.resolve()),
            "overfit_50": str(stage50_path.resolve()),
            "overfit_100": str(stage100_path.resolve()),
        },
        "summaries": {
            "overfit_20": summarize(stage20, "overfit_20"),
            "overfit_50": {
                **summarize(stage50, "overfit_50"),
                "n_original_20_pairs": len(stage20),
                "n_additional_30_pairs": len(extra),
                "target_additional_pairs": target_extra,
                "original_20_is_strict_subset": True,
                "image_overlap_20_and_extra": 0,
            },
            "overfit_100": summarize(all_pairs, "overfit_100"),
        },
        "integrity": {
            "original_20_file_unchanged": True,
            "only_eight_classes_100": set(class_distribution(all_pairs))
            <= set(CHESTXRAY8_DISEASES),
            "all_20_keys_in_100": test_keys.issubset(set(all_by_key)),
            "all_20_keys_in_50": True,
            "existing_split_pair_count": len(existing),
            "existing_vs_universe_key_diff": len(set(all_by_key) ^ existing_keys),
        },
    }
    summary_path = results_dir / "split_integrity_report.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n")

    print("=" * 72)
    print("SANITY OVERFIT SPLIT REPORT")
    print("=" * 72)
    for label in ("overfit_20", "overfit_50", "overfit_100"):
        s = report["summaries"][label]
        print(f"\n[{label}]")
        print(f"  pairs            : {s['n_pairs']}")
        print(f"  unique images    : {s['n_unique_images']}")
        print(f"  unique patients  : {s['n_unique_patients']}")
        print(f"  class_distribution: {s['class_distribution']}")
        if label == "overfit_50":
            print(f"  original_20_pairs : {s['n_original_20_pairs']}")
            print(f"  additional_30_pairs: {s['n_additional_30_pairs']}")
            print(f"  20% strict subset : {s['original_20_is_strict_subset']}")
    print("\nCHECKSUMS")
    for k, v in report["checksums"].items():
        print(f"  {k}: {v}")
    print(f"\nWrote integrity report: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
