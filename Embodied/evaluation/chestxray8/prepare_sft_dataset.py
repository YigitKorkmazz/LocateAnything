#!/usr/bin/env python3
"""
Prepare ChestX-ray8 bounding-box SFT data and a fixed patient-level 80/20 split.

Writes:
  evaluation/chestxray8/splits/train_pairs_seed42.jsonl
  evaluation/chestxray8/splits/test_pairs_seed42.jsonl
  evaluation/chestxray8/splits/val_pairs_seed42.jsonl   (optional, from train only)
  evaluation/chestxray8/splits/split_summary_seed42.json
  evaluation/chestxray8/data/train_sharegpt_seed42.jsonl
  evaluation/chestxray8/data/val_sharegpt_seed42.jsonl
  evaluation/chestxray8/data/test_sharegpt_seed42.jsonl
  evaluation/chestxray8/data/recipe_train_seed42.json
  evaluation/chestxray8/data/recipe_val_seed42.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CHEST_DIR = Path(__file__).resolve().parent
if str(CHEST_DIR) not in sys.path:
    sys.path.insert(0, str(CHEST_DIR))

from sft_common import (  # noqa: E402
    CHESTXRAY8_DISEASES,
    DEFAULT_BBOX_CSV,
    DEFAULT_DATASET_PATH,
    DEFAULT_SEED,
    PROMPT_STRATEGY,
    build_assistant_target,
    build_final_user_query,
    build_image_index,
    build_sharegpt_sample,
    class_distribution,
    load_bbox_annotations,
    patient_id_from_image_index,
    set_global_seed,
    write_jsonl,
    xyxy_pixels_to_norm1000,
)


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("prepare_sft_dataset")


def load_patient_id_map(
    dataset_path: Path,
    image_names: Set[str],
    logger: logging.Logger,
) -> Dict[str, str]:
    """Prefer Data_Entry_2017.csv Patient ID; fall back to filename prefix."""
    mapping: Dict[str, str] = {}
    entry_csv = dataset_path / "Data_Entry_2017.csv"
    if entry_csv.is_file():
        import csv

        with entry_csv.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            # Image Index, Finding Labels, Follow-up #, Patient ID, ...
            try:
                img_idx = header.index("Image Index")
                pid_idx = header.index("Patient ID")
            except ValueError:
                img_idx, pid_idx = 0, 3
            for row in reader:
                if len(row) <= max(img_idx, pid_idx):
                    continue
                name = row[img_idx].strip()
                if name in image_names:
                    mapping[name] = str(row[pid_idx]).strip()
        logger.info(
            "Loaded Patient ID for %d / %d images from Data_Entry_2017.csv",
            len(mapping),
            len(image_names),
        )
    missing = image_names - set(mapping)
    for name in missing:
        mapping[name] = patient_id_from_image_index(name)
    if missing:
        logger.info(
            "Derived Patient ID from filename for %d images (e.g. %s -> %s)",
            len(missing),
            next(iter(missing)),
            mapping[next(iter(missing))],
        )
    return mapping


def build_pair_records(
    annotations: Dict[Tuple[str, str], List[List[float]]],
    image_index: Dict[str, Path],
    patient_ids: Dict[str, str],
    logger: logging.Logger,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    skipped_missing = 0
    for (image_name, disease), gt_boxes in sorted(annotations.items()):
        if image_name not in image_index:
            skipped_missing += 1
            continue
        image_path = image_index[image_name]
        with Image.open(image_path) as im:
            width, height = im.size
        phrase, user_query = build_final_user_query(disease, PROMPT_STRATEGY)
        boxes_norm = xyxy_pixels_to_norm1000(gt_boxes, width, height)
        assistant_target = build_assistant_target(phrase, boxes_norm)
        pairs.append(
            {
                "image_index": image_name,
                "patient_id": str(patient_ids[image_name]),
                "disease": disease,
                "image_path": str(image_path.resolve()),
                "prompt_strategy": PROMPT_STRATEGY,
                "prompt_phrase": phrase,
                "user_query": user_query,
                "gt_boxes_xyxy_px": [[float(v) for v in b] for b in gt_boxes],
                "gt_boxes_norm_1000": boxes_norm,
                "image_width": int(width),
                "image_height": int(height),
                "assistant_target": assistant_target,
                "n_gt_boxes": len(gt_boxes),
            }
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    if skipped_missing:
        logger.warning("Skipped %d pairs with missing image files", skipped_missing)
    return pairs


def patient_level_split(
    pairs: Sequence[Dict[str, Any]],
    seed: int,
    train_ratio: float = 0.8,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Random 80/20 split by patient ID (no image/patient leakage)."""
    rng = __import__("random").Random(seed)
    patient_to_pairs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pairs:
        patient_to_pairs[p["patient_id"]].append(p)
    patients = sorted(patient_to_pairs.keys())
    rng.shuffle(patients)
    n_train_patients = int(round(len(patients) * train_ratio))
    # Ensure both sides non-empty when possible.
    n_train_patients = max(1, min(len(patients) - 1, n_train_patients)) if len(patients) > 1 else len(patients)
    train_patients = set(patients[:n_train_patients])
    test_patients = set(patients[n_train_patients:])
    train_pairs = [p for pid in sorted(train_patients) for p in patient_to_pairs[pid]]
    test_pairs = [p for pid in sorted(test_patients) for p in patient_to_pairs[pid]]
    # Stable ordering
    train_pairs = sorted(train_pairs, key=lambda x: (x["image_index"], x["disease"]))
    test_pairs = sorted(test_pairs, key=lambda x: (x["image_index"], x["disease"]))
    return train_pairs, test_pairs


def carve_validation_from_train(
    train_pairs: Sequence[Dict[str, Any]],
    seed: int,
    val_fraction_of_train: float = 0.1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split validation from the 80% train partition only (patient-level).

    Default 0.1 of train ≈ 8% overall when train is 80% (72/8/20).
    """
    rng = __import__("random").Random(seed + 7)
    patient_to_pairs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in train_pairs:
        patient_to_pairs[p["patient_id"]].append(p)
    patients = sorted(patient_to_pairs.keys())
    rng.shuffle(patients)
    n_val = int(round(len(patients) * val_fraction_of_train))
    n_val = max(1, min(len(patients) - 1, n_val)) if len(patients) > 1 else 0
    val_patients = set(patients[:n_val])
    remain_patients = set(patients[n_val:])
    val_pairs = [p for pid in sorted(val_patients) for p in patient_to_pairs[pid]]
    new_train = [p for pid in sorted(remain_patients) for p in patient_to_pairs[pid]]
    val_pairs = sorted(val_pairs, key=lambda x: (x["image_index"], x["disease"]))
    new_train = sorted(new_train, key=lambda x: (x["image_index"], x["disease"]))
    return new_train, val_pairs


def assert_no_leakage(
    train_pairs: Sequence[Dict[str, Any]],
    test_pairs: Sequence[Dict[str, Any]],
    val_pairs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    train_images = {p["image_index"] for p in train_pairs}
    test_images = {p["image_index"] for p in test_pairs}
    train_patients = {p["patient_id"] for p in train_pairs}
    test_patients = {p["patient_id"] for p in test_pairs}
    image_overlap = sorted(train_images & test_images)
    patient_overlap = sorted(train_patients & test_patients)
    report = {
        "train_test_image_overlap": image_overlap,
        "train_test_patient_overlap": patient_overlap,
        "train_test_leakage_free": len(image_overlap) == 0 and len(patient_overlap) == 0,
    }
    if val_pairs is not None:
        val_images = {p["image_index"] for p in val_pairs}
        val_patients = {p["patient_id"] for p in val_pairs}
        report["train_val_image_overlap"] = sorted(train_images & val_images)
        report["train_val_patient_overlap"] = sorted(train_patients & val_patients)
        report["val_test_image_overlap"] = sorted(val_images & test_images)
        report["val_test_patient_overlap"] = sorted(val_patients & test_patients)
        report["val_leakage_free"] = (
            len(report["train_val_image_overlap"]) == 0
            and len(report["train_val_patient_overlap"]) == 0
            and len(report["val_test_image_overlap"]) == 0
            and len(report["val_test_patient_overlap"]) == 0
        )
    return report


def pairs_to_sharegpt(pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        build_sharegpt_sample(
            image_path=p["image_path"],
            phrase=p["prompt_phrase"],
            user_query=p["user_query"],
            assistant_target=p["assistant_target"],
        )
        for p in pairs
    ]


def write_recipe(path: Path, annotation_jsonl: Path, name: str) -> None:
    recipe = {
        name: {
            "annotation": str(annotation_jsonl.resolve()),
            "root": "",
            "repeat_time": 1.0,
            "data_augment": False,
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe, indent=2) + "\n", encoding="utf-8")


def print_five_sample_validation(pairs: Sequence[Dict[str, Any]]) -> None:
    print("=" * 72)
    print("DATASET CREATION VALIDATION (5 samples)")
    print("=" * 72)
    for i, p in enumerate(pairs[:5]):
        print(f"\n--- sample {i} ---")
        print(f"image_index     : {p['image_index']}")
        print(f"patient_id      : {p['patient_id']}")
        print(f"disease         : {p['disease']}")
        print(f"image_path      : {p['image_path']}")
        print(f"size            : {p['image_width']} x {p['image_height']}")
        print(f"user_query      : {p['user_query']}")
        print(f"gt_boxes_px     : {p['gt_boxes_xyxy_px']}")
        print(f"gt_boxes_norm   : {p['gt_boxes_norm_1000']}")
        print(f"assistant_target: {p['assistant_target']}")
    print("=" * 72)


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
        "--data-dir",
        type=str,
        default=str(CHEST_DIR / "data"),
    )
    parser.add_argument(
        "--val-fraction-of-train",
        type=float,
        default=0.1,
        help="Fraction of train patients held out as validation (default 0.1 => ~72/8/20).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on total image-disease pairs (debug).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files. Without this flag, existing seed42 splits are reused.",
    )
    args = parser.parse_args()

    logger = setup_logger()
    set_global_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    csv_path = Path(args.bbox_csv) if args.bbox_csv else dataset_path / DEFAULT_BBOX_CSV
    split_dir = Path(args.split_dir)
    data_dir = Path(args.data_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_split_path = split_dir / f"train_pairs_seed{args.seed}.jsonl"
    test_split_path = split_dir / f"test_pairs_seed{args.seed}.jsonl"
    val_split_path = split_dir / f"val_pairs_seed{args.seed}.jsonl"
    summary_path = split_dir / f"split_summary_seed{args.seed}.json"

    if train_split_path.exists() and test_split_path.exists() and not args.overwrite:
        logger.info(
            "Existing splits found at %s / %s — reusing (pass --overwrite to rebuild).",
            train_split_path,
            test_split_path,
        )
        from sft_common import read_jsonl

        train_pairs = read_jsonl(train_split_path)
        test_pairs = read_jsonl(test_split_path)
        val_pairs = read_jsonl(val_split_path) if val_split_path.exists() else []
        all_pairs = train_pairs + val_pairs + test_pairs
    else:
        logger.info("Loading annotations from %s", csv_path)
        annotations, _, _ = load_bbox_annotations(csv_path, CHESTXRAY8_DISEASES, logger)
        needed = {img for img, _ in annotations}
        image_index = build_image_index(dataset_path, logger, needed_names=needed)
        patient_ids = load_patient_id_map(dataset_path, needed, logger)
        all_pairs = build_pair_records(
            annotations, image_index, patient_ids, logger, max_pairs=args.max_pairs
        )
        print_five_sample_validation(all_pairs)

        train80, test_pairs = patient_level_split(all_pairs, seed=args.seed, train_ratio=0.8)
        train_pairs, val_pairs = carve_validation_from_train(
            train80, seed=args.seed, val_fraction_of_train=args.val_fraction_of_train
        )

        write_jsonl(train_split_path, train_pairs)
        write_jsonl(test_split_path, test_pairs)
        write_jsonl(val_split_path, val_pairs)
        logger.info("Wrote permanent splits to %s", split_dir)

    leakage = assert_no_leakage(train_pairs, test_pairs, val_pairs)
    if not leakage["train_test_leakage_free"]:
        raise RuntimeError(f"Train/test leakage detected: {leakage}")
    if val_pairs and not leakage.get("val_leakage_free", True):
        raise RuntimeError(f"Validation leakage detected: {leakage}")

    # ShareGPT + recipes (always refresh from split files so formats stay in sync)
    train_sharegpt = data_dir / f"train_sharegpt_seed{args.seed}.jsonl"
    val_sharegpt = data_dir / f"val_sharegpt_seed{args.seed}.jsonl"
    test_sharegpt = data_dir / f"test_sharegpt_seed{args.seed}.jsonl"
    write_jsonl(train_sharegpt, pairs_to_sharegpt(train_pairs))
    write_jsonl(val_sharegpt, pairs_to_sharegpt(val_pairs))
    write_jsonl(test_sharegpt, pairs_to_sharegpt(test_pairs))
    write_recipe(data_dir / f"recipe_train_seed{args.seed}.json", train_sharegpt, "chestxray8_train")
    write_recipe(data_dir / f"recipe_val_seed{args.seed}.json", val_sharegpt, "chestxray8_val")

    summary = {
        "seed": args.seed,
        "prompt_strategy": PROMPT_STRATEGY,
        "split_unit": "patient_id",
        "total_image_disease_pairs": len(train_pairs) + len(val_pairs) + len(test_pairs),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_test_pairs": len(test_pairs),
        "n_unique_train_images": len({p["image_index"] for p in train_pairs}),
        "n_unique_val_images": len({p["image_index"] for p in val_pairs}),
        "n_unique_test_images": len({p["image_index"] for p in test_pairs}),
        "n_unique_train_patients": len({p["patient_id"] for p in train_pairs}),
        "n_unique_val_patients": len({p["patient_id"] for p in val_pairs}),
        "n_unique_test_patients": len({p["patient_id"] for p in test_pairs}),
        "class_distribution_train": class_distribution(train_pairs),
        "class_distribution_val": class_distribution(val_pairs),
        "class_distribution_test": class_distribution(test_pairs),
        "leakage_check": leakage,
        "paths": {
            "train_pairs": str(train_split_path),
            "val_pairs": str(val_split_path),
            "test_pairs": str(test_split_path),
            "train_sharegpt": str(train_sharegpt),
            "val_sharegpt": str(val_sharegpt),
            "test_sharegpt": str(test_sharegpt),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("SPLIT SUMMARY")
    print("=" * 72)
    for k in (
        "total_image_disease_pairs",
        "n_train_pairs",
        "n_val_pairs",
        "n_test_pairs",
        "n_unique_train_images",
        "n_unique_test_images",
        "n_unique_train_patients",
        "n_unique_test_patients",
    ):
        print(f"{k}: {summary[k]}")
    print("class_distribution_train:", summary["class_distribution_train"])
    print("class_distribution_test:", summary["class_distribution_test"])
    print(
        "train/test leakage free:",
        leakage["train_test_leakage_free"],
        "| val leakage free:",
        leakage.get("val_leakage_free", True),
    )
    print(f"summary saved: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
