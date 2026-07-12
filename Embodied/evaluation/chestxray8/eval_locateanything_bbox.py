#!/usr/bin/env python3
"""
Zero-shot ChestX-ray8 bounding-box evaluation for LocateAnything.

Evaluates disease localization on NIH ChestX-ray BBox_List_2017 annotations
using the original ChestX-ray8 disease set only. Does not train or fine-tune.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# Embodied repo root (…/Embodied)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# ChestX-ray8 constants
# ---------------------------------------------------------------------------

CHESTXRAY8_DISEASES = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
)

# NIH BBox_List_2017.csv uses "Infiltrate" for ChestX-ray8 "Infiltration".
LABEL_ALIASES = {
    "Infiltrate": "Infiltration",
}

# Native ground_multi renders:
#   Locate all the instances that match the following description: {phrase}.
# direct_disease bypasses that wrapper and sends the full query via predict().
PROMPT_STRATEGIES = ("bare_label", "radiology_context", "direct_disease")
RADIOLOGY_CONTEXT_PHRASE_TEMPLATE = "region showing {disease} in the chest radiograph"
DIRECT_DISEASE_QUERY_TEMPLATE = "Locate the {disease} in this chest X-ray"
DEFAULT_MODEL_NAME = "nvidia/LocateAnything-3B"
DEFAULT_DATASET_PATH = "/auto/data2/ykorkmaz/nih-chest-xrays/data/versions/3"
DEFAULT_BBOX_CSV = "BBox_List_2017.csv"
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def build_prompt_phrase(disease: str, prompt_strategy: str) -> str:
    """Semantic phrase / query text for the chosen prompt strategy."""
    if prompt_strategy == "bare_label":
        return disease.lower()
    if prompt_strategy == "radiology_context":
        return RADIOLOGY_CONTEXT_PHRASE_TEMPLATE.format(disease=disease.lower())
    if prompt_strategy == "direct_disease":
        # Keep ChestX-ray8 casing, e.g. "Atelectasis", "Infiltration".
        return DIRECT_DISEASE_QUERY_TEMPLATE.format(disease=disease)
    raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")


def render_ground_multi_query(phrase: str) -> str:
    """Exact user query string constructed inside ground_multi()."""
    return f"Locate all the instances that match the following description: {phrase}."


def build_final_user_query(disease: str, prompt_strategy: str) -> Tuple[str, str]:
    """Return (prompt_phrase, final_rendered_user_query)."""
    phrase = build_prompt_phrase(disease, prompt_strategy)
    if prompt_strategy == "direct_disease":
        return phrase, phrase
    return phrase, render_ground_multi_query(phrase)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval_chestxray8.log"
    logger = logging.getLogger("chestxray8_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def canonicalize_disease(label: str) -> Optional[str]:
    mapped = LABEL_ALIASES.get(label, label)
    if mapped in CHESTXRAY8_DISEASES:
        return mapped
    return None


def build_image_index(
    dataset_path: Path,
    logger: logging.Logger,
    needed_names: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """Map image basename -> absolute path by scanning images_*/ once.

    When ``needed_names`` is set, probe ``images_*/images/<name>`` directly
    (fast on large NIH shards) instead of listing every file.
    """
    needed = set(needed_names) if needed_names is not None else None
    index: Dict[str, Path] = {}
    image_dirs = sorted(dataset_path.glob("images_*"))
    if not image_dirs:
        raise FileNotFoundError(f"No images_* directories under {dataset_path}")

    if needed is not None:
        remaining = set(needed)
        for folder in tqdm(image_dirs, desc="Locating images"):
            if not remaining:
                break
            search_dirs = []
            nested = folder / "images"
            if nested.is_dir():
                search_dirs.append(nested)
            search_dirs.append(folder)
            found_now = []
            for name in remaining:
                for search_dir in search_dirs:
                    candidate = search_dir / name
                    if candidate.is_file():
                        index[name] = candidate.resolve()
                        found_now.append(name)
                        break
            for name in found_now:
                remaining.discard(name)
        if remaining:
            logger.warning(
                "Missing %d requested images (showing up to 10): %s",
                len(remaining),
                sorted(remaining)[:10],
            )
        logger.info("Indexed %d / %d requested images under %s",
                    len(index), len(needed), dataset_path)
        return index

    for folder in tqdm(image_dirs, desc="Indexing images"):
        search_dirs = []
        nested = folder / "images"
        if nested.is_dir():
            search_dirs.append(nested)
        else:
            search_dirs.append(folder)
        for search_dir in search_dirs:
            with os.scandir(search_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not (name.endswith(".png") or name.endswith(".jpg")):
                        continue
                    if name not in index:
                        index[name] = Path(entry.path).resolve()

    logger.info("Indexed %d images under %s", len(index), dataset_path)
    return index


def load_bbox_annotations(
    csv_path: Path,
    allowed_diseases: Sequence[str],
    logger: logging.Logger,
) -> Tuple[
    Dict[Tuple[str, str], List[List[float]]],
    Dict[Tuple[str, str], str],
    Dict[str, int],
]:
    """
    Read BBox_List_2017.csv.

    Columns are positional because the header embeds commas inside
    ``Bbox [x,y,w,h]`` without quoting:
      0: Image Index
      1: Finding Label
      2: x
      3: y
      4: width
      5: height

    Returns:
      groups: (image, canonical_disease) -> list of xyxy boxes
      original_labels: (image, canonical_disease) -> raw CSV finding label
      raw_label_counts: raw CSV label -> count
    """
    allowed = set(allowed_diseases)
    groups: Dict[Tuple[str, str], List[List[float]]] = defaultdict(list)
    original_labels: Dict[Tuple[str, str], str] = {}
    raw_label_counts: Dict[str, int] = defaultdict(int)
    ignored_labels: Dict[str, int] = defaultdict(int)
    n_rows = 0

    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        logger.info("CSV columns (raw parsed): %s", header)
        logger.info(
            "Using positional columns: Image Index, Finding Label, x, y, w, h"
        )

        for row in reader:
            if not row or len(row) < 6:
                continue
            n_rows += 1
            image_name = row[0].strip()
            raw_label = row[1].strip()
            raw_label_counts[raw_label] += 1

            disease = canonicalize_disease(raw_label)
            if disease is None or disease not in allowed:
                ignored_labels[raw_label] += 1
                continue

            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            x1, y1, x2, y2 = x, y, x + w, y + h
            key = (image_name, disease)
            groups[key].append([x1, y1, x2, y2])
            original_labels.setdefault(key, raw_label)

    if ignored_labels:
        logger.warning(
            "Ignoring non-ChestX-ray8 / filtered labels: %s",
            dict(ignored_labels),
        )
    if "Infiltrate" in raw_label_counts:
        logger.warning(
            "Mapped label alias Infiltrate -> Infiltration "
            "(%d annotations) to match ChestX-ray8.",
            raw_label_counts["Infiltrate"],
        )

    logger.info("CSV rows read: %d", n_rows)
    logger.info("Unique images in CSV: %d", len({k[0] for k in groups}))
    logger.info("Image-disease pairs kept: %d", len(groups))
    logger.info("Raw annotations per disease: %s", dict(sorted(raw_label_counts.items())))
    return groups, original_labels, dict(raw_label_counts)


def verify_dataset(csv_path: Path, logger: logging.Logger) -> None:
    """Print dataset verification stats requested by the user."""
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and len(r) >= 6]

    images = {r[0] for r in rows}
    labels: Dict[str, int] = defaultdict(int)
    for r in rows:
        labels[r[1]] += 1

    print("=" * 72)
    print("DATASET VERIFICATION")
    print("=" * 72)
    print(f"CSV path: {csv_path}")
    print(f"CSV column names (raw parsed): {header}")
    print("Effective columns: Image Index, Finding Label, x, y, width, height")
    print(f"Number of rows: {len(rows)}")
    print(f"Number of unique images: {len(images)}")
    print(f"Unique disease labels: {sorted(labels)}")
    print("Number of annotations per disease:")
    for name, count in sorted(labels.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {name}: {count}")

    mapped = {LABEL_ALIASES.get(k, k) for k in labels}
    extra = mapped - set(CHESTXRAY8_DISEASES)
    missing = set(CHESTXRAY8_DISEASES) - mapped
    if "Infiltrate" in labels and "Infiltration" not in labels:
        print(
            "NOTE: CSV uses 'Infiltrate'; mapped to ChestX-ray8 'Infiltration'."
        )
    if extra:
        print(f"WARNING: labels outside ChestX-ray8 (will be ignored): {sorted(extra)}")
    if missing:
        print(f"WARNING: ChestX-ray8 labels missing from CSV: {sorted(missing)}")
    if not extra and not missing:
        print("All ChestX-ray8 diseases are present (after alias mapping).")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Geometry / matching
# ---------------------------------------------------------------------------

def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def greedy_match(
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, List[Tuple[int, int, float]], List[float]]:
    """
    Greedy one-to-one matching on the full IoU matrix.

    Returns:
      iou_matrix: [n_gt, n_pred]
      matches: list of (gt_idx, pred_idx, iou) sorted by match order
      gt_ious: IoU assigned to each GT (0 if unmatched)
    """
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = box_iou(gt, pred)

    gt_ious = [0.0] * n_gt
    matches: List[Tuple[int, int, float]] = []
    if n_gt == 0 or n_pred == 0:
        return iou_matrix, matches, gt_ious

    unmatched_gt = set(range(n_gt))
    unmatched_pred = set(range(n_pred))
    # Work on a copy so we can zero out used pairs.
    work = iou_matrix.copy()
    while unmatched_gt and unmatched_pred:
        best_iou = -1.0
        best_i = -1
        best_j = -1
        for i in unmatched_gt:
            for j in unmatched_pred:
                v = work[i, j]
                if v > best_iou:
                    best_iou = float(v)
                    best_i = i
                    best_j = j
        if best_i < 0 or best_iou <= 0:
            break
        matches.append((best_i, best_j, best_iou))
        gt_ious[best_i] = best_iou
        unmatched_gt.remove(best_i)
        unmatched_pred.remove(best_j)

    return iou_matrix, matches, gt_ious


def compute_pair_metrics(
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    iou_matrix, matches, gt_ious = greedy_match(gt_boxes, pred_boxes)
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    n_matched = len(matches)
    matched_ious = [m[2] for m in matches]

    def recall_at(thresh: float) -> float:
        if n_gt == 0:
            return 0.0
        return sum(1 for v in gt_ious if v >= thresh) / n_gt

    mean_matched = float(np.mean(matched_ious)) if matched_ious else 0.0
    max_iou = float(max(gt_ious)) if gt_ious else 0.0

    return {
        "iou_matrix": iou_matrix.tolist(),
        "matches": matches,
        "gt_ious": gt_ious,
        "matched_ious": matched_ious,
        "mean_matched_iou": mean_matched,
        "maximum_iou": max_iou,
        "recall_0_1": recall_at(0.1),
        "recall_0_3": recall_at(0.3),
        "recall_0_5": recall_at(0.5),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_matched": n_matched,
    }


# ---------------------------------------------------------------------------
# Prediction parsing (LocateAnything: normalized ints in [0, 1000])
# ---------------------------------------------------------------------------

def parse_normalized_boxes(answer: str) -> List[List[float]]:
    """Extract raw [0,1000] xyxy boxes from model text."""
    boxes = []
    for m in BOX_RE.finditer(answer or ""):
        boxes.append([float(g) for g in m.groups()])
    return boxes


def convert_boxes_to_pixels(
    norm_boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> List[List[float]]:
    """Convert [0,1000] xyxy -> pixel xyxy, clamp, drop malformed."""
    out: List[List[float]] = []
    for box in norm_boxes:
        x1, y1, x2, y2 = box
        x1 = max(0.0, min(x1 / 1000.0 * width, float(width)))
        y1 = max(0.0, min(y1 / 1000.0 * height, float(height)))
        x2 = max(0.0, min(x2 / 1000.0 * width, float(width)))
        y2 = max(0.0, min(y2 / 1000.0 * height, float(height)))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([x1, y1, x2, y2])
    return out


# ---------------------------------------------------------------------------
# Resume / IO helpers
# ---------------------------------------------------------------------------

def pair_key(image: str, disease: str) -> str:
    return f"{image}||{disease}"


def load_processed_keys(jsonl_path: Path) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("key") or pair_key(obj["Image"], obj["Disease"])
            done[key] = obj
    return done


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def to_jsonable_box(box: Sequence[float]) -> List[float]:
    return [round(float(v), 4) for v in box]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def summarize_model_output_for_vis(raw_answer: str) -> str:
    """Compact model-output summary for visualization titles."""
    if not raw_answer:
        return "empty"
    if re.search(r"<box>\s*None\s*</box>", raw_answer, flags=re.IGNORECASE):
        return "None"
    n_boxes = len(re.findall(r"<box><\d+><\d+><\d+><\d+></box>", raw_answer))
    if n_boxes:
        return f"{n_boxes} box(es)"
    # Truncate long raw strings for the overlay
    compact = re.sub(r"\s+", " ", raw_answer).strip()
    return compact[:80] + ("..." if len(compact) > 80 else "")


def draw_visualization(
    image: Image.Image,
    disease: str,
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    gt_ious: Sequence[float],
    matches: Sequence[Tuple[int, int, float]],
    out_path: Path,
    raw_answer: str = "",
) -> None:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # GT: green
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
        iou = gt_ious[i] if i < len(gt_ious) else 0.0
        draw.text((x1 + 2, max(0, y1 - 18)), f"GT{i} IoU={iou:.2f}", fill=(0, 200, 0), font=font)

    # Pred: red
    matched_pred = {m[1] for m in matches}
    for j, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = box
        color = (255, 80, 80) if j in matched_pred else (255, 160, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y2 + 2), f"P{j}", fill=color, font=font)

    best_iou = float(max(gt_ious)) if gt_ious else 0.0
    title_lines = [
        f"Disease: {disease}",
        f"Ground-truth boxes: {len(gt_boxes)}",
        f"Predicted boxes: {len(pred_boxes)}",
        f"Model output: {summarize_model_output_for_vis(raw_answer)}",
        f"Best IoU: {best_iou:.2f}",
    ]
    y = 8
    for line in title_lines:
        draw.text((8, y), line, fill=(255, 255, 0), font=font)
        y += 18
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def write_excel(
    output_xlsx: Path,
    sample_rows: List[dict],
    box_rows: List[dict],
    disease_rows: List[dict],
    overall_row: dict,
) -> None:
    try:
        from openpyxl import Workbook
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to write Excel results. "
            "Install with: pip install openpyxl"
        ) from exc

    wb = Workbook()

    # Sheet 1
    ws1 = wb.active
    ws1.title = "sample_results"
    sample_cols = [
        "Image", "Image Path", "Disease", "Original CSV Label",
        "Prompt Strategy", "Prompt Phrase", "Final Rendered User Query",
        "Prompt", "Image Width", "Image Height",
        "Number of GT Boxes", "Number of Predicted Boxes", "Number of Matched Boxes",
        "Ground Truth Boxes", "Predicted Boxes", "Matched IoUs",
        "Mean Matched IoU", "Maximum IoU", "Recall@0.1", "Recall@0.3", "Recall@0.5",
        "Inference Time", "Raw Model Output", "Status", "Error",
    ]
    ws1.append(sample_cols)
    for row in sample_rows:
        ws1.append([row.get(c) for c in sample_cols])

    # Sheet 2
    ws2 = wb.create_sheet("box_results")
    box_cols = [
        "Image", "Disease", "Original CSV Label",
        "Prompt Strategy", "Prompt Phrase", "Final Rendered User Query",
        "Prompt", "GT Box Index", "GT Box",
        "Matched Prediction Index", "Matched Prediction", "IoU",
        "IoU>=0.1", "IoU>=0.3", "IoU>=0.5", "Raw Model Output",
    ]
    ws2.append(box_cols)
    for row in box_rows:
        ws2.append([row.get(c) for c in box_cols])

    # Sheet 3
    ws3 = wb.create_sheet("disease_summary")
    disease_cols = [
        "Disease", "Prompt Strategy", "Images", "GT Boxes", "Predicted Boxes",
        "No-Prediction Rate", "Mean IoU", "Median IoU",
        "Recall@0.1", "Recall@0.3", "Recall@0.5",
        "Mean Inference Time", "Errors",
    ]
    ws3.append(disease_cols)
    for row in disease_rows:
        ws3.append([row.get(c) for c in disease_cols])

    # Sheet 4
    ws4 = wb.create_sheet("overall_summary")
    overall_cols = [
        "Prompt Strategy", "Total Images", "Total Image-Disease Pairs", "Total GT Boxes",
        "Total Predicted Boxes", "No-Prediction Rate", "Mean IoU", "Median IoU",
        "Recall@0.1", "Recall@0.3", "Recall@0.5", "Mean Inference Time",
        "Total Errors", "Model Name", "Prompt Template", "Dataset Path",
        "Evaluation Timestamp",
    ]
    ws4.append(overall_cols)
    ws4.append([overall_row.get(c) for c in overall_cols])

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_xlsx)


def summarize_results(
    sample_rows: List[dict],
    box_rows: List[dict],
    model_name: str,
    dataset_path: str,
    prompt_strategy: str = "",
) -> Tuple[List[dict], dict]:
    # Per-disease
    by_disease: Dict[str, List[dict]] = defaultdict(list)
    for row in sample_rows:
        by_disease[row["Disease"]].append(row)

    disease_rows = []
    for disease in CHESTXRAY8_DISEASES:
        rows = by_disease.get(disease, [])
        if not rows:
            continue
        boxes = [b for b in box_rows if b["Disease"] == disease]
        ious = [float(b["IoU"]) for b in boxes]
        ok_rows = [r for r in rows if r["Status"] == "ok"]
        times = [float(r["Inference Time"]) for r in ok_rows if r["Inference Time"] is not None]
        n_gt = sum(int(r["Number of GT Boxes"]) for r in rows)
        n_pred = sum(int(r["Number of Predicted Boxes"] or 0) for r in rows)
        n_err = sum(1 for r in rows if r["Status"] != "ok")
        n_no_pred = sum(1 for r in rows if int(r.get("Number of Predicted Boxes") or 0) == 0)

        def recall_from_boxes(thresh: float) -> float:
            if not boxes:
                return 0.0
            return sum(1 for b in boxes if float(b["IoU"]) >= thresh) / len(boxes)

        disease_rows.append({
            "Disease": disease,
            "Prompt Strategy": prompt_strategy or (rows[0].get("Prompt Strategy") or ""),
            "Images": len({r["Image"] for r in rows}),
            "GT Boxes": n_gt,
            "Predicted Boxes": n_pred,
            "No-Prediction Rate": n_no_pred / len(rows) if rows else 0.0,
            "Mean IoU": float(np.mean(ious)) if ious else 0.0,
            "Median IoU": float(np.median(ious)) if ious else 0.0,
            "Recall@0.1": recall_from_boxes(0.1),
            "Recall@0.3": recall_from_boxes(0.3),
            "Recall@0.5": recall_from_boxes(0.5),
            "Mean Inference Time": float(np.mean(times)) if times else 0.0,
            "Errors": n_err,
        })

    seen = {r["Disease"] for r in disease_rows}
    for disease, rows in by_disease.items():
        if disease in seen:
            continue
        boxes = [b for b in box_rows if b["Disease"] == disease]
        ious = [float(b["IoU"]) for b in boxes]
        ok_rows = [r for r in rows if r["Status"] == "ok"]
        times = [float(r["Inference Time"]) for r in ok_rows if r["Inference Time"] is not None]
        n_no_pred = sum(1 for r in rows if int(r.get("Number of Predicted Boxes") or 0) == 0)

        def recall_from_boxes(thresh: float) -> float:
            if not boxes:
                return 0.0
            return sum(1 for b in boxes if float(b["IoU"]) >= thresh) / len(boxes)

        disease_rows.append({
            "Disease": disease,
            "Prompt Strategy": prompt_strategy or (rows[0].get("Prompt Strategy") or ""),
            "Images": len({r["Image"] for r in rows}),
            "GT Boxes": sum(int(r["Number of GT Boxes"]) for r in rows),
            "Predicted Boxes": sum(int(r["Number of Predicted Boxes"] or 0) for r in rows),
            "No-Prediction Rate": n_no_pred / len(rows) if rows else 0.0,
            "Mean IoU": float(np.mean(ious)) if ious else 0.0,
            "Median IoU": float(np.median(ious)) if ious else 0.0,
            "Recall@0.1": recall_from_boxes(0.1),
            "Recall@0.3": recall_from_boxes(0.3),
            "Recall@0.5": recall_from_boxes(0.5),
            "Mean Inference Time": float(np.mean(times)) if times else 0.0,
            "Errors": sum(1 for r in rows if r["Status"] != "ok"),
        })

    ok_rows = [r for r in sample_rows if r["Status"] == "ok"]
    times = [float(r["Inference Time"]) for r in ok_rows if r["Inference Time"] is not None]
    all_box_ious = [float(b["IoU"]) for b in box_rows]
    n_no_pred = sum(1 for r in sample_rows if int(r.get("Number of Predicted Boxes") or 0) == 0)
    strategy = prompt_strategy or (
        sample_rows[0].get("Prompt Strategy") if sample_rows else ""
    )

    def overall_recall(thresh: float) -> float:
        if not box_rows:
            return 0.0
        return sum(1 for b in box_rows if float(b["IoU"]) >= thresh) / len(box_rows)

    phrase_template = {
        "bare_label": "{disease.lower()}",
        "radiology_context": RADIOLOGY_CONTEXT_PHRASE_TEMPLATE,
        "direct_disease": DIRECT_DISEASE_QUERY_TEMPLATE,
    }.get(strategy, "")

    overall = {
        "Prompt Strategy": strategy,
        "Total Images": len({r["Image"] for r in sample_rows}),
        "Total Image-Disease Pairs": len(sample_rows),
        "Total GT Boxes": sum(int(r["Number of GT Boxes"]) for r in sample_rows),
        "Total Predicted Boxes": sum(int(r["Number of Predicted Boxes"] or 0) for r in sample_rows),
        "No-Prediction Rate": n_no_pred / len(sample_rows) if sample_rows else 0.0,
        "Mean IoU": float(np.mean(all_box_ious)) if all_box_ious else 0.0,
        "Median IoU": float(np.median(all_box_ious)) if all_box_ious else 0.0,
        "Recall@0.1": overall_recall(0.1),
        "Recall@0.3": overall_recall(0.3),
        "Recall@0.5": overall_recall(0.5),
        "Mean Inference Time": float(np.mean(times)) if times else 0.0,
        "Total Errors": sum(1 for r in sample_rows if r["Status"] != "ok"),
        "Model Name": model_name,
        "Prompt Template": phrase_template,
        "Dataset Path": dataset_path,
        "Evaluation Timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return disease_rows, overall


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def init_worker(model_path: str, device: str):
    import torch
    from locateanything_worker import LocateAnythingWorker

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    worker = LocateAnythingWorker(model_path=model_path, device=device)
    return worker


def run_inference(
    worker,
    image: Image.Image,
    phrase: str,
    prompt_strategy: str = "bare_label",
    final_query: Optional[str] = None,
) -> Tuple[str, float]:
    """Run LocateAnything inference for one image.

    bare_label / radiology_context use ground_multi(phrase).
    direct_disease sends the full custom query via predict().
    """
    import torch

    t0 = time.perf_counter()
    with torch.inference_mode():
        if prompt_strategy == "direct_disease":
            query = final_query or phrase
            result = worker.predict(image, query, generation_mode="hybrid", verbose=False)
        else:
            result = worker.ground_multi(image, phrase, generation_mode="hybrid", verbose=False)
    elapsed = time.perf_counter() - t0
    answer = result.get("answer", "")
    if isinstance(answer, tuple):
        answer = answer[0]
    return str(answer), elapsed


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)

    dataset_path = Path(args.dataset_path)
    csv_path = Path(args.bbox_csv) if args.bbox_csv else dataset_path / DEFAULT_BBOX_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(f"BBox CSV not found: {csv_path}")

    verify_dataset(csv_path, logger)

    allowed = list(CHESTXRAY8_DISEASES)
    if args.diseases:
        requested = [d.strip() for d in args.diseases.split(",") if d.strip()]
        unknown = [d for d in requested if d not in CHESTXRAY8_DISEASES]
        if unknown:
            logger.warning("Ignoring unknown --diseases values: %s", unknown)
        allowed = [d for d in requested if d in CHESTXRAY8_DISEASES]
        if not allowed:
            raise ValueError("No valid ChestX-ray8 diseases after filtering --diseases")

    annotations, original_labels, _ = load_bbox_annotations(csv_path, allowed, logger)
    pairs = sorted(annotations.keys(), key=lambda x: (x[0], x[1]))
    if args.limit is not None and args.limit > 0:
        pairs = pairs[: args.limit]
        logger.info("Limiting evaluation to first %d image-disease pairs", len(pairs))

    prompt_strategy = args.prompt_strategy
    if prompt_strategy not in PROMPT_STRATEGIES:
        raise ValueError(
            f"--prompt-strategy must be one of {PROMPT_STRATEGIES}, got {prompt_strategy!r}"
        )
    logger.info("Prompt strategy: %s", prompt_strategy)
    logger.info(
        "Example query for Atelectasis: %s",
        build_final_user_query("Atelectasis", prompt_strategy)[1],
    )

    if args.verify_only:
        logger.info("verify-only complete (%d pairs would be evaluated).", len(pairs))
        return

    needed_images = {image_name for image_name, _ in pairs}
    image_index = build_image_index(dataset_path, logger, needed_names=needed_images)

    jsonl_path = output_dir / "intermediate_results.jsonl"
    processed = load_processed_keys(jsonl_path) if args.resume else {}
    if args.resume:
        logger.info("Resume enabled: %d pairs already in %s", len(processed), jsonl_path)
    else:
        if jsonl_path.exists():
            bak = jsonl_path.with_suffix(jsonl_path.suffix + f".bak_{int(time.time())}")
            jsonl_path.rename(bak)
            logger.info("Starting fresh; moved previous JSONL to %s", bak)
        processed = {}

    # Device selection (respect CUDA_VISIBLE_DEVICES)
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        logger.info(
            "Using CUDA device (visible count=%d). CUDA_VISIBLE_DEVICES=%s",
            torch.cuda.device_count(),
            os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        )
        if torch.cuda.device_count() > 0:
            logger.info("GPU 0 (visible): %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.warning("CUDA not available; running on CPU")

    logger.info("Loading model once: %s", args.model_path)
    worker = init_worker(args.model_path, device)

    sample_rows: List[dict] = []
    box_rows: List[dict] = []
    # Reload completed rows for final Excel when resuming
    if args.resume and processed:
        for obj in processed.values():
            if "sample_row" in obj:
                sample_rows.append(obj["sample_row"])
            if "box_rows" in obj:
                box_rows.extend(obj["box_rows"])

    since_save = 0
    visualized = 0
    verbose_validation = bool(args.verbose_validation)

    pbar = tqdm(pairs, desc="Evaluating")
    for image_name, disease in pbar:
        key = pair_key(image_name, disease)
        pbar.set_postfix(image=image_name[:18], disease=disease[:12])

        if key in processed:
            continue

        phrase, final_query = build_final_user_query(disease, prompt_strategy)
        original_csv_label = original_labels.get((image_name, disease), disease)
        gt_boxes = annotations[(image_name, disease)]
        image_path = image_index.get(image_name)

        sample_row: Dict[str, Any] = {
            "Image": image_name,
            "Image Path": str(image_path) if image_path else "",
            "Disease": disease,
            "Original CSV Label": original_csv_label,
            "Prompt Strategy": prompt_strategy,
            "Prompt Phrase": phrase,
            "Final Rendered User Query": final_query,
            "Prompt": final_query,
            "Image Width": None,
            "Image Height": None,
            "Number of GT Boxes": len(gt_boxes),
            "Number of Predicted Boxes": 0,
            "Number of Matched Boxes": 0,
            "Ground Truth Boxes": json.dumps([to_jsonable_box(b) for b in gt_boxes]),
            "Predicted Boxes": "[]",
            "Matched IoUs": "[]",
            "Mean Matched IoU": 0.0,
            "Maximum IoU": 0.0,
            "Recall@0.1": 0.0,
            "Recall@0.3": 0.0,
            "Recall@0.5": 0.0,
            "Inference Time": None,
            "Raw Model Output": "",
            "Status": "ok",
            "Error": "",
        }
        pair_box_rows: List[dict] = []

        try:
            if image_path is None or not Path(image_path).is_file():
                raise FileNotFoundError(f"Image not found in index: {image_name}")

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                raise RuntimeError(f"Corrupted/unreadable image: {image_path}: {exc}") from exc

            width, height = image.size
            sample_row["Image Width"] = width
            sample_row["Image Height"] = height

            raw_answer, infer_time = run_inference(
                worker,
                image,
                phrase,
                prompt_strategy=prompt_strategy,
                final_query=final_query,
            )
            sample_row["Inference Time"] = round(infer_time, 4)
            sample_row["Raw Model Output"] = raw_answer

            norm_boxes = parse_normalized_boxes(raw_answer)
            pred_boxes = convert_boxes_to_pixels(norm_boxes, width, height)
            metrics = compute_pair_metrics(gt_boxes, pred_boxes)

            sample_row["Number of Predicted Boxes"] = metrics["n_pred"]
            sample_row["Number of Matched Boxes"] = metrics["n_matched"]
            sample_row["Predicted Boxes"] = json.dumps(
                [to_jsonable_box(b) for b in pred_boxes]
            )
            sample_row["Matched IoUs"] = json.dumps(
                [round(float(v), 6) for v in metrics["matched_ious"]]
            )
            sample_row["Mean Matched IoU"] = round(metrics["mean_matched_iou"], 6)
            sample_row["Maximum IoU"] = round(metrics["maximum_iou"], 6)
            sample_row["Recall@0.1"] = round(metrics["recall_0_1"], 6)
            sample_row["Recall@0.3"] = round(metrics["recall_0_3"], 6)
            sample_row["Recall@0.5"] = round(metrics["recall_0_5"], 6)

            # Map gt -> matched pred
            gt_to_pred = {m[0]: (m[1], m[2]) for m in metrics["matches"]}
            for gi, gt in enumerate(gt_boxes):
                pred_idx, iou = gt_to_pred.get(gi, (None, metrics["gt_ious"][gi]))
                pred_box = pred_boxes[pred_idx] if pred_idx is not None else None
                pair_box_rows.append({
                    "Image": image_name,
                    "Disease": disease,
                    "Original CSV Label": original_csv_label,
                    "Prompt Strategy": prompt_strategy,
                    "Prompt Phrase": phrase,
                    "Final Rendered User Query": final_query,
                    "Prompt": final_query,
                    "GT Box Index": gi,
                    "GT Box": json.dumps(to_jsonable_box(gt)),
                    "Matched Prediction Index": pred_idx if pred_idx is not None else "",
                    "Matched Prediction": json.dumps(to_jsonable_box(pred_box)) if pred_box else "",
                    "IoU": round(float(iou), 6),
                    "IoU>=0.1": int(float(iou) >= 0.1),
                    "IoU>=0.3": int(float(iou) >= 0.3),
                    "IoU>=0.5": int(float(iou) >= 0.5),
                    "Raw Model Output": raw_answer,
                })

            if verbose_validation:
                print("\n" + "=" * 72)
                print("VALIDATION SAMPLE")
                print("=" * 72)
                print(f"image name: {image_name}")
                print(f"disease: {disease}")
                print(f"prompt strategy: {prompt_strategy}")
                print(f"prompt phrase: {phrase}")
                print(f"final rendered user query: {final_query}")
                print(f"image size: {width} x {height}")
                print(f"raw LocateAnything output:\n{raw_answer}")
                print(f"parsed predicted boxes ([0,1000]): {norm_boxes}")
                print(f"converted predicted boxes (pixels): {pred_boxes}")
                print(f"ground-truth boxes (pixels): {gt_boxes}")
                print(f"IoU matrix:\n{np.array(metrics['iou_matrix'])}")
                print(f"final matching (gt_idx, pred_idx, iou): {metrics['matches']}")
                print(
                    "final metrics: "
                    f"mean_matched_iou={metrics['mean_matched_iou']:.4f}, "
                    f"max_iou={metrics['maximum_iou']:.4f}, "
                    f"R@0.1={metrics['recall_0_1']:.4f}, "
                    f"R@0.3={metrics['recall_0_3']:.4f}, "
                    f"R@0.5={metrics['recall_0_5']:.4f}"
                )
                print("=" * 72 + "\n")

            if visualized < args.visualize_count:
                safe_disease = disease.replace(" ", "_")
                out_vis = vis_dir / f"{Path(image_name).stem}__{safe_disease}__{prompt_strategy}.png"
                draw_visualization(
                    image,
                    f"{disease} [{prompt_strategy}]",
                    gt_boxes,
                    pred_boxes,
                    metrics["gt_ious"],
                    metrics["matches"],
                    out_vis,
                    raw_answer=raw_answer,
                )
                visualized += 1

        except Exception as exc:
            sample_row["Status"] = "error"
            sample_row["Error"] = str(exc)
            logger.error("Failed %s / %s: %s", image_name, disease, exc)
            logger.debug(traceback.format_exc())
            # Still emit GT box rows with IoU=0 on failure
            for gi, gt in enumerate(gt_boxes):
                pair_box_rows.append({
                    "Image": image_name,
                    "Disease": disease,
                    "Original CSV Label": original_csv_label,
                    "Prompt Strategy": prompt_strategy,
                    "Prompt Phrase": phrase,
                    "Final Rendered User Query": final_query,
                    "Prompt": final_query,
                    "GT Box Index": gi,
                    "GT Box": json.dumps(to_jsonable_box(gt)),
                    "Matched Prediction Index": "",
                    "Matched Prediction": "",
                    "IoU": 0.0,
                    "IoU>=0.1": 0,
                    "IoU>=0.3": 0,
                    "IoU>=0.5": 0,
                    "Raw Model Output": "",
                })

        sample_rows.append(sample_row)
        box_rows.extend(pair_box_rows)

        record = {
            "key": key,
            "Image": image_name,
            "Disease": disease,
            "sample_row": sample_row,
            "box_rows": pair_box_rows,
        }
        append_jsonl(jsonl_path, record)
        processed[key] = record
        since_save += 1

        if since_save >= args.save_every:
            disease_summary, overall = summarize_results(
                sample_rows, box_rows, args.model_path, str(dataset_path), prompt_strategy
            )
            xlsx_path = output_dir / "chestxray8_locateanything_bbox_results.xlsx"
            write_excel(xlsx_path, sample_rows, box_rows, disease_summary, overall)
            logger.info("Periodic Excel save -> %s", xlsx_path)
            since_save = 0

    disease_summary, overall = summarize_results(
        sample_rows, box_rows, args.model_path, str(dataset_path), prompt_strategy
    )
    xlsx_path = output_dir / "chestxray8_locateanything_bbox_results.xlsx"
    write_excel(xlsx_path, sample_rows, box_rows, disease_summary, overall)
    logger.info("Wrote final Excel: %s", xlsx_path)
    logger.info(
        "Overall [%s]: pairs=%s GT=%s Pred=%s no_pred_rate=%.3f MeanIoU=%.4f R@0.5=%.4f errors=%s",
        prompt_strategy,
        overall["Total Image-Disease Pairs"],
        overall["Total GT Boxes"],
        overall["Total Predicted Boxes"],
        overall["No-Prediction Rate"],
        overall["Mean IoU"],
        overall["Recall@0.5"],
        overall["Total Errors"],
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Zero-shot ChestX-ray8 bbox evaluation with LocateAnything"
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="Root containing BBox_List_2017.csv and images_* folders",
    )
    p.add_argument(
        "--bbox-csv",
        type=str,
        default="",
        help="Optional explicit path to BBox_List_2017.csv",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HF model id or local checkpoint path",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results"),
        help="Directory for Excel, JSONL, logs, visualizations",
    )
    p.add_argument("--limit", type=int, default=None, help="Max image-disease pairs")
    p.add_argument("--resume", action="store_true", help="Skip pairs already in JSONL")
    p.add_argument("--save-every", type=int, default=25, help="Periodic Excel flush interval")
    p.add_argument("--visualize-count", type=int, default=20, help="Number of vis images")
    p.add_argument(
        "--diseases",
        type=str,
        default="",
        help="Comma-separated ChestX-ray8 subset (default: all 8)",
    )
    p.add_argument(
        "--prompt-strategy",
        type=str,
        default="bare_label",
        choices=list(PROMPT_STRATEGIES),
        help=(
            "bare_label / radiology_context (ground_multi wrapper) or "
            "direct_disease: 'Locate the {Disease} in this chest X-ray'"
        ),
    )
    p.add_argument(
        "--verbose-validation",
        action="store_true",
        help="Print detailed per-sample parser/matching diagnostics",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the dataset CSV and exit (no model load)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
