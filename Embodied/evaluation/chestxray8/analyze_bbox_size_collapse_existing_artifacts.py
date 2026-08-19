#!/usr/bin/env python3
"""Offline-only bounding-box size-collapse analysis from existing artifacts."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import statistics
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib.colors import Color, HexColor
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
EVAL_RESULTS = HERE / "results/chestxray8_hybrid_grpo_native"
MODEL_RESULTS = REPO / "Embodied/results/chestxray8_hybrid_grpo_native"
OUT = (
    EVAL_RESULTS
    / "analysis/bbox_size_collapse_existing_artifacts_20260816"
)
MANIFEST = HERE / "splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl"
MANIFEST_SHA256 = "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838"
MIN_CORRELATION_N = 3


def validation_specs() -> list[dict[str, Any]]:
    pf = EVAL_RESULTS / "validation/g8_kl0_loraonly_projectorfrozen_100_seed42"
    cb = EVAL_RESULTS / "validation/g8_caseb_150_optimization710_seed42"
    ntp = MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_100_FUNCTIONAL_KV_CKPT"
    lr1 = MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_50_FUNCTIONAL_KV_CKPT"
    lr5 = MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR5E6_50_FUNCTIONAL_KV_CKPT"
    ctl = MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_500_CONTROLLED_FUNCTIONAL_KV_CKPT"
    specs: list[dict[str, Any]] = []

    def add(series: str, step: int, path: Path, *, decoding: str, lr: str,
            projector: str, temperature: str, top_p: str) -> None:
        specs.append({
            "experiment": series, "step": step, "path": path, "G": 8,
            "decoding": decoding, "temperature": temperature, "top_p": top_p,
            "learning_rate": lr, "kl_beta": 0.0, "projector": projector,
        })

    for step in (0, 25, 50, 75, 100):
        add("G8 Hybrid projector-frozen", step,
            pf / f"step_{step:03d}/step_{step:03d}_per_sample.jsonl",
            decoding="hybrid", lr="2e-5", projector="frozen",
            temperature="1.0", top_p="1.0")
    for step in (0, 25, 50, 75):
        name = "frozen_base" if step == 0 else f"step_{step:03d}"
        add("G8 Hybrid Case-B projector-trainable", step,
            cb / f"all_checkpoints_internal_validation/{name}_per_sample.jsonl",
            decoding="hybrid", lr="2e-5", projector="trainable",
            temperature="1.0", top_p="1.0")
    for step in (100, 125, 150):
        add("G8 Hybrid Case-B projector-trainable", step,
            cb / f"step_{step:03d}_isolated/step_{step:03d}_per_sample.jsonl",
            decoding="hybrid", lr="2e-5", projector="trainable",
            temperature="1.0", top_p="1.0")
    for step in (0, 25, 50, 75, 100):
        name = "frozen_base" if step == 0 else f"step_{step:03d}"
        add("G8 NTP high-exploration LR2e-5", step,
            ntp / f"internal_validation80_step000_025_050_075_100/{name}_per_sample.jsonl",
            decoding="ntp_only", lr="2e-5", projector="frozen",
            temperature="1.2", top_p="0.9")
    for root, label, lr in (
        (lr1, "G8 NTP high-exploration LR1e-5 short", "1e-5"),
        (lr5, "G8 NTP high-exploration LR5e-6 short", "5e-6"),
    ):
        for step in (25, 50):
            add(label, step, root / f"internal_validation80_step025_050/step_{step:03d}_per_sample.jsonl",
                decoding="ntp_only", lr=lr, projector="frozen",
                temperature="1.2", top_p="0.9")
    add("G8 NTP controlled LR1e-5", 0,
        ctl / "internal_validation80_step000_200_300_400_500/frozen_base_per_sample.jsonl",
        decoding="ntp_only", lr="1e-5", projector="frozen",
        temperature="1.2", top_p="0.9")
    add("G8 NTP controlled LR1e-5", 100,
        ctl / "internal_validation_step100/step_100_per_sample.jsonl",
        decoding="ntp_only", lr="1e-5", projector="frozen",
        temperature="1.2", top_p="0.9")
    for step in (200, 300, 400, 500):
        add("G8 NTP controlled LR1e-5", step,
            ctl / f"internal_validation80_step000_200_300_400_500/step_{step:03d}_per_sample.jsonl",
            decoding="ntp_only", lr="1e-5", projector="frozen",
            temperature="1.2", top_p="0.9")
    return specs


def training_specs() -> list[dict[str, Any]]:
    tr = EVAL_RESULTS / "training"
    kl_tr = HERE / "results/chestxray8_hybrid_grpo_native_medground_kl/training"
    ctl = MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_500_CONTROLLED_FUNCTIONAL_KV_CKPT"
    return [
        {"experiment": "G4 Hybrid KL0 early100", "paths": [
            tr / "two_gpu_18x18_production_100_seed42/two_gpu_g4_grpo_multistep_metrics.jsonl"],
         "G": 4, "decoding": "hybrid", "temperature": "1.0", "top_p": "1.0",
         "learning_rate": "2e-5", "kl_beta": 0.0, "projector": "trainable"},
        {"experiment": "G4 Hybrid KL0 full500", "paths": [
            tr / "two_gpu_18x18_production_500_train80_seed42_revalidated_20260808/two_gpu_g4_grpo_multistep_metrics.jsonl"],
         "G": 4, "decoding": "hybrid", "temperature": "1.0", "top_p": "1.0",
         "learning_rate": "2e-5", "kl_beta": 0.0, "projector": "trainable"},
        {"experiment": "G4 Hybrid KL.04 full500", "paths": [
            kl_tr / "two_gpu_18x18_medground_kl_500_train80_seed42_20260808_v2/two_gpu_g4_grpo_multistep_metrics.jsonl"],
         "G": 4, "decoding": "hybrid", "temperature": "1.0", "top_p": "1.0",
         "learning_rate": "2e-5", "kl_beta": 0.04, "projector": "trainable"},
        {"experiment": "G8 Hybrid Case-B", "paths": [
            tr / "two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_metrics.jsonl"],
         "G": 8, "decoding": "hybrid", "temperature": "1.0", "top_p": "1.0",
         "learning_rate": "2e-5", "kl_beta": 0.0, "projector": "trainable"},
        {"experiment": "G8 Hybrid projector-frozen", "paths": [
            tr / "g8_kl0_loraonly_projectorfrozen_100_optimization710_seed42/two_gpu_g8_loraonly_projectorfrozen_metrics.jsonl"],
         "G": 8, "decoding": "hybrid", "temperature": "1.0", "top_p": "1.0",
         "learning_rate": "2e-5", "kl_beta": 0.0, "projector": "frozen"},
        {"experiment": "G8 NTP T1/top_p1 projector-frozen", "paths": [
            tr / "g8_kl0_loraonly_projectorfrozen_ntponly_100_optimization710_seed42/two_gpu_g8_loraonly_projectorfrozen_ntponly_metrics.jsonl"],
         "G": 8, "decoding": "ntp_only", "temperature": "1.0", "top_p": "1.0",
         "learning_rate": "2e-5", "kl_beta": 0.0, "projector": "frozen"},
        {"experiment": "G8 NTP high-exploration LR2e-5", "paths": [
            MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_100_FUNCTIONAL_KV_CKPT/two_gpu_g8_ntponly_t1p2_topp0p9_metrics.jsonl"],
         "G": 8, "decoding": "ntp_only", "temperature": "1.2", "top_p": "0.9",
         "learning_rate": "2e-5", "kl_beta": 0.0, "projector": "frozen"},
        {"experiment": "G8 NTP high-exploration LR1e-5 short", "paths": [
            MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_50_FUNCTIONAL_KV_CKPT/two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_metrics.jsonl"],
         "G": 8, "decoding": "ntp_only", "temperature": "1.2", "top_p": "0.9",
         "learning_rate": "1e-5", "kl_beta": 0.0, "projector": "frozen"},
        {"experiment": "G8 NTP high-exploration LR5e-6 short", "paths": [
            MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR5E6_50_FUNCTIONAL_KV_CKPT/two_gpu_g8_ntponly_t1p2_topp0p9_lr5e6_metrics.jsonl"],
         "G": 8, "decoding": "ntp_only", "temperature": "1.2", "top_p": "0.9",
         "learning_rate": "5e-6", "kl_beta": 0.0, "projector": "frozen"},
        {"experiment": "G8 NTP controlled LR1e-5", "paths": [
            ctl / "segment_000_100/two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_metrics.jsonl",
            ctl / "segment_100_200/two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_metrics.jsonl",
            ctl / "segment_200_500/two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_metrics.jsonl"],
         "G": 8, "decoding": "ntp_only", "temperature": "1.2", "top_p": "0.9",
         "learning_rate": "1e-5", "kl_beta": 0.0, "projector": "frozen"},
        {"experiment": "G8 NTP partial KL.04 LR1e-6", "paths": [
            MODEL_RESULTS / "G8_MEDGROUND_KL004_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E6_3000_FUNCTIONAL_KV_CKPT/two_gpu_g8_ntponly_t1p2_topp0p9_lr1e6_kl004_3000_metrics.jsonl"],
         "G": 8, "decoding": "ntp_only", "temperature": "1.2", "top_p": "0.9",
         "learning_rate": "1e-6", "kl_beta": 0.04, "projector": "frozen"},
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def rate(values: Iterable[bool]) -> float | None:
    values = list(values)
    return statistics.fmean(float(value) for value in values) if values else None


def val_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if bool(row.get("valid_native_box"))]
    areas = [float(row["box_area_norm01"]) for row in valid]
    near = [
        (float(row["committed_final_bbox_norm_1000"][2]) -
         float(row["committed_final_bbox_norm_1000"][0])) / 1000 >= 0.9
        and
        (float(row["committed_final_bbox_norm_1000"][3]) -
         float(row["committed_final_bbox_norm_1000"][1])) / 1000 >= 0.9
        for row in valid
    ]
    rewards = [row.get("reward") or {} for row in rows]
    return {
        "n_all_samples": len(rows),
        "n_valid_native_boxes": len(valid),
        "valid_box_rate_all_samples": len(valid) / len(rows) if rows else None,
        "mean_area_valid_boxes": mean(areas),
        "median_area_valid_boxes": median(areas),
        "near_full_rate_valid_boxes": rate(near),
        "tiny_area_lt_0_001_rate_valid_boxes": rate(area < 0.001 for area in areas),
        "tiny_area_lt_0_01_rate_valid_boxes": rate(area < 0.01 for area in areas),
        "near_full_rate_all_samples": sum(near) / len(rows) if rows else None,
        "tiny_area_lt_0_001_rate_all_samples": sum(area < 0.001 for area in areas) / len(rows) if rows else None,
        "tiny_area_lt_0_01_rate_all_samples": sum(area < 0.01 for area in areas) / len(rows) if rows else None,
        "mean_iou_all_samples": mean(float(row.get("iou", 0.0)) for row in rows),
        "median_iou_all_samples": median(float(row.get("iou", 0.0)) for row in rows),
        "iou_gt_0_5_rate_all_samples": rate(float(row.get("iou", 0.0)) > 0.5 for row in rows),
        "mean_semantic_reward_all_samples": mean(float(r.get("semantic_reward", 0.0)) for r in rewards),
        "mean_format_reward_all_samples": mean(float(r.get("format_reward", 0.0)) for r in rewards),
        "mean_total_reward_all_samples": mean(float(r.get("total_reward", 0.0)) for r in rewards),
    }


def load_validation() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overall, disease, inventory = [], [], []
    expected_identity: set[tuple[int, str, int]] | None = None
    for spec in validation_specs():
        path = spec["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        identities = {(int(r["sample_index"]), str(r["image_index"]), int(r["seed"])) for r in rows}
        if len(rows) != 80 or len(identities) != 80:
            raise RuntimeError(f"{path}: expected 80 unique rows, got {len(rows)}/{len(identities)}")
        if any(not str(r.get("disease", "")).strip() for r in rows):
            raise RuntimeError(f"{path}: missing disease labels")
        if expected_identity is None:
            expected_identity = identities
        elif identities != expected_identity:
            raise RuntimeError(f"{path}: val80 sample/seed identity mismatch")
        base = {key: value for key, value in spec.items() if key != "path"}
        overall.append({
            **base, "source_path": str(path.resolve()),
            "split_name": "internal_validation80",
            "split_manifest_path": str(MANIFEST.resolve()),
            "manifest_sha256": MANIFEST_SHA256, **val_metrics(rows),
        })
        by_disease: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_disease[str(row["disease"])].append(row)
        for label, subset in sorted(by_disease.items()):
            disease.append({
                **base, "disease": label, "source_path": str(path.resolve()),
                "split_name": "internal_validation80",
                "split_manifest_path": str(MANIFEST.resolve()),
                "manifest_sha256": MANIFEST_SHA256, **val_metrics(subset),
            })
        inventory.append({
            "artifact_type": "validation_checkpoint", **base,
            "status": "included", "scope": "internal_validation80",
            "rows": len(rows), "unique_rows": len(identities),
            "source_path": str(path.resolve()),
            "note": "80 aligned rows; disease labels present",
        })
    return overall, disease, inventory


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area1 = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area2 = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area1 + area2 - inter) if area1 + area2 - inter > 0 else 0.0


def group_record(record: dict[str, Any]) -> dict[str, Any]:
    rollouts = []
    for trajectory in record.get("trajectories", []):
        reward = trajectory.get("reward") or {}
        box = reward.get("final_box_norm_1000") or trajectory.get("committed_final_bbox_norm_1000")
        valid = bool(reward.get("format_valid") and reward.get("geometry_valid")
                     and reward.get("has_unambiguous_committed_box") and box is not None)
        area = None
        near = False
        if valid:
            box = [float(x) for x in box]
            width = max(0.0, (box[2] - box[0]) / 1000)
            height = max(0.0, (box[3] - box[1]) / 1000)
            area, near = width * height, width >= 0.9 and height >= 0.9
        rollouts.append({
            "area": area, "near": near, "valid": valid, "box": box if valid else None,
            "iou": float(reward.get("final_iou", 0.0)),
            "semantic": float(reward.get("semantic_reward", 0.0)),
            "format": float(reward.get("format_reward", 0.0)),
            "total": float(reward.get("total_reward", 0.0)),
        })
    areas = [r["area"] for r in rollouts if r["area"] is not None]
    boxes = [r["box"] for r in rollouts if r["box"] is not None]
    pairwise = [bbox_iou(a, b) for a, b in combinations(boxes, 2)]
    coord_std = None
    if len(boxes) >= 2:
        coord_std = mean(statistics.pstdev(box[i] for box in boxes) for i in range(4))
    ious = [r["iou"] for r in rollouts]
    loss = record.get("loss_components") or {}
    kl = record.get("effective_kl_config") or {}
    before = int(record.get("optimizer_step_count_before_group", max(0, int(record.get("global_step", 1)) - 1)))
    after = int(record.get("optimizer_step_count_after_group", record.get("global_step", before)))
    return {
        "before": before, "after": after, "skipped": bool(record.get("optimizer_step_skipped", False)),
        "rollouts": rollouts, "n_rollouts": len(rollouts), "n_valid": len(areas),
        "mean_area": mean(areas), "median_area": median(areas),
        "near_valid": rate(r["near"] for r in rollouts if r["valid"]),
        "near_all": rate(r["near"] for r in rollouts),
        "tiny001_valid": rate(a < 0.001 for a in areas),
        "tiny01_valid": rate(a < 0.01 for a in areas),
        "tiny001_all": sum(a < 0.001 for a in areas) / len(rollouts) if rollouts else None,
        "tiny01_all": sum(a < 0.01 for a in areas) / len(rollouts) if rollouts else None,
        "valid_rate": len(areas) / len(rollouts) if rollouts else None,
        "mixed": float(0 < sum(i > 0.5 for i in ious) < len(ious)) if ious else None,
        "p_iou": rate(i > 0.5 for i in ious), "max_iou": max(ious) if ious else None,
        "mean_iou": mean(ious), "semantic": mean(r["semantic"] for r in rollouts),
        "format": mean(r["format"] for r in rollouts), "total": mean(r["total"] for r in rollouts),
        "pairwise_iou": mean(pairwise), "coordinate_std": coord_std,
        "kl_value": float(loss.get("mean_kl_value", 0.0)) if loss else None,
        "kl_contribution": float(loss.get("mean_kl_loss_contribution", 0.0)) if loss else None,
        "kl_beta_observed": float(loss.get("beta", kl.get("beta", 0.0))) if (loss or kl) else None,
    }


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for pos in order[index:end]:
            result[pos] = rank
        index = end
    return result


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < MIN_CORRELATION_N or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def correlation_row(experiment: str, level: str, x_name: str, y_name: str,
                    pairs: list[tuple[float, float]]) -> dict[str, Any]:
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "experiment": experiment, "level": level, "x_metric": x_name,
        "y_metric": y_name, "n": len(pairs), "minimum_n": MIN_CORRELATION_N,
        "pearson_r": pearson(xs, ys),
        "spearman_rho": pearson(ranks(xs), ranks(ys)) if len(pairs) >= MIN_CORRELATION_N else None,
        "interpretation": "association only; no causal inference",
    }


def training_analysis() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    windows, correlations, inventory = [], [], []
    for spec in training_specs():
        records: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        duplicate_count = 0
        for path in spec["paths"]:
            if not path.is_file():
                raise FileNotFoundError(path)
            for record in read_jsonl(path):
                key = (
                    record.get("optimizer_step_count_before_group"), record.get("attempted_group_count"),
                    record.get("sample_index"), record.get("attempt_seed"),
                )
                if key in seen:
                    duplicate_count += 1
                    continue
                seen.add(key)
                records.append(record)
        groups = [group_record(record) for record in records if record.get("trajectories")]
        max_step = max((g["after"] for g in groups), default=0)
        inventory.append({
            "artifact_type": "training_metrics", "experiment": spec["experiment"],
            "step": "", "G": spec["G"], "decoding": spec["decoding"],
            "temperature": spec["temperature"], "top_p": spec["top_p"],
            "learning_rate": spec["learning_rate"], "kl_beta": spec["kl_beta"],
            "projector": spec["projector"], "status": "included",
            "scope": f"optimizer steps 1-{max_step}", "rows": len(records),
            "unique_rows": len(seen), "source_path": ";".join(str(p.resolve()) for p in spec["paths"]),
            "note": f"{duplicate_count} exact resumed identities deduplicated",
        })
        for end in range(25, max_step + 1, 25):
            start = end - 25
            # A skipped attempt has after==before and is assigned by its pre-step counter.
            subset = [g for g in groups if start <= g["before"] < end]
            rollouts = [r for g in subset for r in g["rollouts"]]
            valid_areas = [r["area"] for r in rollouts if r["area"] is not None]
            valid_boxes = [r for r in rollouts if r["valid"]]
            row = {
                "experiment": spec["experiment"], "G": spec["G"],
                "decoding": spec["decoding"], "temperature": spec["temperature"],
                "top_p": spec["top_p"], "learning_rate": spec["learning_rate"],
                "kl_beta_configured": spec["kl_beta"], "projector": spec["projector"],
                "optimizer_step_start_exclusive": start,
                "optimizer_step_end_inclusive": end,
                "attempted_groups": len(subset), "skipped_optimizer_groups": sum(g["skipped"] for g in subset),
                "rollouts": len(rollouts), "valid_boxes": len(valid_areas),
                "valid_box_rate_all_rollouts": len(valid_areas) / len(rollouts) if rollouts else None,
                "mean_bbox_area_valid_boxes": mean(valid_areas),
                "median_bbox_area_valid_boxes": median(valid_areas),
                "near_full_rate_valid_boxes": rate(r["near"] for r in valid_boxes),
                "near_full_rate_all_rollouts": rate(r["near"] for r in rollouts),
                "tiny_lt_0_001_rate_valid_boxes": rate(a < 0.001 for a in valid_areas),
                "tiny_lt_0_01_rate_valid_boxes": rate(a < 0.01 for a in valid_areas),
                "tiny_lt_0_001_rate_all_rollouts": sum(a < 0.001 for a in valid_areas) / len(rollouts) if rollouts else None,
                "tiny_lt_0_01_rate_all_rollouts": sum(a < 0.01 for a in valid_areas) / len(rollouts) if rollouts else None,
                "spatial_mixed_group_rate": mean(g["mixed"] for g in subset if g["mixed"] is not None),
                "rollout_p_iou_gt_0_5": rate(r["iou"] > 0.5 for r in rollouts),
                "mean_group_max_raw_iou": mean(g["max_iou"] for g in subset if g["max_iou"] is not None),
                "mean_raw_iou_all_rollouts": mean(r["iou"] for r in rollouts),
                "mean_semantic_reward": mean(r["semantic"] for r in rollouts),
                "mean_format_reward": mean(r["format"] for r in rollouts),
                "mean_total_reward": mean(r["total"] for r in rollouts),
                "mean_kl_value": mean(g["kl_value"] for g in subset if g["kl_value"] is not None),
                "mean_kl_contribution": mean(g["kl_contribution"] for g in subset if g["kl_contribution"] is not None),
                "mean_kl_beta_observed": mean(g["kl_beta_observed"] for g in subset if g["kl_beta_observed"] is not None),
                "mean_pairwise_bbox_iou_within_group": mean(g["pairwise_iou"] for g in subset if g["pairwise_iou"] is not None),
                "mean_coordinate_std_norm_1000": mean(g["coordinate_std"] for g in subset if g["coordinate_std"] is not None),
            }
            windows.append(row)
        for level, observations in (
            ("group", groups),
            ("rollout", [r for g in groups for r in g["rollouts"]]),
        ):
            fields = (
                ("area", "mean_area" if level == "group" else "area"),
                ("semantic", "semantic"), ("iou", "mean_iou" if level == "group" else "iou"),
            )
            for x, y in (("area", "semantic"), ("area", "iou"), ("semantic", "iou")):
                x_key = dict(fields)[x]
                y_key = dict(fields)[y]
                pairs = [(float(o[x_key]), float(o[y_key])) for o in observations
                         if o.get(x_key) is not None and o.get(y_key) is not None]
                correlations.append(correlation_row(spec["experiment"], level, x, y, pairs))
        own_windows = [row for row in windows if row["experiment"] == spec["experiment"]]
        for x in ("mean_bbox_area_valid_boxes", "near_full_rate_all_rollouts",
                  "tiny_lt_0_001_rate_all_rollouts", "tiny_lt_0_01_rate_all_rollouts"):
            for y in ("mean_semantic_reward", "mean_raw_iou_all_rollouts",
                      "rollout_p_iou_gt_0_5", "spatial_mixed_group_rate"):
                pairs = [(float(row[x]), float(row[y])) for row in own_windows
                         if row.get(x) is not None and row.get(y) is not None]
                correlations.append(correlation_row(spec["experiment"], "25_step_window", x, y, pairs))
    return windows, correlations, inventory


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = OUT / name
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


PALETTE = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2")


def line_plot(stem: str, title: str, series: list[tuple[str, list[tuple[float, float]]]],
              y_label: str, *, log_y: bool = False, x_label: str = "Optimizer step") -> None:
    width, height = 1400, 820
    left, right, top, bottom = 140, 60, 100, 150
    points = [(x, y) for _, values in series for x, y in values if y is not None and (y > 0 or not log_y)]
    if not points:
        return
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    transformed = [(x, math.log10(y) if log_y else y) for x, y in points]
    ymin, ymax = min(y for _, y in transformed), max(y for _, y in transformed)
    if ymin == ymax:
        ymin, ymax = ymin - 0.5, ymax + 0.5

    def xy(x: float, y: float) -> tuple[float, float]:
        yy = math.log10(y) if log_y else y
        px = left + (x - xmin) / max(1e-12, xmax - xmin) * (width - left - right)
        py = height - bottom - (yy - ymin) / (ymax - ymin) * (height - top - bottom)
        return px, py

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,sans-serif;fill:#222}</style>',
        f'<text x="{width/2}" y="55" text-anchor="middle" font-size="34" font-weight="bold">{html.escape(title)}</text>',
    ]
    for i in range(6):
        y = top + i * (height - top - bottom) / 5
        value = ymax - i * (ymax - ymin) / 5
        label = f"{10 ** value:.3g}" if log_y else f"{value:.3g}"
        svg += [
            f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#ddd"/>',
            f'<text x="{left-15}" y="{y+7}" text-anchor="end" font-size="19">{label}</text>',
        ]
    svg += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222" stroke-width="2"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222" stroke-width="2"/>',
        f'<text x="{width/2}" y="{height-65}" text-anchor="middle" font-size="23">{html.escape(x_label)}</text>',
        f'<text x="38" y="{height/2}" text-anchor="middle" font-size="23" transform="rotate(-90 38 {height/2})">{html.escape(y_label)}</text>',
    ]
    for idx, (label, values) in enumerate(series):
        color = PALETTE[idx % len(PALETTE)]
        coords = [xy(x, y) for x, y in values if y is not None and (y > 0 or not log_y)]
        if not coords:
            continue
        svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in coords)}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, y in coords:
            svg.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>')
        ly = 88 + idx * 29
        svg += [f'<line x1="{width-520}" y1="{ly}" x2="{width-475}" y2="{ly}" stroke="{color}" stroke-width="5"/>',
                f'<text x="{width-465}" y="{ly+7}" font-size="18">{html.escape(label)}</text>']
    svg.append("</svg>")
    svg_path = OUT / f"{stem}.svg"
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    subprocess.run(["/usr/bin/convert", "-density", "180", str(svg_path), str(OUT / f"{stem}.png")],
                   check=True, capture_output=True, text=True)

    pdf = canvas.Canvas(str(OUT / f"{stem}.pdf"), pagesize=(700, 410))
    pdf.setTitle(title)
    scale = 0.5
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawCentredString(350, 382, title)
    for i in range(6):
        sy = (height - (top + i * (height - top - bottom) / 5)) * scale
        value = ymax - i * (ymax - ymin) / 5
        label = f"{10 ** value:.3g}" if log_y else f"{value:.3g}"
        pdf.setStrokeColor(HexColor("#dddddd")); pdf.line(left*scale, sy, (width-right)*scale, sy)
        pdf.setFillColor(HexColor("#222222")); pdf.setFont("Helvetica", 9)
        pdf.drawRightString((left-15)*scale, sy-3, label)
    pdf.setStrokeColor(HexColor("#222222"))
    pdf.line(left*scale, bottom*scale, left*scale, (height-top)*scale)
    pdf.line(left*scale, bottom*scale, (width-right)*scale, bottom*scale)
    for idx, (_, values) in enumerate(series):
        color = HexColor(PALETTE[idx % len(PALETTE)])
        coords = [(x*scale, (height-y)*scale) for x, y in
                  (xy(x, y) for x, y in values if y is not None and (y > 0 or not log_y))]
        if not coords:
            continue
        path = pdf.beginPath(); path.moveTo(*coords[0])
        for point in coords[1:]:
            path.lineTo(*point)
        pdf.setStrokeColor(color); pdf.setLineWidth(2); pdf.drawPath(path)
        pdf.setFillColor(color)
        for x, y in coords:
            pdf.circle(x, y, 3, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#222222")); pdf.setFont("Helvetica", 11)
    pdf.drawCentredString(350, 28, x_label)
    pdf.saveState(); pdf.translate(18, 205); pdf.rotate(90); pdf.drawCentredString(0, 0, y_label); pdf.restoreState()
    pdf.save()


def disease_small_multiples(rows: list[dict[str, Any]]) -> None:
    regimes = ("G8 Hybrid projector-frozen", "G8 Hybrid Case-B projector-trainable",
               "G8 NTP controlled LR1e-5")
    diseases = (
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    )
    width, height = 1500, 1780
    margin_x, margin_y, gap_x, gap_y = 110, 125, 80, 76
    panel_w = (width - 2 * margin_x - gap_x) / 2
    panel_h = (height - 2 * margin_y - 3 * gap_y - 70) / 4
    selected = [row for row in rows if row["experiment"] in regimes and row["disease"] in diseases
                and row["median_area_valid_boxes"] is not None and row["median_area_valid_boxes"] > 0]
    log_values = [math.log10(float(row["median_area_valid_boxes"])) for row in selected]
    ymin, ymax = min(log_values), max(log_values)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:DejaVu Sans,sans-serif;fill:#222}</style>',
           f'<text x="{width/2}" y="54" text-anchor="middle" font-size="34" font-weight="bold">Disease-wise area trajectories (key regimes)</text>']
    panel_geometry: dict[str, tuple[float, float]] = {}
    for index, disease in enumerate(diseases):
        col, row_index = index % 2, index // 2
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_y + row_index * (panel_h + gap_y)
        panel_geometry[disease] = (x0, y0)
        svg += [f'<text x="{x0+panel_w/2}" y="{y0-18}" text-anchor="middle" font-size="24" font-weight="bold">{disease}</text>',
                f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#333" stroke-width="2"/>']
        for tick in range(5):
            y = y0 + tick * panel_h / 4
            value = ymax - tick * (ymax-ymin) / 4
            svg += [f'<line x1="{x0}" y1="{y}" x2="{x0+panel_w}" y2="{y}" stroke="#ddd"/>',
                    f'<text x="{x0-9}" y="{y+6}" text-anchor="end" font-size="15">{10**value:.2g}</text>']
        subset = [item for item in selected if item["disease"] == disease]
        xmax = max(float(item["step"]) for item in subset)
        for ridx, regime in enumerate(regimes):
            values = sorted((float(item["step"]), float(item["median_area_valid_boxes"]))
                            for item in subset if item["experiment"] == regime)
            if not values:
                continue
            coords = [(x0 + step/max(1.0, xmax)*panel_w,
                       y0 + (ymax-math.log10(value))/max(1e-12, ymax-ymin)*panel_h)
                      for step, value in values]
            color = PALETTE[ridx]
            svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in coords)}" fill="none" stroke="{color}" stroke-width="4"/>')
            for x, y in coords:
                svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>')
        svg += [f'<text x="{x0+panel_w/2}" y="{y0+panel_h+34}" text-anchor="middle" font-size="18">Optimizer step</text>']
    for index, regime in enumerate(regimes):
        x = 250 + index * 430
        svg += [f'<line x1="{x}" y1="{height-35}" x2="{x+50}" y2="{height-35}" stroke="{PALETTE[index]}" stroke-width="5"/>',
                f'<text x="{x+60}" y="{height-28}" font-size="17">{html.escape(regime)}</text>']
    svg.append("</svg>")
    svg_path = OUT / "disease_wise_area_vs_step.svg"
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    subprocess.run(["/usr/bin/convert", "-density", "180", str(svg_path),
                    str(OUT / "disease_wise_area_vs_step.png")],
                   check=True, capture_output=True, text=True)
    pdf = canvas.Canvas(str(OUT / "disease_wise_area_vs_step.pdf"), pagesize=(750, 890))
    pdf.setTitle("Disease-wise area trajectories")
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawCentredString(375, 864, "Disease-wise area trajectories (key regimes)")
    for disease, (x0, y0_svg) in panel_geometry.items():
        x0 *= .5
        y0 = (height - y0_svg - panel_h) * .5
        pdf.setStrokeColor(HexColor("#333333")); pdf.rect(x0, y0, panel_w*.5, panel_h*.5)
        pdf.setFillColor(HexColor("#222222")); pdf.setFont("Helvetica-Bold", 11)
        pdf.drawCentredString(x0+panel_w*.25, y0+panel_h*.5+8, disease)
        subset = [item for item in selected if item["disease"] == disease]
        xmax = max(float(item["step"]) for item in subset)
        for ridx, regime in enumerate(regimes):
            values = sorted((float(item["step"]), float(item["median_area_valid_boxes"]))
                            for item in subset if item["experiment"] == regime)
            coords = [(x0 + step/max(1.0, xmax)*panel_w*.5,
                       y0 + (math.log10(value)-ymin)/max(1e-12, ymax-ymin)*panel_h*.5)
                      for step, value in values]
            if not coords:
                continue
            path = pdf.beginPath(); path.moveTo(*coords[0])
            for point in coords[1:]:
                path.lineTo(*point)
            pdf.setStrokeColor(HexColor(PALETTE[ridx])); pdf.setLineWidth(2); pdf.drawPath(path)
    pdf.save()


def heatmap_plot(rows: list[dict[str, Any]]) -> None:
    selected = [r for r in rows if r["experiment"] in {
        "G8 Hybrid projector-frozen", "G8 NTP high-exploration LR2e-5",
        "G8 NTP controlled LR1e-5"}]
    labels = sorted({f"{r['experiment']} s{r['step']}" for r in selected})
    label_display = {
        label: (
            label.replace("G8 Hybrid projector-frozen", "PF")
            .replace("G8 NTP high-exploration LR2e-5", "NTP2")
            .replace("G8 NTP controlled LR1e-5", "CTL")
        )
        for label in labels
    }
    diseases = sorted({r["disease"] for r in selected})
    lookup = {(f"{r['experiment']} s{r['step']}", r["disease"]): r["median_area_valid_boxes"] for r in selected}
    width, height = 2100, 260 + 60 * len(diseases)
    left, top, cell_w, cell_h = 250, 150, (width - 300) / len(labels), 55
    values = [v for v in lookup.values() if v is not None and v > 0]
    lo, hi = min(math.log10(v) for v in values), max(math.log10(v) for v in values)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:DejaVu Sans,sans-serif;fill:#222}</style>',
           f'<text x="{width/2}" y="45" text-anchor="middle" font-size="34" font-weight="bold">Disease × checkpoint median valid-box area</text>',
           f'<text x="{width/2}" y="76" text-anchor="middle" font-size="18">Hybrid-PF, NTP-2e-5, and controlled NTP-1e-5 trajectories · log-scaled color</text>']
    for j, label in enumerate(labels):
        x = left + (j + .5) * cell_w
        svg.append(f'<text x="{x}" y="{top-12}" text-anchor="middle" font-size="16">{html.escape(label_display[label])}</text>')
    for i, disease in enumerate(diseases):
        y = top + i * cell_h
        svg.append(f'<text x="{left-12}" y="{y+34}" text-anchor="end" font-size="18">{html.escape(disease)}</text>')
        for j, label in enumerate(labels):
            value = lookup.get((label, disease))
            if value is None or value <= 0:
                color, text = "#eeeeee", "NA"
            else:
                t = (math.log10(value) - lo) / max(1e-12, hi-lo)
                color = f"rgb({int(250-170*t)},{int(245-120*t)},{int(235-25*t)})"
                text = f"{value:.3g}"
            x = left + j * cell_w
            svg += [f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="white"/>',
                    f'<text x="{x+cell_w/2}" y="{y+34}" text-anchor="middle" font-size="13">{text}</text>']
    svg.append("</svg>")
    svg_path = OUT / "disease_checkpoint_median_area_heatmap.svg"
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    subprocess.run(["/usr/bin/convert", "-density", "160", str(svg_path),
                    str(OUT / "disease_checkpoint_median_area_heatmap.png")],
                   check=True, capture_output=True, text=True)
    pdf = canvas.Canvas(str(OUT / "disease_checkpoint_median_area_heatmap.pdf"),
                        pagesize=(1050, height / 2))
    pdf.setFont("Helvetica-Bold", 17)
    pdf.drawCentredString(525, height/2-28, "Disease x checkpoint median valid-box area")
    pdf.setFont("Helvetica", 7)
    for j, label in enumerate(labels):
        x = left / 2 + (j + .5) * cell_w / 2
        y = height / 2 - top / 2 + 4
        pdf.drawCentredString(x, y, label_display[label])
    for i, disease in enumerate(diseases):
        y = height/2 - top/2 - (i+1)*cell_h/2
        pdf.setFillColor(HexColor("#222222")); pdf.setFont("Helvetica", 8)
        pdf.drawRightString((left-12)/2, y+8, disease)
        for j, label in enumerate(labels):
            value = lookup.get((label, disease))
            if value is None or value <= 0:
                color, text = HexColor("#eeeeee"), "NA"
            else:
                t = (math.log10(value)-lo)/max(1e-12,hi-lo)
                color = Color((250-170*t)/255, (245-120*t)/255, (235-25*t)/255)
                text = f"{value:.2g}"
            x = left/2 + j*cell_w/2
            pdf.setFillColor(color); pdf.rect(x, y, cell_w/2, cell_h/2, fill=1, stroke=0)
            pdf.setFillColor(HexColor("#222222")); pdf.drawCentredString(x+cell_w/4, y+8, text)
    pdf.save()


def plots(overall: list[dict[str, Any]], disease: list[dict[str, Any]],
          windows: list[dict[str, Any]]) -> None:
    def series(metric: str, rows: list[dict[str, Any]] = overall,
               keep: set[str] | None = None) -> list[tuple[str, list[tuple[float, float]]]]:
        by: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            if keep is None or row["experiment"] in keep:
                value = row.get(metric)
                if value is not None:
                    by[row["experiment"]].append((float(row["step"]), float(value)))
        return [(label, sorted(values)) for label, values in by.items()]
    line_plot("overall_bbox_area_vs_step", "Overall bbox area vs step",
              series("median_area_valid_boxes"), "Median area among valid boxes", log_y=True)
    line_plot("near_full_rate_vs_step", "Near-full boxes vs step",
              series("near_full_rate_valid_boxes"), "Conditional rate (valid boxes)")
    line_plot("tiny_box_rates_vs_step", "Tiny-box collapse vs step",
              series("tiny_area_lt_0_001_rate_valid_boxes", keep={
                  "G8 NTP high-exploration LR2e-5", "G8 NTP controlled LR1e-5",
                  "G8 NTP high-exploration LR1e-5 short", "G8 NTP high-exploration LR5e-6 short"}),
              "P(area < 0.001 | valid)")
    key = {"G8 Hybrid projector-frozen", "G8 Hybrid Case-B projector-trainable",
           "G8 NTP controlled LR1e-5"}
    disease_small_multiples(disease)
    heatmap_plot(disease)
    reward_series = []
    by: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in windows:
        if row["mean_bbox_area_valid_boxes"] is not None and row["mean_semantic_reward"] is not None:
            by[row["experiment"]].append((float(row["mean_bbox_area_valid_boxes"]),
                                          float(row["mean_semantic_reward"])))
    # x-axis is area although the generic renderer says optimizer step.
    reward_series = [(name, values) for name, values in by.items()]
    line_plot("reward_size_relationship", "Training-window reward–size relationship",
              reward_series, "Mean semantic reward", x_label="Mean bbox area among valid rollouts")


def regime_comparison(
    overall: list[dict[str, Any]], windows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    near = next(r for r in overall if r["experiment"] == "G8 Hybrid projector-frozen" and r["step"] == 100)
    stable = next(
        r
        for r in overall
        if r["experiment"] == "G8 Hybrid Case-B projector-trainable" and r["step"] == 150
    )
    tiny = next(r for r in overall if r["experiment"] == "G8 NTP controlled LR1e-5" and r["step"] == 500)
    rows = []
    for regime, source, training_name in (
        ("near-full", near, "G8 Hybrid projector-frozen"),
        ("stable val80-selected", stable, "G8 Hybrid Case-B"),
        ("tiny", tiny, "G8 NTP controlled LR1e-5"),
    ):
        matching_window = next(
            (
                row
                for row in windows
                if row["experiment"] == training_name
                and row["optimizer_step_end_inclusive"] == source["step"]
            ),
            None,
        )
        rows.append({
            "regime": regime, "selection_basis": (
                "fixed requested checkpoint"
                if regime != "stable val80-selected"
                else "existing Case-B internal-validation80 selection; no heldout selection"
            ),
            **{key: source[key] for key in (
                "experiment", "step", "G", "decoding", "temperature", "top_p",
                "learning_rate", "kl_beta", "projector", "n_all_samples",
                "n_valid_native_boxes", "valid_box_rate_all_samples",
                "mean_area_valid_boxes", "median_area_valid_boxes",
                "near_full_rate_valid_boxes", "tiny_area_lt_0_001_rate_valid_boxes",
                "tiny_area_lt_0_01_rate_valid_boxes", "mean_iou_all_samples",
                "median_iou_all_samples", "iou_gt_0_5_rate_all_samples",
                "mean_semantic_reward_all_samples", "mean_format_reward_all_samples",
                "mean_total_reward_all_samples", "source_path")},
            "spatial_mixed_group_rate_last_25_training_steps": (
                matching_window["spatial_mixed_group_rate"] if matching_window else None
            ),
        })
    return rows


def report_text(regimes: list[dict[str, Any]], overall: list[dict[str, Any]],
                windows: list[dict[str, Any]],
                correlations: list[dict[str, Any]]) -> str:
    near, stable, tiny = regimes
    controlled = [r for r in overall if r["experiment"] == "G8 NTP controlled LR1e-5"]
    strict_best = max(overall, key=lambda row: float(row["mean_iou_all_samples"]))

    def corr(experiment: str, level: str, x_metric: str, y_metric: str) -> dict[str, Any]:
        return next(
            row
            for row in correlations
            if row["experiment"] == experiment
            and row["level"] == level
            and row["x_metric"] == x_metric
            and row["y_metric"] == y_metric
        )

    controlled_area_sem = corr(
        "G8 NTP controlled LR1e-5",
        "25_step_window",
        "mean_bbox_area_valid_boxes",
        "mean_semantic_reward",
    )
    controlled_tiny_sem = corr(
        "G8 NTP controlled LR1e-5",
        "25_step_window",
        "tiny_lt_0_001_rate_all_rollouts",
        "mean_semantic_reward",
    )
    near_sem = corr(
        "G8 Hybrid projector-frozen",
        "25_step_window",
        "near_full_rate_all_rollouts",
        "mean_semantic_reward",
    )
    return f"""# Offline bounding-box size-collapse analysis

## Scope and denominator definitions

This analysis is read-only and uses existing internal-validation80 per-sample outputs and existing training metrics. No inference, training, checkpoint evaluation, or training-code modification was performed. Heldout194 results are inventory context only and are not used for selection or primary metrics.

- **Valid-box conditional** area, near-full, and tiny rates use only rows with `valid_native_box=true`. Near-full means width >= 0.9 and height >= 0.9 in normalized image coordinates; tiny means area < 0.001 or area < 0.01.
- **All-sample/all-rollout** rates divide by all 80 validation rows or all logged rollouts, treating invalid boxes as neither near-full nor tiny.
- IoU and reward means divide by all rows/rollouts, including invalid outputs with their recorded values.
- Training windows are non-overlapping `(start, end]` blocks of 25 optimizer steps. Attempts are assigned by `optimizer_step_count_before_group`; skipped optimizer groups are counted once in the next applicable boundary and exact resumed identities are deduplicated.
- Correlations require N >= {MIN_CORRELATION_N}; Pearson and tie-aware Spearman are associations only, never causal estimates.

## Validation checks

All {len(overall)} included checkpoints contain exactly 80 unique `(sample_index, image_index, seed)` rows, aligned identities, non-empty disease labels, and the pinned manifest SHA `{MANIFEST_SHA256}`. The duplicate high-exploration step-100 resume was excluded in favor of the byte-equivalent primary lineage output.

## Key findings

- Near-full reference: {near['experiment']} step {near['step']} has median valid-box area {near['median_area_valid_boxes']:.6g}, conditional near-full rate {near['near_full_rate_valid_boxes']:.1%}, and mean IoU {near['mean_iou_all_samples']:.4f}.
- Stable reference: {stable['experiment']} step {stable['step']} was the existing Case-B val80 selection; mean IoU is {stable['mean_iou_all_samples']:.4f}, conditional near-full rate is {stable['near_full_rate_valid_boxes']:.1%}, and neither tiny threshold is populated.
- Strict maximum existing val80 mean IoU is {strict_best['experiment']} step {strict_best['step']} at {strict_best['mean_iou_all_samples']:.4f}; it is not labeled stable because its conditional near-full rate is {strict_best['near_full_rate_valid_boxes']:.1%}.
- Tiny reference: {tiny['experiment']} step {tiny['step']} has median valid-box area {tiny['median_area_valid_boxes']:.6g}, conditional area<0.001 rate {tiny['tiny_area_lt_0_001_rate_valid_boxes']:.1%}, and mean IoU {tiny['mean_iou_all_samples']:.4f}.
- Controlled LR1e-5 median valid-box area by checkpoint: {", ".join(f"s{r['step']}={r['median_area_valid_boxes']:.4g}" for r in controlled)}.

## Training-log evidence

The CSVs provide per-group, per-rollout, and 25-step-window diagnostics. Legacy G4 logs, including the KL beta .04 run, do not contain the newer explicit spatial-density payload, so the same quantities are reconstructed from stored trajectories and reward components. The partial G8 KL.04 LR1e-6 log is usable through its existing final optimizer step only.

### Tests of the proposed relationships

- **A — spatial starvation with semantic/format dominance:** supported as an association, not a causal result. In G4 KL-off full500, spatial mixed-group rate is zero through most windows while valid/format reward approaches one and near-full rollout rates exceed 90%. In the projector-frozen Hybrid run, last-window trajectories move toward near-full boxes while localization-positive groups remain scarce; across its four windows, near-full rate versus semantic reward has Pearson r={near_sem['pearson_r']:.3f} and Spearman rho={near_sem['spearman_rho']:.3f} (N={near_sem['n']}). The very small window count prevents strong inference.
- **B — tiny collapse while semantic reward rises and IoU degrades:** supported within the controlled LR1e-5 lineage. From val80 step200 to step300, mean semantic reward rises from 0.01875 to 0.04180 while mean IoU falls from 0.1548 to 0.0524 and median valid-box area falls from 0.1568 to 0.00057. Across 20 training windows, tiny<0.001 rate versus semantic reward has Pearson r={controlled_tiny_sem['pearson_r']:.3f}, Spearman rho={controlled_tiny_sem['spearman_rho']:.3f}.
- **C — area shrinks as semantic reward rises:** strongly associated only in the controlled lineage, not universally. Controlled-window area versus semantic reward has Pearson r={controlled_area_sem['pearson_r']:.3f}, Spearman rho={controlled_area_sem['spearman_rho']:.3f} (N={controlled_area_sem['n']}); other configurations have different signs. Step/time is a major confounder.

## Most likely explanation for near-full vs tiny-box collapse

The two failures are distinct endpoint policies under a sparse thresholded spatial reward. Near-full collapse is consistent with a degenerate high-coverage strategy: boxes expand until formatting remains valid while localization specificity is lost. Tiny collapse in the long controlled NTP lineage appears later and is consistent with a low-area coordinate-token attractor that remains format-valid but almost never earns IoU>0.5. Existing artifacts support an interaction among sparse binary spatial credit, semantic/format reward availability, exploration distribution, and optimization duration/LR. They do **not** isolate one causal factor: projector status, decoding mode, temperature/top-p, LR, duration, and lineage are not jointly randomized.

## Coverage gaps and limitations

There are no pinned-val80 outputs for G4 Hybrid KL0, G4 Hybrid KL.04, or G8 NTP T1/top_p1 projector-frozen. Older G4 models trained on the full train80, so evaluating them on this pinned validation80 would not constitute scientifically unseen validation. The G4 KL beta .04 training log is included, but it does not provide a matched held-out internal-validation trajectory. Disease strata are small and no confidence intervals or causal claims are warranted.

## Minimal ablation plan to distinguish the causes

1. **KL on versus off, matched:** G8, frozen projector, NTP-only, T=1.2/top_p=.9, LR1e-5, same seed/rewards. Hypothesis: policy drift permits endpoint collapse. Expected signature: KL suppresses area drift while preserving or improving raw-IoU density. Compute: one new KL-on branch can reuse the existing KL-off lineage. Reward design unchanged.
2. **LR1e-5 versus LR1e-6, matched:** same KL setting, G, sampling, seed, and rewards. Hypothesis: tiny collapse is optimization-rate instability rather than cumulative duration alone. Expected signature: LR1e-6 delays/prevents the area break near step300 at comparable updates. Compute: one matched branch if an existing lineage is reused. Reward design unchanged.
3. **Semantic weight 0 diagnostic:** same setup versus preserved MedGround-R1 reward. Hypothesis: semantic reward supplies the dominant advantage during spatial starvation and selects size attractors. Expected signature: the area-semantic trajectory and collapse timing change without necessarily improving raw-IoU density. Compute: one short diagnostic branch. This changes reward design and is not a production proposal.
4. **Binary spatial reward versus continuous IoU:** future diagnostic only. Hypothesis: thresholding erases localization ranking. Expected signature: denser within-group localization advantages and improved raw IoU before size collapse. Compute: one short matched branch. This changes reward design.
5. **Warm-start SFT versus no SFT:** future two-branch comparison. Hypothesis: an SFT localization prior keeps coordinate generation in a useful basin. Expected signature: higher initial valid-IoU density and slower area drift under otherwise matched GRPO. Compute: SFT preparation plus matched GRPO. Reward design unchanged, but initialization changes.

Preserve MedGround-R1 semantics in the primary path. Semantic-weight0 and continuous IoU are explicitly diagnostic/future reward designs, not retrospective reinterpretations of the existing reward.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(MANIFEST) != MANIFEST_SHA256:
        raise RuntimeError("Pinned manifest SHA256 mismatch")
    overall, disease, inventory_val = load_validation()
    windows, correlations, inventory_train = training_analysis()
    regimes = regime_comparison(overall, windows)
    gaps = [
        {"requested_artifact": "G4 Hybrid KL0 pinned val80", "status": "missing",
         "impact": "no primary validation metric", "note": "Older G4 trained on full train80; pinned val80 is not scientifically unseen."},
        {"requested_artifact": "G4 Hybrid KL.04 pinned val80", "status": "missing",
         "impact": "no primary validation metric", "note": "No existing per-sample internal-validation80 output."},
        {"requested_artifact": "G8 NTP T1/top_p1 projector-frozen pinned val80", "status": "missing",
         "impact": "training-only coverage", "note": "No existing per-sample internal-validation80 output."},
        {"requested_artifact": "G4 Hybrid KL beta .04 full500 metrics", "status": "available",
         "impact": "training-window analysis included", "note": "Legacy density metrics reconstructed from stored trajectories."},
        {"requested_artifact": "G8 partial KL.04 LR1e-6", "status": "partial",
         "impact": "windows only through existing step", "note": "Usable existing records; run is incomplete relative to planned 3000."},
    ]
    ablations = [
        {"rank": 1, "ablation": "KL on vs off (matched)", "hypothesis": "unconstrained policy drift permits endpoint collapse",
         "expected_signature": "KL suppresses bbox-area drift without reducing raw-IoU density", "compute": "1 new KL-on branch; reuse existing KL-off lineage",
         "reward_design_change": "none"},
        {"rank": 2, "ablation": "LR 1e-5 vs 1e-6 (matched)", "hypothesis": "tiny collapse is optimization-rate instability",
         "expected_signature": "LR1e-6 delays or prevents the area break near step300", "compute": "1 matched branch if an existing lineage is reused",
         "reward_design_change": "none"},
        {"rank": 3, "ablation": "semantic weight 0 diagnostic", "hypothesis": "semantic reward dominates advantages during spatial starvation",
         "expected_signature": "area-semantic trajectory and collapse timing change without guaranteed IoU gain", "compute": "1 matched short diagnostic branch",
         "reward_design_change": "yes; diagnostic only, not production"},
        {"rank": 4, "ablation": "binary spatial vs continuous IoU", "hypothesis": "thresholding erases localization ranking",
         "expected_signature": "denser localization advantages and improving raw IoU before area drift", "compute": "1 matched short future branch",
         "reward_design_change": "yes; future diagnostic"},
        {"rank": 5, "ablation": "warm-start SFT vs no SFT", "hypothesis": "SFT keeps coordinate generation in a useful basin",
         "expected_signature": "higher initial valid-IoU density and slower bbox-area drift", "compute": "SFT preparation plus 2 matched GRPO branches",
         "reward_design_change": "none; initialization changes"},
    ]
    heldout_inventory = [
        {"artifact_type": "heldout_comparison_inventory", "experiment": "G8 NTP high-exploration base/step100",
         "step": "0;100", "status": "inventory_only", "scope": "heldout194",
         "source_path": str((MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_100_FUNCTIONAL_KV_CKPT/heldout_test194_base_step100_confirmatory").resolve()),
         "note": "already completed; never used for selection or primary metrics"},
        {"artifact_type": "heldout_comparison_inventory", "experiment": "G8 NTP controlled frozen-base/step200",
         "step": "0;200", "status": "inventory_only", "scope": "heldout194",
         "source_path": str((MODEL_RESULTS / "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_500_CONTROLLED_FUNCTIONAL_KV_CKPT/heldout_test194_frozen_base_selected_step200_final_confirmatory").resolve()),
         "note": "already completed; never used for selection or primary metrics"},
        {"artifact_type": "heldout_comparison_inventory", "experiment": "G8 Hybrid/NTP projector-frozen step100",
         "step": "0;100", "status": "inventory_only", "scope": "heldout194",
         "source_path": str((EVAL_RESULTS / "test194/projector_frozen_hybrid_vs_ntponly_seed42").resolve()),
         "note": "already completed; never used for selection or primary metrics"},
    ]
    inventory = inventory_val + inventory_train + heldout_inventory
    source_paths = [MANIFEST] + [s["path"] for s in validation_specs()]
    for spec in training_specs():
        source_paths.extend(spec["paths"])
    unique_paths = list(dict.fromkeys(path.resolve() for path in source_paths))
    sources = [{
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "role": "pinned_manifest" if path == MANIFEST.resolve() else (
            "validation_per_sample" if "per_sample" in path.name else "training_metrics"),
    } for path in unique_paths]
    write_csv("experiment_inventory.csv", inventory)
    write_csv("validation_overall.csv", overall)
    write_csv("validation_disease.csv", disease)
    write_csv("training_windows.csv", windows)
    write_csv("training_correlations.csv", correlations)
    write_csv("collapse_regime_comparison.csv", regimes)
    write_csv("coverage_gaps.csv", gaps)
    write_csv("ablation_plan.csv", ablations)
    write_csv("source_artifacts.csv", sources)
    plots(overall, disease, windows)
    report = report_text(regimes, overall, windows, correlations)
    (OUT / "bbox_size_collapse_report.md").write_text(report, encoding="utf-8")
    metadata = {
        "analysis": "offline-only bounding-box size collapse from existing artifacts",
        "created_date": "2026-08-16", "new_inference_run": False,
        "training_run": False, "checkpoint_evaluation_run": False,
        "training_code_modified": False, "heldout_used_for_selection": False,
        "heldout_used_for_primary_metrics": False,
        "manifest": str(MANIFEST.resolve()), "manifest_sha256": MANIFEST_SHA256,
        "definitions": {
            "valid_native_box": "source valid_native_box=true",
            "area": "((x2-x1)/1000)*((y2-y1)/1000) among valid native boxes",
            "near_full": "width>=0.9 and height>=0.9",
            "tiny_0_001": "area<0.001", "tiny_0_01": "area<0.01",
            "conditional_denominator": "valid native boxes only",
            "compatibility_denominator": "all samples/rollouts; invalid boxes count in denominator but not numerator",
            "iou_reward_denominator": "all samples/rollouts using recorded values",
            "training_window": "non-overlapping 25 optimizer-step interval (start,end], assigned by pre-step counter",
            "correlations": f"Pearson and tie-aware Spearman; minimum N={MIN_CORRELATION_N}; no causality",
        },
        "deduplication": {
            "validation": "Excluded duplicate resumed high-exploration step100; primary lineage used.",
            "training": "Exact (before-step, attempted-group, sample, attempt-seed) identities deduplicated across segments.",
            "skipped_groups": "Skipped attempts retained once and assigned by optimizer_step_count_before_group.",
        },
        "validation_checks": {
            "checkpoint_count": len(overall), "rows_per_checkpoint": 80,
            "all_sample_identities_aligned": True, "all_disease_labels_present": True,
            "manifest_hash_verified": True,
        },
        "sources": sources, "coverage_gaps": gaps,
        "outputs": sorted(str((OUT / name).resolve()) for name in (
            "experiment_inventory.csv", "validation_overall.csv", "validation_disease.csv",
            "training_windows.csv", "training_correlations.csv", "collapse_regime_comparison.csv",
            "coverage_gaps.csv", "ablation_plan.csv", "source_artifacts.csv",
            "overall_bbox_area_vs_step.png", "overall_bbox_area_vs_step.pdf",
            "overall_bbox_area_vs_step.svg",
            "disease_wise_area_vs_step.png", "disease_wise_area_vs_step.pdf",
            "disease_wise_area_vs_step.svg",
            "near_full_rate_vs_step.png", "near_full_rate_vs_step.pdf",
            "near_full_rate_vs_step.svg",
            "tiny_box_rates_vs_step.png", "tiny_box_rates_vs_step.pdf",
            "tiny_box_rates_vs_step.svg",
            "disease_checkpoint_median_area_heatmap.png",
            "disease_checkpoint_median_area_heatmap.pdf",
            "disease_checkpoint_median_area_heatmap.svg",
            "reward_size_relationship.png", "reward_size_relationship.pdf",
            "reward_size_relationship.svg",
            "bbox_size_collapse_report.md", "analysis_metadata.json", "supervisor_update.txt",
        )),
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    stable = regimes[1]
    near, tiny = regimes[0], regimes[2]
    update = (
        f"Offline-only analysis completed across {len(overall)} aligned val80 checkpoints and "
        f"{len(windows)} non-overlapping training windows. Near-full reference: projector-frozen "
        f"Hybrid step100 conditional near-full={near['near_full_rate_valid_boxes']:.1%}; stable "
        f"val80 reference={stable['experiment']} step{stable['step']} mean IoU={stable['mean_iou_all_samples']:.4f}; "
        f"tiny reference: controlled LR1e-5 step500 median area={tiny['median_area_valid_boxes']:.6g}, "
        f"P(area<0.001|valid)={tiny['tiny_area_lt_0_001_rate_valid_boxes']:.1%}. "
        "No heldout selection, inference, training, or checkpoint evaluation was performed."
    )
    (OUT / "supervisor_update.txt").write_text(update + "\n", encoding="utf-8")
    print(update)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
