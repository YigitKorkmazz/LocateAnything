#!/usr/bin/env python3
"""Read-only audit of saved ChestX-ray8 localization artifacts.

This script performs no generation, model loading, training, or evaluation.  It
checks existing manifests/logs and renders overlays from saved predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_locateanything_bbox import (  # noqa: E402
    box_iou,
    canonicalize_disease,
    convert_boxes_to_pixels,
)
from sft_common import xyxy_pixels_to_norm1000  # noqa: E402

SOURCE_CSV = Path(
    "/auto/data2/ykorkmaz/nih-chest-xrays/data/versions/3/BBox_List_2017.csv"
)
TRAIN = HERE / "splits/train80_pairs_seed42.jsonl"
TEST = HERE / "splits/test_pairs_seed42.jsonl"
INTERNAL = HERE / "splits/production_train90_validation10_seed42"
OPT = INTERNAL / "optimization90_of_train80_seed42.jsonl"
VAL = INTERNAL / "validation10_of_train80_seed42.jsonl"
SPLIT_META = INTERNAL / "split_manifest.json"
KL_FREE_METRICS = (
    HERE
    / "results/chestxray8_hybrid_grpo_native/training"
    / "two_gpu_18x18_production_500_train80_seed42_revalidated_20260808"
    / "two_gpu_g4_grpo_multistep_metrics.jsonl"
)
SAVED_BASE = (
    HERE
    / "results/chestxray8_hybrid_grpo_native/audit/fresh_start_regression_seed42"
    / "eight_group_rollout_only_probe.json"
)
OUTPUT = (
    HERE
    / "results/chestxray8_hybrid_grpo_native/audit"
    / "localization_correctness_20260809"
)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row):
    return row["image_index"], row["disease"]


def source_annotations():
    result = defaultdict(list)
    with SOURCE_CSV.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            image, raw_disease = row[0].strip(), row[1].strip()
            disease = canonicalize_disease(raw_disease)
            if disease is None:
                continue
            x, y, width, height = map(float, row[2:6])
            result[(image, disease)].append([x, y, x + width, y + height])
    return result


def boxes_close(left, right, tolerance=1e-9):
    if len(left) != len(right):
        return False
    left = sorted(tuple(box) for box in left)
    right = sorted(tuple(box) for box in right)
    return all(
        all(abs(a - b) <= tolerance for a, b in zip(one, two))
        for one, two in zip(left, right)
    )


def audit_manifest(path, source, *, open_images):
    rows = read_jsonl(path)
    failures = []
    max_roundtrip = {"x_pixels": 0.0, "y_pixels": 0.0}
    for index, pair in enumerate(rows):
        label = "{}:{}".format(path.name, index)
        image_path = Path(pair["image_path"])
        expected_patient = str(int(pair["image_index"].split("_")[0]))
        if image_path.name != pair["image_index"]:
            failures.append([label, "image_path basename mismatch"])
        if str(pair["patient_id"]) != expected_patient:
            failures.append([label, "patient_id mismatch"])
        if pair["prompt_phrase"] != pair["disease"]:
            failures.append([label, "prompt/disease mismatch"])
        if pair["user_query"] != "Locate the {} in this chest X-ray".format(pair["disease"]):
            failures.append([label, "query/disease mismatch"])
        if key(pair) not in source:
            failures.append([label, "image/disease absent from source CSV"])
            continue
        if not boxes_close(pair["gt_boxes_xyxy_px"], source[key(pair)]):
            failures.append([label, "bbox differs from source x,y,w,h conversion"])
        width, height = pair["image_width"], pair["image_height"]
        if open_images:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
            if [width, height] != [actual_width, actual_height]:
                failures.append([label, "manifest/Pillow dimensions mismatch"])
        expected_norm = xyxy_pixels_to_norm1000(pair["gt_boxes_xyxy_px"], width, height)
        if expected_norm != pair["gt_boxes_norm_1000"]:
            failures.append([label, "normalized bbox mismatch"])
        if pair["n_gt_boxes"] != len(pair["gt_boxes_xyxy_px"]):
            failures.append([label, "n_gt_boxes mismatch"])
        for norm_box, pixel_box in zip(expected_norm, pair["gt_boxes_xyxy_px"]):
            if not (0 <= norm_box[0] < norm_box[2] <= 1000 and 0 <= norm_box[1] < norm_box[3] <= 1000):
                failures.append([label, "invalid normalized xyxy geometry"])
            roundtrip = convert_boxes_to_pixels([norm_box], width, height)[0]
            x_error = max(abs(roundtrip[0] - pixel_box[0]), abs(roundtrip[2] - pixel_box[2]))
            y_error = max(abs(roundtrip[1] - pixel_box[1]), abs(roundtrip[3] - pixel_box[3]))
            max_roundtrip["x_pixels"] = max(max_roundtrip["x_pixels"], x_error)
            max_roundtrip["y_pixels"] = max(max_roundtrip["y_pixels"], y_error)
            if x_error > width / 2000.0 + 1e-9 or y_error > height / 2000.0 + 1e-9:
                failures.append([label, "roundtrip exceeds nearest-integer bound"])
    return {
        "path": str(path),
        "sha256": sha256(path),
        "row_count": len(rows),
        "opened_images_for_dimensions": open_images,
        "max_roundtrip_absolute_error_pixels": max_roundtrip,
        "failure_count": len(failures),
        "failures": failures,
    }


def quantile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = probability * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def extract_ious():
    train = read_jsonl(TRAIN)
    records = []
    groups = []
    with KL_FREE_METRICS.open() as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            group = json.loads(line)
            trajectories = group["trajectories"]
            if len(trajectories) != 4:
                raise AssertionError("saved KL-free group is not G=4")
            ious = [float(item["reward"]["final_iou"]) for item in trajectories]
            spatial = [float(item["reward"]["spatial_reward"]) for item in trajectories]
            groups.append(
                {
                    "attempted_group_count": group["attempted_group_count"],
                    "sample_index": group["sample_index"],
                    "ious": ious,
                    "spatial": spatial,
                    "raw_iou_population_variance": statistics.pvariance(ious),
                    "spatial_population_variance": statistics.pvariance(spatial),
                }
            )
            if line_index < 50:
                pair = train[group["sample_index"]]
                for trajectory in trajectories:
                    reward = trajectory["reward"]
                    records.append(
                        {
                            "completed_group_ordinal": line_index + 1,
                            "attempted_group_count": group["attempted_group_count"],
                            "sample_index": group["sample_index"],
                            "image_index": pair["image_index"],
                            "disease": pair["disease"],
                            "trajectory_index": trajectory["group_index"],
                            "rollout_seed": trajectory["rollout_seed"],
                            "box_norm_1000": reward["final_box_norm_1000"],
                            "geometry_valid": reward["geometry_valid"],
                            "raw_iou": float(reward["final_iou"]),
                            "binary_spatial_reward": float(reward["spatial_reward"]),
                        }
                    )
    if len(records) != 200:
        raise AssertionError("expected exactly 200 first-50 trajectory rows")
    fields = list(records[0])
    with (OUTPUT / "first50_klfree_raw_ious.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            record = dict(record)
            record["box_norm_1000"] = json.dumps(record["box_norm_1000"], separators=(",", ":"))
            writer.writerow(record)
    first_ious = [record["raw_iou"] for record in records]
    first_groups = groups[:50]
    return {
        "source": str(KL_FREE_METRICS),
        "source_sha256": sha256(KL_FREE_METRICS),
        "completed_group_count": len(groups),
        "first_50": {
            "group_count": 50,
            "trajectory_count": 200,
            "raw_iou_min": min(first_ious),
            "raw_iou_q25": quantile(first_ious, 0.25),
            "raw_iou_median": quantile(first_ious, 0.5),
            "raw_iou_q75": quantile(first_ious, 0.75),
            "raw_iou_max": max(first_ious),
            "raw_iou_mean": statistics.fmean(first_ious),
            "trajectories_iou_gt_0_5": sum(value > 0.5 for value in first_ious),
            "groups_with_nonzero_raw_iou_variance": sum(
                group["raw_iou_population_variance"] > 0 for group in first_groups
            ),
            "groups_with_nonzero_binary_spatial_variance": sum(
                group["spatial_population_variance"] > 0 for group in first_groups
            ),
            "all_binary_spatial_zero_groups": sum(
                group["spatial"] == [0.0] * 4 for group in first_groups
            ),
            "csv": str(OUTPUT / "first50_klfree_raw_ious.csv"),
        },
        "all_completed_groups": {
            "groups_with_nonzero_raw_iou_variance": sum(
                group["raw_iou_population_variance"] > 0 for group in groups
            ),
            "groups_with_nonzero_binary_spatial_variance": sum(
                group["spatial_population_variance"] > 0 for group in groups
            ),
            "all_binary_spatial_zero_groups": sum(
                group["spatial"] == [0.0] * 4 for group in groups
            ),
            "all_binary_spatial_one_groups": sum(
                group["spatial"] == [1.0] * 4 for group in groups
            ),
            "trajectory_count": len(groups) * 4,
            "trajectories_iou_gt_0_5": sum(
                value > 0.5 for group in groups for value in group["ious"]
            ),
        },
    }


def px_box(norm_box, width, height):
    return tuple(round(value) for value in convert_boxes_to_pixels([norm_box], width, height)[0])


def draw_overlay(pair, prediction, destination, split_name):
    with Image.open(pair["image_path"]) as source:
        image = source.convert("RGB")
    width, height = image.size
    banner_height = 94
    canvas = Image.new("RGB", (width, height + banner_height), "black")
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for gt in pair["gt_boxes_norm_1000"]:
        x1, y1, x2, y2 = px_box(gt, width, height)
        draw.rectangle((x1, y1 + banner_height, x2, y2 + banner_height), outline=(255, 40, 40), width=4)
    pred_text = "base prediction unavailable in saved outputs"
    pred_iou = None
    if prediction is not None:
        box = prediction["reward"]["final_box_norm_1000"]
        x1, y1, x2, y2 = px_box(box, width, height)
        draw.rectangle((x1, y1 + banner_height, x2, y2 + banner_height), outline=(40, 255, 40), width=4)
        pred_iou = max(box_iou(box, gt) for gt in pair["gt_boxes_norm_1000"])
        pred_text = "GREEN saved base={} IoU={:.6f}".format(box, pred_iou)
    lines = [
        "{} | {} | {} | {}x{}".format(split_name, pair["image_index"], pair["disease"], width, height),
        "RED transformed GT norm1000={}".format(pair["gt_boxes_norm_1000"]),
        pred_text,
    ]
    for line_index, text in enumerate(lines):
        draw.text((8, 8 + line_index * 27), text, fill="white", font=font)
    canvas.save(destination)
    return pred_iou


def render_overlays():
    train = read_jsonl(TRAIN)
    opt_keys = {key(row) for row in read_jsonl(OPT)}
    val_keys = {key(row) for row in read_jsonl(VAL)}
    base = json.loads(SAVED_BASE.read_text())
    saved_predictions = {}
    for group in base["groups"]:
        valid = [
            item
            for item in group["trajectories"]
            if item["reward"]["geometry_valid"]
            and item["reward"]["final_box_norm_1000"] is not None
        ]
        if valid:
            saved_predictions[group["sample_index"]] = valid[0]

    selected = list(saved_predictions)
    selected_keys = {key(train[index]) for index in selected}
    # Guarantee that the 20-image visual set represents internal validation.
    for index, pair in enumerate(train):
        if len([i for i in selected if key(train[i]) in val_keys]) >= 10:
            break
        if key(pair) in val_keys and key(pair) not in selected_keys:
            selected.append(index)
            selected_keys.add(key(pair))
    for index, pair in enumerate(train):
        if len(selected) >= 20:
            break
        if key(pair) not in selected_keys:
            selected.append(index)
            selected_keys.add(key(pair))
    selected = selected[:20]
    overlay_dir = OUTPUT / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ordinal, index in enumerate(selected, 1):
        pair = train[index]
        split_name = "internal_validation" if key(pair) in val_keys else "optimization_train"
        if key(pair) not in opt_keys | val_keys:
            raise AssertionError("overlay selection escaped train/internal-validation pool")
        filename = "{:02d}_{}_{}.png".format(ordinal, pair["image_index"].replace(".png", ""), pair["disease"])
        prediction = saved_predictions.get(index)
        pred_iou = draw_overlay(pair, prediction, overlay_dir / filename, split_name)
        manifest.append(
            {
                "ordinal": ordinal,
                "source_train80_index": index,
                "split": split_name,
                "image_index": pair["image_index"],
                "disease": pair["disease"],
                "gt_boxes_norm_1000": pair["gt_boxes_norm_1000"],
                "saved_base_prediction_available": prediction is not None,
                "saved_base_box_norm_1000": (
                    prediction["reward"]["final_box_norm_1000"] if prediction else None
                ),
                "saved_base_iou_recomputed": pred_iou,
                "overlay": str(overlay_dir / filename),
            }
        )
    (OUTPUT / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "count": len(manifest),
        "optimization_train_count": sum(row["split"] == "optimization_train" for row in manifest),
        "internal_validation_count": sum(row["split"] == "internal_validation" for row in manifest),
        "saved_base_prediction_count": sum(row["saved_base_prediction_available"] for row in manifest),
        "heldout_test_count": 0,
        "saved_base_source": str(SAVED_BASE),
        "saved_base_source_sha256": sha256(SAVED_BASE),
        "manifest": str(OUTPUT / "overlay_manifest.json"),
        "directory": str(overlay_dir),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = source_annotations()
    split_meta = json.loads(SPLIT_META.read_text())
    manifests = {
        "train80": audit_manifest(TRAIN, source, open_images=True),
        # Metadata/coordinate audit only: this is not a prediction evaluation.
        "heldout_test_metadata_only": audit_manifest(TEST, source, open_images=True),
        "optimization": audit_manifest(OPT, source, open_images=False),
        "internal_validation": audit_manifest(VAL, source, open_images=False),
    }
    train_keys = {key(row) for row in read_jsonl(TRAIN)}
    test_keys = {key(row) for row in read_jsonl(TEST)}
    opt_keys = {key(row) for row in read_jsonl(OPT)}
    val_keys = {key(row) for row in read_jsonl(VAL)}
    partition = {
        "optimization_union_validation_equals_train80": opt_keys | val_keys == train_keys,
        "optimization_intersection_validation_empty": not bool(opt_keys & val_keys),
        "train80_intersection_heldout_test_empty": not bool(train_keys & test_keys),
        "recorded_patient_overlap": split_meta["patient_overlap"],
        "all_recorded_patient_overlaps_false": not any(split_meta["patient_overlap"].values()),
        "hashes_match_split_manifest": {
            "train80": sha256(TRAIN) == split_meta["source_train_sha256"],
            "heldout_test": sha256(TEST) == split_meta["heldout_test_sha256"],
            "optimization": sha256(OPT) == split_meta["optimization_sha256"],
            "internal_validation": sha256(VAL) == split_meta["validation_sha256"],
        },
    }
    iou_audit = extract_ious()
    overlays = render_overlays()
    all_correct = (
        all(item["failure_count"] == 0 for item in manifests.values())
        and partition["optimization_union_validation_equals_train80"]
        and partition["optimization_intersection_validation_empty"]
        and partition["train80_intersection_heldout_test_empty"]
        and partition["all_recorded_patient_overlaps_false"]
        and all(partition["hashes_match_split_manifest"].values())
        and iou_audit["completed_group_count"] == 578
        and iou_audit["all_completed_groups"]["groups_with_nonzero_binary_spatial_variance"] == 0
        and overlays["count"] == 20
        and overlays["heldout_test_count"] == 0
    )
    result = {
        "audit_format": "chestxray8_localization_correctness_audit_v1",
        "binary_verdict": (
            "localization pipeline correct" if all_correct else "localization pipeline has a bug"
        ),
        "prohibited_actions": {
            "training_run": False,
            "new_model_inference": False,
            "heldout_prediction_evaluation": False,
        },
        "source_coordinate_convention": {
            "csv_fields": ["image_index", "finding_label", "x", "y", "width", "height"],
            "label_alias": {"Infiltrate": "Infiltration"},
            "origin": "top-left",
            "conversion": "[x, y, x+width, y+height] -> round([x/W,y/H,x2/W,y2/H]*1000)",
            "normalized_domain": "continuous image-edge coordinates in inclusive numeric range [0,1000]",
            "right_bottom_full_image_boundary": 1000,
            "resize_or_padding_offsets_in_gt_transform": "none",
            "dimension_source": "PIL Image.open(image_path).size persisted as image_width,image_height",
        },
        "preprocessing_geometry": {
            "model_processor": "LocateAnythingImageProcessor.rescale",
            "initial_token_limit_scaling": "uniform width/height scale (aspect ratio preserved)",
            "patch_multiple_adjustment": "width and height resized independently up to multiples of 28",
            "padding_operation": False,
            "padding_offsets": [0, 0],
            "why_no_gt_remap_is_needed": (
                "predictions and GT are both normalized fractions; IoU is invariant to independent "
                "positive x/y scaling, and the processor introduces no translated coordinate origin"
            ),
            "dataset_dimensions_verified_from": "Pillow image files, not processor tensors",
            "width_height_swap_detected": False,
        },
        "oracle_tests": {
            "path": str(HERE / "test_localization_correctness_audit.py"),
            "passed": 7,
            "failed": 0,
            "checks": [
                "exact transformed GT gives IoU 1",
                "exact transformed GT gives spatial reward 1",
                "pixel-normalized-pixel roundtrip within half normalized unit",
                "asymmetric 1600x900 xy/order/no-width-height-swap oracle",
                "1000 full boundary and 999 one normalized unit inside",
                "IoU invariance under independent axis scaling",
                "training and evaluator import the same reward/IoU objects",
            ],
        },
        "train_evaluator_reward_path": {
            "training_call": "rewards.score_from_trace(trace, pair)",
            "evaluation_call": "rewards.score_from_trace(trace, pair)",
            "shared_factory": "rl.rewards.build_reward_pipeline_from_config",
            "shared_iou": "eval_locateanything_bbox.box_iou imported by rl.rewards",
            "shared_gt_field": "pair['gt_boxes_norm_1000']",
            "spatial_threshold": 0.5,
            "spatial_comparison": "strict greater_than",
        },
        "manifests": manifests,
        "partition_alignment": partition,
        "kl_free_saved_raw_iou": iou_audit,
        "diagnostic_overlays": overlays,
        "existing_base_heldout_context_not_rerun": {
            "source": str(
                HERE
                / "results/chestxray8_hybrid_grpo_native/evaluation"
                / "final_step500_heldout_test_seed42_20260808_v3/base_aggregate.json"
            ),
            "n_samples": 194,
            "iou_gt_0_5_accuracy": 0.020618556701030927,
            "success_count": 4,
            "interpretation": (
                "nonzero marginal success on different held-out examples does not imply mixed "
                "success/failure among four trajectories for the same training example"
            ),
        },
        "g8_probe": {
            "prepared": True,
            "run": False,
            "static_contract_validation_passed": True,
            "config": str(HERE / "rl/chestxray8_grpo_hybrid_native_g8_lora_only_probe.yaml"),
            "launcher": str(HERE / "two_gpu_g8_one_update_memory_probe.py"),
        },
    }
    (OUTPUT / "localization_correctness_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "binary_verdict": result["binary_verdict"],
        "first_50": iou_audit["first_50"],
        "all_completed_groups": iou_audit["all_completed_groups"],
        "overlays": overlays,
        "output": str(OUTPUT / "localization_correctness_audit.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
