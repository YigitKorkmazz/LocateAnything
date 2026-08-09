#!/usr/bin/env python3
"""Three-way comparative follow-up evaluation after prior held-out observation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from eval_final_two_gpu_hybrid_grpo import (  # noqa: E402
    PINNED_BASE_REVISION,
    _bootstrap,
    _checkpoint_preflight,
    _evaluate_condition,
    _overlap,
    _read_jsonl,
    _point,
)
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    load_resolved_config,
    sampling_from_config,
    sha256_file,
    write_json,
)
from two_gpu_g4_grpo_multistep_smoke import _validate_config  # noqa: E402


EVALUATION_LABEL = "comparative_followup_after_prior_heldout_observation"
EXPECTED_TEST_SAMPLES = 194
EXPECTED_TRAINING_STEPS = 500
CONDITIONS = ("base", "kl_free_step500", "medground_kl_step500")
PAIR_DEFINITIONS = (
    ("medground_kl_minus_kl_free", "kl_free_step500", "medground_kl_step500"),
    ("kl_free_minus_base", "base", "kl_free_step500"),
    ("medground_kl_minus_base", "base", "medground_kl_step500"),
)
PAIRED_METRICS = {
    "valid_native_box_rate": "valid_native_box",
    "mean_iou": "iou",
    "iou_gt_0_5_accuracy": "iou_gt_0_5",
    "mean_semantic_medclip_score": "semantic_reward",
    "mean_total_reward": "total_reward",
}
REPORT_METRICS = (
    ("Valid native box rate", "valid_native_box_rate"),
    ("Malformed/no-box rate", "malformed_or_no_box_rate"),
    ("Mean IoU", "mean_iou"),
    ("Median IoU", "median_iou"),
    ("IoU > 0.5 accuracy", "iou_gt_0_5_accuracy"),
    ("Mean semantic MedCLIP", "mean_semantic_medclip_score"),
    ("Mean format reward", "mean_format_reward"),
    ("Mean spatial reward", "mean_spatial_reward"),
    ("Mean semantic reward", "mean_semantic_reward"),
    ("Mean total reward", "mean_total_reward"),
    ("PBD branch rate", "pbd_branch_rate"),
    ("NTP-fallback branch rate", "ntp_fallback_branch_rate"),
    ("None branch rate", "none_branch_rate"),
    ("Mean generated tokens", "mean_generated_tokens"),
    ("Truncation rate", "truncation_rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG))
    parser.add_argument("--kl-free-checkpoint", required=True)
    parser.add_argument("--medground-kl-checkpoint", required=True)
    parser.add_argument("--medground-kl-training-metrics", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    return parser.parse_args()


def _semantic_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    sampling = sampling_from_config(config)
    return {
        "model_revision": config["model"]["revision"],
        "prompt_mode": config["prompt"]["mode"],
        "prompt_template": config["prompt"]["template"],
        "parser": config["rewards"]["parser"],
        "format": dict(config["rewards"]["format"]),
        "spatial": dict(config["rewards"]["spatial"]),
        "semantic": dict(config["rewards"]["semantic"]),
        "hybrid": dict(config["rollout"]["hybrid"]),
        "sampling": {
            "temperature": sampling.temperature,
            "top_k": sampling.top_k,
            "top_p": sampling.top_p,
            "repetition_penalty": sampling.repetition_penalty,
            "block_size": sampling.block_size,
            "max_new_tokens": int(config["evaluation"]["max_new_tokens"]),
        },
    }


def _preflight(
    args: argparse.Namespace, config: Dict[str, Any]
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _validate_config(config)
    if config["model"]["revision"] != PINNED_BASE_REVISION:
        raise RuntimeError("configured base revision differs from the pinned revision")
    if int(args.bootstrap_replicates) <= 0:
        raise ValueError("--bootstrap-replicates must be positive")

    manifest = Path(args.manifest).resolve()
    train_manifest = Path(args.train_manifest).resolve()
    expected_hash = str(args.manifest_sha256)
    actual_hash = sha256_file(manifest)
    if actual_hash != expected_hash or config["data"]["test_sha256"] != expected_hash:
        raise RuntimeError("held-out manifest SHA-256 mismatch")
    configured_test = Path(config["data"]["test_split"])
    if not configured_test.is_absolute():
        configured_test = CHEST_DIR / configured_test
    if configured_test.resolve() != manifest:
        raise RuntimeError("explicit manifest is not the configured held-out test split")
    train_hash = sha256_file(train_manifest)
    if train_hash != config["data"]["train_sha256"]:
        raise RuntimeError("train80 manifest SHA-256 mismatch")

    test_rows, train_rows = _read_jsonl(manifest), _read_jsonl(train_manifest)
    if len(test_rows) != EXPECTED_TEST_SAMPLES:
        raise RuntimeError(
            f"expected exactly {EXPECTED_TEST_SAMPLES} held-out samples, found {len(test_rows)}"
        )
    overlaps = {
        key: _overlap(test_rows, train_rows, key)
        for key in ("patient_id", "image_index", "image_path")
    }
    if any(overlaps.values()):
        raise RuntimeError(f"held-out/train80 overlap detected: {overlaps}")

    checkpoints = {
        "kl_free_step500": _checkpoint_preflight(
            Path(args.kl_free_checkpoint).resolve()
        ),
        "medground_kl_step500": _checkpoint_preflight(
            Path(args.medground_kl_checkpoint).resolve()
        ),
    }
    return test_rows, {
        "status": "passed",
        "evaluation_label": EVALUATION_LABEL,
        "heldout_test_previously_observed": True,
        "interpretation": (
            "comparative follow-up; this is not a first-look pristine held-out evaluation"
        ),
        "test_manifest": str(manifest),
        "test_manifest_sha256": actual_hash,
        "test_sample_count": len(test_rows),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": train_hash,
        "zero_overlap": True,
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "pinned_base_revision": PINNED_BASE_REVISION,
        "checkpoints": checkpoints,
        "shared_semantic_contract": _semantic_contract(config),
        "same_samples_seeds_semantics_for_all_conditions": True,
        "training_performed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "inference_mode_required": True,
        "test_time_tuning": False,
    }


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"nonfinite training metric: {label}={result}")
    return result


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(statistics.fmean(materialized)) if materialized else 0.0


def _training_step_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    step = int(record["global_step"])
    loss = record["loss_components"]
    kl = _finite(loss["mean_kl_value"], f"step {step} KL")
    beta = _finite(loss["beta"], f"step {step} beta")
    beta_kl = _finite(
        loss["mean_kl_loss_contribution"], f"step {step} beta*KL"
    )
    if not math.isclose(beta_kl, beta * kl, rel_tol=1e-6, abs_tol=1e-7):
        raise RuntimeError(f"step {step} logged beta*KL is inconsistent")
    trajectories = record.get("trajectories") or []
    if len(trajectories) != 4:
        raise RuntimeError(f"step {step} does not contain exactly four trajectories")
    format_positive = 0
    spatial_positive = 0
    branches: Counter[str] = Counter()
    for trajectory in trajectories:
        reward = trajectory["reward"]
        format_positive += int(float(reward["format_reward"]) > 0.0)
        spatial_positive += int(float(reward["spatial_reward"]) > 0.0)
        branch = str(trajectory["committed_branch"])
        branches[branch] += 1
        if trajectory.get("reference_gradients_present") is not False:
            raise RuntimeError(f"step {step} reports a reference gradient")
    kl_contract = record.get("effective_kl_config") or {}
    if kl_contract.get("enabled") is not True or float(kl_contract.get("beta")) != 0.04:
        raise RuntimeError(f"step {step} does not carry the validated KL contract")
    return {
        "step": step,
        "attempted_group_count": int(record["attempted_group_count"]),
        "kl": kl,
        "beta": beta,
        "beta_times_kl": beta_kl,
        "grpo_loss": _finite(loss["mean_grpo_loss"], f"step {step} GRPO loss"),
        "total_loss": _finite(loss["mean_total_loss"], f"step {step} total loss"),
        "global_grad_norm_before_clip": _finite(
            record["global_grad_norm_before_clip"], f"step {step} gradient norm"
        ),
        "global_grad_norm_after_clip": _finite(
            record["global_grad_norm_after_clip"],
            f"step {step} clipped gradient norm",
        ),
        "clipping_applied": bool(record["clipping_applied"]),
        "format_reward_positive_count": format_positive,
        "format_reward_positive_rate": format_positive / 4.0,
        "spatial_reward_positive_count": spatial_positive,
        "spatial_reward_positive_rate": spatial_positive / 4.0,
        "branch_counts": {
            "pbd": int(branches.get("pbd", 0)),
            "ntp_fallback": int(branches.get("ntp_fallback", 0)),
            "none": int(branches.get("none", 0)),
        },
        "branch_rates": {
            "pbd": branches.get("pbd", 0) / 4.0,
            "ntp_fallback": branches.get("ntp_fallback", 0) / 4.0,
            "none": branches.get("none", 0) / 4.0,
        },
    }


def _window_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    branch_counts = {
        branch: sum(int(record["branch_counts"][branch]) for record in records)
        for branch in ("pbd", "ntp_fallback", "none")
    }
    trajectories = len(records) * 4
    return {
        "start_step": int(records[0]["step"]),
        "end_step": int(records[-1]["step"]),
        "step_count": len(records),
        "mean_kl": _mean(record["kl"] for record in records),
        "mean_beta_times_kl": _mean(
            record["beta_times_kl"] for record in records
        ),
        "mean_grpo_loss": _mean(record["grpo_loss"] for record in records),
        "mean_total_loss": _mean(record["total_loss"] for record in records),
        "mean_global_grad_norm_before_clip": _mean(
            record["global_grad_norm_before_clip"] for record in records
        ),
        "mean_global_grad_norm_after_clip": _mean(
            record["global_grad_norm_after_clip"] for record in records
        ),
        "clipping_rate": _mean(
            float(record["clipping_applied"]) for record in records
        ),
        "format_reward_positive_frequency": (
            sum(int(record["format_reward_positive_count"]) for record in records)
            / trajectories
        ),
        "spatial_reward_positive_frequency": (
            sum(int(record["spatial_reward_positive_count"]) for record in records)
            / trajectories
        ),
        "branch_counts": branch_counts,
        "branch_rates": {
            branch: count / trajectories for branch, count in branch_counts.items()
        },
    }


def analyze_training_metrics(path: Path) -> Dict[str, Any]:
    raw = _read_jsonl(path)
    if len(raw) != EXPECTED_TRAINING_STEPS:
        raise RuntimeError(
            f"expected {EXPECTED_TRAINING_STEPS} training records, found {len(raw)}"
        )
    per_step = [_training_step_record(record) for record in raw]
    actual_steps = [record["step"] for record in per_step]
    expected_steps = list(range(1, EXPECTED_TRAINING_STEPS + 1))
    if actual_steps != expected_steps:
        raise RuntimeError("training metrics do not contain exact ordered steps 1..500")
    windows = [
        _window_summary(per_step[start : start + 50])
        for start in range(0, EXPECTED_TRAINING_STEPS, 50)
    ]
    return {
        "source": str(path.resolve()),
        "record_count": len(per_step),
        "exact_steps_1_through_500": True,
        "window_size_steps": 50,
        "per_step": per_step,
        "windows": windows,
        "overall": _window_summary(per_step),
    }


def _distribution(values: Sequence[float]) -> Dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
    }


def generic_box_collapse(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    measurements: List[Dict[str, Any]] = []
    coordinates: Counter[tuple[int, int, int, int]] = Counter()
    for row in rows:
        box = row.get("committed_final_bbox_norm_1000")
        if not bool(row.get("valid_native_box")) or not isinstance(box, list) or len(box) != 4:
            continue
        coordinate = tuple(int(value) for value in box)
        x1, y1, x2, y2 = coordinate
        width = (x2 - x1) / 1000.0
        height = (y2 - y1) / 1000.0
        measurements.append(
            {
                "width": width,
                "height": height,
                "area": width * height,
                "center_x": (x1 + x2) / 2000.0,
                "center_y": (y1 + y2) / 2000.0,
                "near_full_image": width >= 0.9 and height >= 0.9,
            }
        )
        coordinates[coordinate] += 1
    valid = len(measurements)
    repeated = {box: count for box, count in coordinates.items() if count >= 2}
    repeated_instances = sum(repeated.values())
    most_common = coordinates.most_common(20)
    return {
        "definition": {
            "coordinate_scale": "normalized from integer [0,1000] to [0,1]",
            "population": "valid committed native boxes only unless denominator says all samples",
            "near_full_image": "width >= 0.90 and height >= 0.90",
            "repeated_coordinate": "an exact integer (x1,y1,x2,y2) tuple appearing at least twice",
        },
        "all_sample_count": len(rows),
        "valid_box_count": valid,
        "width": _distribution([item["width"] for item in measurements]),
        "height": _distribution([item["height"] for item in measurements]),
        "area": _distribution([item["area"] for item in measurements]),
        "center_x": _distribution([item["center_x"] for item in measurements]),
        "center_y": _distribution([item["center_y"] for item in measurements]),
        "near_full_image_count": sum(
            int(item["near_full_image"]) for item in measurements
        ),
        "near_full_image_rate_among_valid": (
            _mean(float(item["near_full_image"]) for item in measurements)
            if measurements
            else 0.0
        ),
        "near_full_image_rate_among_all_samples": (
            sum(int(item["near_full_image"]) for item in measurements) / len(rows)
            if rows
            else 0.0
        ),
        "unique_exact_coordinate_count": len(coordinates),
        "unique_exact_coordinate_rate_among_valid": (
            len(coordinates) / valid if valid else 0.0
        ),
        "repeated_exact_coordinate_tuple_count": len(repeated),
        "valid_boxes_in_repeated_tuples": repeated_instances,
        "repeated_exact_coordinate_rate_among_valid": (
            repeated_instances / valid if valid else 0.0
        ),
        "most_common_exact_coordinate_rate_among_valid": (
            most_common[0][1] / valid if most_common and valid else 0.0
        ),
        "top_exact_coordinates": [
            {"box": list(box), "count": count, "rate_among_valid": count / valid}
            for box, count in most_common
        ],
    }


def _aligned_pairs(
    left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]
) -> List[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if len(left_rows) != len(right_rows):
        raise RuntimeError("paired conditions have different sample counts")
    pairs = []
    for left, right in zip(left_rows, right_rows):
        identity = (left["sample_index"], left["image_index"], left["seed"])
        if identity != (right["sample_index"], right["image_index"], right["seed"]):
            raise RuntimeError("paired condition sample/seed alignment failure")
        pairs.append((left, right))
    return pairs


def paired_comparison(
    left_label: str,
    right_label: str,
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
) -> Dict[str, Any]:
    pairs = _aligned_pairs(left_rows, right_rows)
    deltas = {}
    for index, (name, field) in enumerate(PAIRED_METRICS.items()):
        values = [float(right[field]) - float(left[field]) for left, right in pairs]
        deltas[name] = _bootstrap(
            values, statistics.fmean, replicates, seed + index
        )
    return {
        "definition": f"{right_label} - {left_label}",
        "left_condition": left_label,
        "right_condition": right_label,
        "same_samples_and_seeds_verified": True,
        "n_paired_samples": len(pairs),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "paired_deltas": deltas,
    }


def _collapse_point(collapse: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = collapse
    for key in path:
        value = value[key]
    return float(value) if value is not None else float("nan")


def collapse_deltas(collapse: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "mean_width": ("width", "mean"),
        "mean_height": ("height", "mean"),
        "mean_area": ("area", "mean"),
        "mean_center_x": ("center_x", "mean"),
        "mean_center_y": ("center_y", "mean"),
        "near_full_image_rate_among_valid": ("near_full_image_rate_among_valid",),
        "repeated_exact_coordinate_rate_among_valid": (
            "repeated_exact_coordinate_rate_among_valid",
        ),
        "most_common_exact_coordinate_rate_among_valid": (
            "most_common_exact_coordinate_rate_among_valid",
        ),
    }
    result = {}
    for name, left, right in PAIR_DEFINITIONS:
        result[name] = {
            metric: _collapse_point(collapse[right], path)
            - _collapse_point(collapse[left], path)
            for metric, path in metrics.items()
        }
    return result


def _write_markdown(
    output: Path,
    preflight: Mapping[str, Any],
    aggregates: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    collapse: Mapping[str, Mapping[str, Any]],
    training: Mapping[str, Any],
    command: str,
) -> None:
    lines = [
        "# Comparative follow-up: base vs KL-free vs MedGround-KL",
        "",
        "> This is explicitly a comparative follow-up. The held-out test set was observed previously and is not a pristine first-look holdout.",
        "",
        f"- Samples: {preflight['test_sample_count']}",
        f"- Test SHA-256: `{preflight['test_manifest_sha256']}`",
        f"- Base revision: `{preflight['pinned_base_revision']}`",
        "- All conditions use identical sample order, seeds, prompt, parser, rewards, sampling, and Hybrid decoder semantics.",
        "- Execution is inference-only: eval mode, all parameters frozen, `torch.inference_mode()`, no optimizer/backward.",
        "",
        "## Three-way metrics",
        "",
        "| Metric | Base | KL-free step500 | MedGround-KL step500 |",
        "|---|---:|---:|---:|",
    ]
    for label, metric in REPORT_METRICS:
        lines.append(
            f"| {label} | {_point(aggregates['base'], metric):.8f} | "
            f"{_point(aggregates['kl_free_step500'], metric):.8f} | "
            f"{_point(aggregates['medground_kl_step500'], metric):.8f} |"
        )
    lines.extend(["", "## Paired deltas with bootstrap 95% CIs", ""])
    for name, _, _ in PAIR_DEFINITIONS:
        comparison = comparisons[name]
        lines.extend(
            [
                f"### {comparison['definition']}",
                "",
                "| Metric | Point | 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for metric, value in comparison["paired_deltas"].items():
            lines.append(
                f"| {metric} | {value['point']:.8f} | "
                f"[{value['ci95_low']:.8f}, {value['ci95_high']:.8f}] |"
            )
        lines.append("")
    lines.extend(
        [
            "## Generic-box collapse diagnostics",
            "",
            "Near-full-image is predeclared as width ≥ 0.90 and height ≥ 0.90. Repetition means an exact integer coordinate tuple occurs at least twice.",
            "",
            "| Condition | Mean width | Mean height | Mean area | Mean center | Near-full / valid | Repeated / valid | Most-common / valid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS:
        item = collapse[condition]
        lines.append(
            f"| {condition} | {item['width']['mean']!s} | {item['height']['mean']!s} | "
            f"{item['area']['mean']!s} | ({item['center_x']['mean']!s}, {item['center_y']['mean']!s}) | "
            f"{item['near_full_image_rate_among_valid']:.8f} | "
            f"{item['repeated_exact_coordinate_rate_among_valid']:.8f} | "
            f"{item['most_common_exact_coordinate_rate_among_valid']:.8f} |"
        )
    overall = training["overall"]
    lines.extend(
        [
            "",
            "## MedGround-KL training metrics (steps 1–500)",
            "",
            f"- Mean KL: {overall['mean_kl']:.8f}",
            f"- Mean beta×KL: {overall['mean_beta_times_kl']:.8f}",
            f"- Mean GRPO loss: {overall['mean_grpo_loss']:.8f}",
            f"- Mean total loss: {overall['mean_total_loss']:.8f}",
            f"- Mean gradient norm before/after clip: {overall['mean_global_grad_norm_before_clip']:.8f} / {overall['mean_global_grad_norm_after_clip']:.8f}",
            f"- Clipping rate: {overall['clipping_rate']:.8f}",
            f"- Format/spatial positive frequency: {overall['format_reward_positive_frequency']:.8f} / {overall['spatial_reward_positive_frequency']:.8f}",
            f"- Branch rates: {json.dumps(overall['branch_rates'], sort_keys=True)}",
            "",
            "Per-step and 50-step-window values are in `medground_kl_training_metrics_analysis.json`.",
            "",
            "## Exact command",
            "",
            "```tcsh",
            command,
            "```",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("requires exactly two visible CUDA devices")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("RTX 3090" not in name for name in gpu_names):
        raise RuntimeError(f"requires two visible RTX 3090 GPUs, found {gpu_names}")

    config = load_resolved_config(args.config)
    pairs, preflight = _preflight(args, config)
    preflight["visible_gpu_names"] = gpu_names
    training = analyze_training_metrics(
        Path(args.medground_kl_training_metrics).resolve()
    )
    write_json(output / "preflight.json", preflight)
    write_json(output / "medground_kl_training_metrics_analysis.json", training)
    print(json.dumps({"event": "preflight_passed", **preflight}, sort_keys=True), flush=True)

    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    condition_checkpoints = {
        "base": None,
        "kl_free_step500": Path(args.kl_free_checkpoint).resolve(),
        "medground_kl_step500": Path(args.medground_kl_checkpoint).resolve(),
    }
    rows: Dict[str, List[Dict[str, Any]]] = {}
    aggregates: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows[condition], aggregates[condition] = _evaluate_condition(
            condition,
            condition_checkpoints[condition],
            config=config,
            pairs=pairs,
            devices=devices,
            output=output,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )

    comparisons = {
        name: paired_comparison(
            left,
            right,
            rows[left],
            rows[right],
            args.bootstrap_replicates,
            args.bootstrap_seed + pair_index * 100,
        )
        for pair_index, (name, left, right) in enumerate(PAIR_DEFINITIONS)
    }
    collapse = {
        condition: generic_box_collapse(rows[condition]) for condition in CONDITIONS
    }
    collapse_report = {
        "evaluation_label": EVALUATION_LABEL,
        "conditions": collapse,
        "paired_point_deltas": collapse_deltas(collapse),
    }
    comparison_report = {
        "evaluation_label": EVALUATION_LABEL,
        "heldout_test_previously_observed": True,
        "n_samples": len(pairs),
        "same_samples_and_seeds_verified": True,
        "comparisons": comparisons,
        "condition_aggregate_json": {
            condition: str(output / f"{condition}_aggregate.json")
            for condition in CONDITIONS
        },
        "generic_box_collapse_json": str(output / "generic_box_collapse.json"),
        "training_metrics_analysis_json": str(
            output / "medground_kl_training_metrics_analysis.json"
        ),
        "training_performed": False,
        "optimizer_created": False,
        "backward_executed": False,
    }
    write_json(output / "generic_box_collapse.json", collapse_report)
    write_json(output / "comparative_followup_comparison.json", comparison_report)
    command = " ".join(sys.argv)
    _write_markdown(
        output,
        preflight,
        aggregates,
        comparisons,
        collapse,
        training,
        command,
    )
    write_json(
        output / "completion.json",
        {
            "status": "passed",
            "evaluation_label": EVALUATION_LABEL,
            "heldout_test_previously_observed": True,
            "output_dir": str(output),
            "report": str(output / "REPORT.md"),
            "comparison": str(output / "comparative_followup_comparison.json"),
            "training_performed": False,
            "optimizer_created": False,
            "backward_executed": False,
        },
    )
    print(
        json.dumps(
            {
                "event": "comparative_followup_evaluation_complete",
                "output_dir": str(output),
                "report": str(output / "REPORT.md"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
