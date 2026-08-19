#!/usr/bin/env python3
"""CPU-only G-subsampling analysis of an existing frozen-base rollout pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
DEFAULT_POOL = (
    HERE
    / "results/chestxray8_hybrid_grpo_native/validation"
    / "sampling_exploration_ntp_base_seed42/temp1.2_topp0.9_per_rollout.jsonl"
)
DEFAULT_META = DEFAULT_POOL.with_name("RUN_META.json")
DEFAULT_CONFIG = (
    HERE
    / "rl/chestxray8_grpo_native_g8_loraonly_projectorfrozen_ntponly_"
    "t1p2_topp0p9_100.yaml"
)
DEFAULT_OUTPUT_DIR = (
    HERE
    / "results/chestxray8_hybrid_grpo_native/validation"
    / "g_subsampling_spatial_sparsity_seed20260815"
)
GROUP_SIZES = (2, 4, 8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--run-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--repetitions-per-prompt", type=int, default=10_000)
    parser.add_argument("--pool-size", type=int, default=32)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_pools(path: Path, pool_size: int) -> tuple[dict[int, list[float]], dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"No rollouts found in {path}")

    expected_condition = {
        "temperature": 1.2,
        "top_p": 0.9,
        "top_k": 0,
        "decoding_mode": "ntp_only",
    }
    for key, expected in expected_condition.items():
        observed = {row.get(key) for row in rows}
        if observed != {expected}:
            raise RuntimeError(f"Unexpected {key}: expected {expected!r}, observed {observed!r}")

    seen: set[tuple[int, int]] = set()
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    reward_mismatches = 0
    for row in rows:
        sample_index = int(row["sample_index"])
        rollout_index = int(row["rollout_index"])
        key = (sample_index, rollout_index)
        if key in seen:
            raise RuntimeError(f"Duplicate rollout identity: {key}")
        seen.add(key)
        by_sample[sample_index].append(row)

        expected_reward = float(float(row["iou"]) > 0.5)
        if float(row["spatial_reward"]) != expected_reward:
            reward_mismatches += 1
    if reward_mismatches:
        raise RuntimeError(
            f"{reward_mismatches} rows violate spatial_reward = 1 iff IoU > 0.5"
        )

    rollout_count_distribution = Counter(len(group) for group in by_sample.values())
    complete: dict[int, list[float]] = {}
    for sample_index, group in sorted(by_sample.items()):
        if len(group) != pool_size:
            continue
        indices = sorted(int(row["rollout_index"]) for row in group)
        if indices != list(range(pool_size)):
            raise RuntimeError(
                f"Complete sample {sample_index} does not have rollout indices 0..{pool_size - 1}"
            )
        ordered = sorted(group, key=lambda row: int(row["rollout_index"]))
        complete[sample_index] = [float(row["iou"]) for row in ordered]

    if not complete:
        raise RuntimeError(f"No prompts have exactly {pool_size} rollouts")

    validation = {
        "raw_rows": len(rows),
        "raw_prompts": len(by_sample),
        "rollout_count_distribution": dict(sorted(rollout_count_distribution.items())),
        "complete_prompts_used": len(complete),
        "pool_size_used": pool_size,
        "excluded_incomplete_prompts": len(by_sample) - len(complete),
        "reward_definition": "spatial_reward = 1 iff IoU > 0.5, else 0",
        "reward_mismatches": reward_mismatches,
        "condition": expected_condition,
        "sample_indices_used": sorted(complete),
    }
    return complete, validation


def group_metrics(ious: list[float]) -> tuple[int, int, int, float, int]:
    positives = sum(iou > 0.5 for iou in ious)
    all_zero = int(positives == 0)
    all_one = int(positives == len(ious))
    mixed = int(0 < positives < len(ious))
    iou_range = max(ious) - min(ious)
    raw_range_binary_identical = int(iou_range > 0.05 and (all_zero or all_one))
    return all_zero, mixed, positives, iou_range, raw_range_binary_identical


def analyze_group_size(
    pools: dict[int, list[float]],
    group_size: int,
    repetitions_per_prompt: int,
    seed: int,
) -> dict[str, Any]:
    full_pool = group_size == len(next(iter(pools.values())))
    repetitions = 1 if full_pool else repetitions_per_prompt
    rng = random.Random(seed + group_size * 1009)

    all_zero: list[int] = []
    mixed: list[int] = []
    positives: list[int] = []
    iou_ranges: list[float] = []
    raw_range_binary_identical: list[int] = []
    all_one: list[int] = []

    for pool in pools.values():
        for _ in range(repetitions):
            chosen = list(pool) if full_pool else rng.sample(pool, group_size)
            a0, mix, n_pos, iou_range, raw_same = group_metrics(chosen)
            all_zero.append(a0)
            mixed.append(mix)
            positives.append(n_pos)
            iou_ranges.append(iou_range)
            raw_range_binary_identical.append(raw_same)
            all_one.append(int(n_pos == group_size))

    return {
        "G": group_size,
        "n_prompts": len(pools),
        "pool_rollouts_per_prompt": len(next(iter(pools.values()))),
        "subsampling_repetitions_per_prompt": repetitions,
        "n_groups_evaluated": len(all_zero),
        "all_zero_spatial_group_rate": statistics.fmean(all_zero),
        "mixed_spatial_group_rate": statistics.fmean(mixed),
        "group_contains_iou_gt_0_5_rate": statistics.fmean(
            int(value > 0) for value in positives
        ),
        "mean_positive_spatial_rewards_per_group": statistics.fmean(positives),
        "mean_raw_iou_range_within_group": statistics.fmean(iou_ranges),
        "raw_iou_range_gt_0_05_binary_identical_rate": statistics.fmean(
            raw_range_binary_identical
        ),
        "all_one_spatial_group_rate": statistics.fmean(all_one),
        "sampling_mode": "full pool; no subsampling" if full_pool else "without replacement",
        "rng_seed_for_G": seed + group_size * 1009,
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def choose_knee(results: list[dict[str, Any]]) -> int:
    baseline = float(results[0]["all_zero_spatial_group_rate"])
    for row in results[1:]:
        if baseline - float(row["all_zero_spatial_group_rate"]) >= 0.10:
            return int(row["G"])
    return int(results[-1]["G"])


def write_figure(path_png: Path, path_pdf: Path, results: list[dict[str, Any]], knee: int) -> None:
    gs = [int(row["G"]) for row in results]
    series = (
        ("all_zero_spatial_group_rate", "All-zero group", "#C44E52", "circle"),
        ("mixed_spatial_group_rate", "Mixed group", "#4C72B0", "square"),
        ("group_contains_iou_gt_0_5_rate", "Contains ≥1 IoU > 0.5", "#55A868", "triangle"),
    )
    width, height = 1800, 1140
    left, right, top, bottom = 180, 80, 150, 210
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom

    def x_coord(index: int) -> float:
        return plot_left + index * (plot_right - plot_left) / (len(gs) - 1)

    def y_coord(value: float) -> float:
        return plot_bottom - value * (plot_bottom - plot_top)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: "DejaVu Sans", sans-serif; fill: #222; }</style>',
        f'<text x="{width / 2}" y="78" text-anchor="middle" font-size="44" '
        'font-weight="bold">Spatial reward sparsity vs. group size</text>',
    ]
    for tenth in range(0, 11, 2):
        value = tenth / 10
        y = y_coord(value)
        svg.append(
            f'<line x1="{plot_left}" y1="{y}" x2="{plot_right}" y2="{y}" '
            'stroke="#D9D9D9" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{plot_left - 22}" y="{y + 9}" text-anchor="end" '
            f'font-size="25" fill="#444">{value:.1f}</text>'
        )
    svg.extend(
        [
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
            f'y2="{plot_bottom}" stroke="#333" stroke-width="3"/>',
            f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
            f'y2="{plot_bottom}" stroke="#333" stroke-width="3"/>',
        ]
    )

    knee_index = gs.index(knee)
    knee_x = x_coord(knee_index)
    svg.extend(
        [
            f'<line x1="{knee_x}" y1="{plot_top}" x2="{knee_x}" y2="{plot_bottom}" '
            'stroke="#8172B2" stroke-width="3" stroke-dasharray="14,14"/>',
            f'<text x="{knee_x + 14}" y="{plot_top + 34}" font-size="22" fill="#5B4C86">'
            f'<tspan x="{knee_x + 14}">G≈{knee}: all-zero rate ≥10 pp</tspan>'
            f'<tspan x="{knee_x + 14}" dy="28">below G=2</tspan></text>',
        ]
    )

    def svg_marker(x: float, y: float, color: str, kind: str) -> str:
        radius = 10
        if kind == "circle":
            return f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>'
        if kind == "square":
            return (
                f'<rect x="{x - radius}" y="{y - radius}" width="{2 * radius}" '
                f'height="{2 * radius}" fill="{color}"/>'
            )
        points = f"{x},{y - radius - 2} {x - radius - 2},{y + radius} {x + radius + 2},{y + radius}"
        return f'<polygon points="{points}" fill="{color}"/>'

    for key, _, color, kind in series:
        points = [
            (x_coord(index), y_coord(float(row[key])))
            for index, row in enumerate(results)
        ]
        point_text = " ".join(f"{x},{y}" for x, y in points)
        svg.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" '
            'stroke-width="6" stroke-linejoin="round"/>'
        )
        for x, y in points:
            svg.append(svg_marker(x, y, color, kind))

    for index, g in enumerate(gs):
        x = x_coord(index)
        svg.append(
            f'<text x="{x}" y="{plot_bottom + 42}" text-anchor="middle" '
            f'font-size="25">{g}</text>'
        )
    svg.extend(
        [
            f'<text x="{width / 2}" y="{plot_bottom + 98}" text-anchor="middle" '
            'font-size="29">Generated samples per GRPO group (G)</text>',
            f'<text x="54" y="{(plot_top + plot_bottom) / 2}" text-anchor="middle" '
            'font-size="29" transform="rotate(-90 54 '
            f'{(plot_top + plot_bottom) / 2})">Fraction of groups</text>',
        ]
    )

    legend_x, legend_y = plot_right - 420, plot_top + 260
    for index, (_, label, color, kind) in enumerate(series):
        y = legend_y + index * 52
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="{y + 10}" x2="{legend_x + 54}" '
                f'y2="{y + 10}" stroke="{color}" stroke-width="5"/>',
                svg_marker(legend_x + 27, y + 10, color, kind),
                f'<text x="{legend_x + 72}" y="{y + 18}" font-size="25">{label}</text>',
            ]
        )

    footer = (
        "Frozen LocateAnything-3B · NTP-only · T=1.2 · top_p=0.9 · "
        "29 internal-validation prompts; G=32 uses each full pool once"
    )
    svg.extend(
        [
            f'<text x="{width / 2}" y="{height - 40}" text-anchor="middle" '
            f'font-size="22" fill="#444">{footer}</text>',
            "</svg>",
        ]
    )
    svg_path = path_png.with_suffix(".svg")
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    subprocess.run(
        ["convert", "-density", "220", str(svg_path), str(path_png)],
        check=True,
        capture_output=True,
        text=True,
    )

    pdf_width, pdf_height = 900, 570
    scale = 0.5
    pdf = canvas.Canvas(str(path_pdf), pagesize=(pdf_width, pdf_height))
    pdf.setTitle("Spatial reward sparsity vs. group size")

    def py(y: float) -> float:
        return pdf_height - y * scale

    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawCentredString(pdf_width / 2, py(78), "Spatial reward sparsity vs. group size")
    for tenth in range(0, 11, 2):
        value = tenth / 10
        y = y_coord(value)
        pdf.setStrokeColorRGB(0.85, 0.85, 0.85)
        pdf.setLineWidth(1)
        pdf.line(plot_left * scale, py(y), plot_right * scale, py(y))
        pdf.setFillColorRGB(0.27, 0.27, 0.27)
        pdf.setFont("Helvetica", 12)
        pdf.drawRightString((plot_left - 22) * scale, py(y + 8), f"{value:.1f}")
    pdf.setStrokeColorRGB(0.2, 0.2, 0.2)
    pdf.setLineWidth(1.5)
    pdf.line(plot_left * scale, py(plot_top), plot_left * scale, py(plot_bottom))
    pdf.line(plot_left * scale, py(plot_bottom), plot_right * scale, py(plot_bottom))

    pdf.setStrokeColorRGB(0.51, 0.45, 0.70)
    pdf.setDash(7, 7)
    pdf.line(knee_x * scale, py(plot_top), knee_x * scale, py(plot_bottom))
    pdf.setDash()
    pdf.setFillColorRGB(0.36, 0.30, 0.53)
    pdf.setFont("Helvetica", 10)
    pdf.drawString((knee_x + 14) * scale, py(plot_top + 28), f"G~{knee}: all-zero rate >=10 pp")
    pdf.drawString((knee_x + 14) * scale, py(plot_top + 52), "below G=2")

    rgb = {"#C44E52": (0.77, 0.31, 0.32), "#4C72B0": (0.30, 0.45, 0.69), "#55A868": (0.33, 0.66, 0.41)}
    for key, _, color, kind in series:
        points = [
            (x_coord(index) * scale, py(y_coord(float(row[key]))))
            for index, row in enumerate(results)
        ]
        pdf.setStrokeColorRGB(*rgb[color])
        pdf.setFillColorRGB(*rgb[color])
        pdf.setLineWidth(3)
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        pdf.drawPath(path)
        for x, y in points:
            if kind == "circle":
                pdf.circle(x, y, 5, stroke=0, fill=1)
            elif kind == "square":
                pdf.rect(x - 5, y - 5, 10, 10, stroke=0, fill=1)
            else:
                marker_path = pdf.beginPath()
                marker_path.moveTo(x, y + 6)
                marker_path.lineTo(x - 6, y - 5)
                marker_path.lineTo(x + 6, y - 5)
                marker_path.close()
                pdf.drawPath(marker_path, stroke=0, fill=1)

    pdf.setFillColorRGB(0.13, 0.13, 0.13)
    pdf.setFont("Helvetica", 12)
    for index, g in enumerate(gs):
        pdf.drawCentredString(x_coord(index) * scale, py(plot_bottom + 42), str(g))
    pdf.setFont("Helvetica", 14)
    pdf.drawCentredString(pdf_width / 2, py(plot_bottom + 98), "Generated samples per GRPO group (G)")
    pdf.saveState()
    pdf.translate(27, pdf_height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Fraction of groups")
    pdf.restoreState()

    pdf.setFont("Helvetica", 11)
    legend_x_pdf, legend_y_pdf = legend_x * scale, legend_y * scale
    for index, (_, label, color, _) in enumerate(series):
        y = pdf_height - legend_y_pdf - index * 26
        pdf.setStrokeColorRGB(*rgb[color])
        pdf.setLineWidth(2.5)
        pdf.line(legend_x_pdf, y, legend_x_pdf + 27, y)
        pdf.setFillColorRGB(0.13, 0.13, 0.13)
        pdf.drawString(legend_x_pdf + 36, y - 4, label.replace("≥", ">="))
    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(
        pdf_width / 2,
        20,
        "Frozen LocateAnything-3B | NTP-only | T=1.2 | top_p=0.9 | "
        "29 internal-validation prompts; G=32 uses each full pool once",
    )
    pdf.save()


def interpretation(results: list[dict[str, Any]], knee: int) -> str:
    first, last = results[0], results[-1]
    mixed_steps = [
        float(results[index]["mixed_spatial_group_rate"])
        - float(results[index - 1]["mixed_spatial_group_rate"])
        for index in range(1, len(results))
    ]
    saturation_note = (
        "The mixed-group curve has not clearly saturated by G=32."
        if mixed_steps[-1] >= 0.02
        else "The mixed-group curve begins to saturate by G=32."
    )
    return (
        f"Across the same 29 complete internal-validation prompt pools, increasing G "
        f"from 2 to 32 reduced the all-zero spatial-group rate from "
        f"{100 * float(first['all_zero_spatial_group_rate']):.1f}% to "
        f"{100 * float(last['all_zero_spatial_group_rate']):.1f}% and increased the "
        f"mixed-group rate from {100 * float(first['mixed_spatial_group_rate']):.1f}% "
        f"to {100 * float(last['mixed_spatial_group_rate']):.1f}%. A practical easing "
        f"of sparsity starts around G≈{knee} (at least a 10 percentage-point drop in "
        f"all-zero groups versus G=2). {saturation_note}"
    )


def main() -> None:
    args = parse_args()
    if args.repetitions_per_prompt <= 0:
        raise ValueError("--repetitions-per-prompt must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pools, validation = load_and_validate_pools(args.pool, args.pool_size)
    results = [
        analyze_group_size(
            pools,
            group_size=group_size,
            repetitions_per_prompt=args.repetitions_per_prompt,
            seed=args.seed,
        )
        for group_size in GROUP_SIZES
    ]
    knee = choose_knee(results)

    csv_path = args.output_dir / "g_subsampling_spatial_sparsity.csv"
    png_path = args.output_dir / "g_subsampling_spatial_sparsity.png"
    pdf_path = args.output_dir / "g_subsampling_spatial_sparsity.pdf"
    metadata_path = args.output_dir / "analysis_metadata.json"
    update_path = args.output_dir / "supervisor_update.txt"

    write_csv(csv_path, results)
    write_figure(png_path, pdf_path, results, knee)
    update = interpretation(results, knee)

    run_meta = json.loads(args.run_meta.read_text(encoding="utf-8"))
    manifest_path = Path(run_meta["manifest"])
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != run_meta["manifest_sha256"]:
        raise RuntimeError(
            "Internal-validation manifest SHA256 does not match RUN_META.json: "
            f"{manifest_sha256}"
        )
    metadata = {
        "analysis_type": "CPU-only repeated subsampling of existing offline rollouts",
        "cuda_visible_devices": "",
        "new_inference_run": False,
        "training_run": False,
        "checkpoint_evaluation_run": False,
        "seed": args.seed,
        "subsampling_repetitions_per_prompt_for_G_lt_32": args.repetitions_per_prompt,
        "G32_note": "Each complete 32-rollout prompt pool is used once with no subsampling.",
        "meaningful_sparsity_decrease_G": knee,
        "source_files": {
            "rollout_pool": str(args.pool.resolve()),
            "rollout_pool_sha256": sha256_file(args.pool),
            "run_metadata": str(args.run_meta.resolve()),
            "run_metadata_sha256": sha256_file(args.run_meta),
            "experiment_config": str(args.config.resolve()),
            "experiment_config_sha256": sha256_file(args.config),
            "internal_validation_manifest": run_meta["manifest"],
            "internal_validation_manifest_sha256": manifest_sha256,
        },
        "source_metadata_note": (
            "RUN_META.json lists top_ps=[1.0], but every one of the 956 rows in the "
            "condition-specific rollout file records top_p=0.9; the condition-specific "
            "filename, partial summary, and experiment config also identify top_p=0.9. "
            "The analysis validates and uses the per-rollout records."
        ),
        "split": {
            "name": "internal validation subset",
            "heldout_test_used": bool(run_meta["heldout_test_used"]),
            "manifest": run_meta["manifest"],
        },
        "pool_validation": validation,
        "method": {
            "group_sizes": list(GROUP_SIZES),
            "sampling": "without replacement within each prompt pool",
            "prompt_weighting": "equal: the same repetition count for every prompt",
            "binary_reward": "1 iff raw IoU > 0.5; unchanged from source",
            "knee_rule": "first G with all-zero rate at least 0.10 below the G=2 rate",
        },
        "results": results,
        "supervisor_update": update,
        "output_files": {
            "csv": str(csv_path.resolve()),
            "png": str(png_path.resolve()),
            "pdf": str(pdf_path.resolve()),
            "svg": str(png_path.with_suffix(".svg").resolve()),
            "metadata": str(metadata_path.resolve()),
            "supervisor_update": str(update_path.resolve()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    update_path.write_text(update + "\n", encoding="utf-8")

    print("CUDA_VISIBLE_DEVICES=")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
