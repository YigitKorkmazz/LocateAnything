#!/usr/bin/env python3
"""Create paired G=8 Case-B internal-validation selection artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

EXPECTED_MANIFEST_SHA256 = "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838"
ORDER = ["frozen_base", "step_025", "step_050", "step_075", "step_100", "step_125", "step_150"]
TIE_TOLERANCE_MEAN_IOU = 0.005
VALID_BOX_COLLAPSE_TOLERANCE = 0.05


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def point(condition, metric):
    return float(condition["aggregate"]["metrics"][metric]["point"])


def paired_bootstrap_mean_iou(base, candidate, *, replicates=10000, seed=20260810):
    deltas = [float(right["iou"]) - float(left["iou"]) for left, right in zip(base, candidate)]
    rng, n = random.Random(seed), len(deltas)
    samples = sorted(statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(replicates))
    return {
        "point": statistics.fmean(deltas),
        "ci95_low": samples[int(0.025 * (replicates - 1))],
        "ci95_high": samples[int(0.975 * (replicates - 1))],
        "replicates": replicates,
        "seed": seed,
    }


def paired_delta(base, candidate, field):
    return statistics.fmean(float(right[field]) - float(left[field]) for left, right in zip(base, candidate))


def verify_alignment(base, candidate):
    for left, right in zip(base, candidate):
        left_id = (left["sample_index"], left["image_index"], left["disease"], left["seed"])
        right_id = (right["sample_index"], right["image_index"], right["disease"], right["seed"])
        if left_id != right_id:
            raise RuntimeError("paired sample/seed alignment failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-aggregate", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    aggregate_path = Path(args.evaluation_aggregate).resolve()
    output = Path(args.output_dir).resolve()
    report = json.loads(aggregate_path.read_text())
    if report["split_label"] != "internal_validation":
        raise RuntimeError("refusing non-internal-validation results")
    if report["manifest_sha256"] != EXPECTED_MANIFEST_SHA256 or report.get("heldout_test_used") is not False:
        raise RuntimeError("validation manifest/held-out isolation contract failed")
    if list(report["conditions"]) != ORDER:
        raise RuntimeError("expected frozen base and all six ordered checkpoints")

    repo = Path(__file__).resolve().parents[3]
    python = "/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python"
    evaluator = "Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py"
    config = "Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml"
    manifest = "Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl"
    def evaluation_command(output_path, condition_args):
        return "tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; {} {} --config {} --output-dir {} --manifest {} --manifest-sha256 {} --split-label internal_validation {} --bootstrap-replicates 2000 --bootstrap-seed 20260810'".format(
            python, evaluator, config, output_path, manifest,
            EXPECTED_MANIFEST_SHA256, " ".join(condition_args)
        )
    initial_conditions = ["--condition frozen_base=base"] + [
        "--condition {}={}".format(label, Path(report["conditions"][label]["checkpoint"]).relative_to(repo))
        for label in ORDER[1:]
    ]
    initial_command = evaluation_command(output.relative_to(repo), initial_conditions)
    evaluation_commands = [initial_command]
    for label in ORDER[4:]:
        item = report["conditions"][label]
        isolated_output = Path(item["per_sample_jsonl"]).parent.relative_to(repo)
        checkpoint = Path(item["checkpoint"]).relative_to(repo)
        evaluation_commands.append(evaluation_command(
            isolated_output, ["--condition {}={}".format(label, checkpoint)]
        ))

    rows = {label: read_jsonl(report["conditions"][label]["per_sample_jsonl"]) for label in ORDER}
    if any(len(value) != 80 for value in rows.values()):
        raise RuntimeError("every condition must contain exactly 80 samples")
    for label in ORDER[1:]:
        verify_alignment(rows["frozen_base"], rows[label])

    conditions = {}
    for label in ORDER:
        item = report["conditions"][label]
        aggregate = item["aggregate"]
        metrics = aggregate["metrics"]
        conditions[label] = {
            "checkpoint": item["checkpoint"],
            "n_samples": aggregate["n_samples"],
            "valid_native_box_rate": point(item, "valid_native_box_rate"),
            "malformed_or_no_box_rate": point(item, "malformed_or_no_box_rate"),
            "mean_iou": point(item, "mean_iou"),
            "median_iou": point(item, "median_iou"),
            "strict_iou_gt_0_5_rate": point(item, "iou_gt_0_5"),
            "mean_semantic_reward": point(item, "mean_medclip_semantic_reward"),
            "mean_total_reward": point(item, "total_reward_mean"),
            "mean_normalized_box_area_among_valid": point(item, "mean_box_area_norm01_among_valid"),
            "near_full_image_rate_all_samples": point(item, "near_full_image_box_rate"),
            "near_full_image_rate_among_valid": point(item, "near_full_image_box_rate_among_valid"),
            "branch_distribution_counts": aggregate["branch_distribution"],
            "branch_distribution_rates": {
                name: count / aggregate["n_samples"] for name, count in aggregate["branch_distribution"].items()
            },
            "mean_generated_token_count": point(item, "average_generated_token_count"),
            "truncation_count": aggregate["truncation_count"],
            "per_sample_jsonl": item["per_sample_jsonl"],
        }

    base_rows = rows["frozen_base"]
    base_metrics = conditions["frozen_base"]
    paired = {}
    for index, label in enumerate(ORDER[1:]):
        candidate_rows = rows[label]
        mean_iou = paired_bootstrap_mean_iou(base_rows, candidate_rows, seed=20260810 + index)
        paired[label] = {
            "mean_iou_delta": mean_iou,
            "median_iou_delta": statistics.median(float(row["iou"]) for row in candidate_rows)
            - statistics.median(float(row["iou"]) for row in base_rows),
            "strict_iou_gt_0_5_rate_delta": paired_delta(base_rows, candidate_rows, "iou_gt_0_5"),
            "valid_native_box_rate_delta": paired_delta(base_rows, candidate_rows, "valid_native_box"),
            "near_full_image_rate_all_samples_delta": paired_delta(base_rows, candidate_rows, "near_full_image"),
        }

    eligible = [
        label for label in ORDER[1:]
        if conditions[label]["valid_native_box_rate"]
        >= base_metrics["valid_native_box_rate"] - VALID_BOX_COLLAPSE_TOLERANCE
    ]
    if not eligible:
        raise RuntimeError("all checkpoints failed the valid-box non-collapse gate")
    numerical_best_mean = max(conditions[label]["mean_iou"] for label in eligible)
    effectively_tied = [
        label for label in eligible
        if numerical_best_mean - conditions[label]["mean_iou"] <= TIE_TOLERANCE_MEAN_IOU
    ]
    selected = sorted(
        effectively_tied,
        key=lambda label: (
            -conditions[label]["strict_iou_gt_0_5_rate"],
            conditions[label]["near_full_image_rate_all_samples"],
            -conditions[label]["valid_native_box_rate"],
            ORDER.index(label),
        ),
    )[0]
    selected_ci = paired[selected]["mean_iou_delta"]
    if selected_ci["ci95_low"] > 0:
        verdict = "localization improved"
    elif selected_ci["ci95_high"] < 0:
        verdict = "localization worsened"
    else:
        verdict = "localization unchanged"

    step100_improved = conditions["step_100"]["mean_iou"] > base_metrics["mean_iou"]
    selection = {
        "primary_rule": "highest validation mean IoU",
        "effective_tie_definition": "within 0.005 absolute mean IoU of the numerical best",
        "tie_breakers": ["higher strict IoU > 0.5 rate", "lower near-full-image rate over all 80 samples", "higher valid-native-box rate", "earlier step"],
        "valid_box_noncollapse_gate": "checkpoint valid-box rate >= frozen-base rate - 0.05",
        "eligible_checkpoints": eligible,
        "effectively_tied_checkpoints": effectively_tied,
        "numerical_best_mean_iou": numerical_best_mean,
        "selected_label": selected,
        "selected_checkpoint": conditions[selected]["checkpoint"],
        "selected_verdict_vs_frozen_base": verdict,
    }
    stopping = {
        "rule": "if step100 mean IoU has not improved over frozen base, flag unlikely to benefit from continuing to 500",
        "frozen_base_mean_iou": base_metrics["mean_iou"],
        "step100_mean_iou": conditions["step_100"]["mean_iou"],
        "step100_delta": conditions["step_100"]["mean_iou"] - base_metrics["mean_iou"],
        "step100_improved_strictly": step100_improved,
        "flag_unlikely_to_benefit_from_continuing_to_500": not step100_improved,
        "continuing_beyond_150_justified_by_step100_rule": step100_improved,
        "automatic_action": "none",
    }
    result = {
        "format": "g8_caseb_internal_validation_selection_v1",
        "scope": "internal validation model selection only; not final test performance",
        "manifest": report["manifest"],
        "manifest_sha256": report["manifest_sha256"],
        "sample_count": 80,
        "heldout_test_used": False,
        "same_samples_and_seeds_verified": True,
        "conditions": conditions,
        "paired_deltas_vs_frozen_base": paired,
        "selection": selection,
        "step100_stopping_rule": stopping,
        "concise_verdict": verdict,
        "evaluation_commands": evaluation_commands,
        "assembly_command": report.get("assembly_command"),
        "summarizer_command": " ".join(sys.argv),
    }
    json_path = output / "g8_caseb_validation_metrics.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    csv_path = output / "g8_caseb_validation_summary.csv"
    fields = [
        "condition", "checkpoint", "valid_native_box_rate", "malformed_or_no_box_rate", "mean_iou", "median_iou",
        "strict_iou_gt_0_5_rate", "mean_semantic_reward", "mean_total_reward", "mean_normalized_box_area_among_valid",
        "near_full_image_rate_all_samples", "near_full_image_rate_among_valid", "pbd_rate", "ntp_fallback_rate", "none_rate",
        "mean_generated_token_count", "truncation_count", "mean_iou_delta_vs_base", "mean_iou_delta_ci95_low",
        "mean_iou_delta_ci95_high", "median_iou_delta_vs_base", "strict_iou_gt_0_5_delta_vs_base",
        "valid_box_delta_vs_base", "near_full_image_delta_vs_base",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for label in ORDER:
            item = conditions[label]; delta = paired.get(label)
            writer.writerow({
                "condition": label, "checkpoint": item["checkpoint"],
                **{key: item[key] for key in fields if key in item},
                "pbd_rate": item["branch_distribution_rates"]["pbd"],
                "ntp_fallback_rate": item["branch_distribution_rates"]["ntp_fallback"],
                "none_rate": item["branch_distribution_rates"]["none"],
                "mean_iou_delta_vs_base": delta["mean_iou_delta"]["point"] if delta else 0.0,
                "mean_iou_delta_ci95_low": delta["mean_iou_delta"]["ci95_low"] if delta else 0.0,
                "mean_iou_delta_ci95_high": delta["mean_iou_delta"]["ci95_high"] if delta else 0.0,
                "median_iou_delta_vs_base": delta["median_iou_delta"] if delta else 0.0,
                "strict_iou_gt_0_5_delta_vs_base": delta["strict_iou_gt_0_5_rate_delta"] if delta else 0.0,
                "valid_box_delta_vs_base": delta["valid_native_box_rate_delta"] if delta else 0.0,
                "near_full_image_delta_vs_base": delta["near_full_image_rate_all_samples_delta"] if delta else 0.0,
            })

    def pct(value): return "{:.2%}".format(value)
    lines = [
        "# G8 Case-B internal validation report", "", "## Scope", "",
        "This report uses only the pinned 80-example internal-validation manifest. The 194-case held-out test was not evaluated or used for selection. These are model-selection results, not final test performance.", "",
        "## Per-condition metrics", "",
        "| Condition | Valid | Malformed/no-box | Mean IoU | Median IoU | IoU > .5 | Semantic | Total | Area (valid) | Near-full (all) | PBD/NTP/none | Tokens | Trunc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ORDER:
        item = conditions[label]; branch = item["branch_distribution_rates"]
        lines.append("| {} | {} | {} | {:.6f} | {:.6f} | {} | {:.6f} | {:.6f} | {:.6f} | {} | {}/{}/{} | {:.2f} | {} |".format(
            label, pct(item["valid_native_box_rate"]), pct(item["malformed_or_no_box_rate"]), item["mean_iou"], item["median_iou"],
            pct(item["strict_iou_gt_0_5_rate"]), item["mean_semantic_reward"], item["mean_total_reward"],
            item["mean_normalized_box_area_among_valid"], pct(item["near_full_image_rate_all_samples"]),
            pct(branch["pbd"]), pct(branch["ntp_fallback"]), pct(branch["none"]), item["mean_generated_token_count"], item["truncation_count"]))
    lines += ["", "Mean normalized area is among valid native boxes. Near-full paired/model-selection rate is over all 80 samples; near-full means width and height are both at least 0.9.", "", "## Paired deltas versus frozen base", "",
              "| Checkpoint | Mean IoU delta (95% CI) | Median IoU delta | IoU>.5 delta | Valid delta | Near-full delta |", "|---|---:|---:|---:|---:|---:|"]
    for label in ORDER[1:]:
        delta = paired[label]; ci = delta["mean_iou_delta"]
        lines.append("| {} | {:+.6f} [{:+.6f}, {:+.6f}] | {:+.6f} | {:+.2%} | {:+.2%} | {:+.2%} |".format(
            label, ci["point"], ci["ci95_low"], ci["ci95_high"], delta["median_iou_delta"],
            delta["strict_iou_gt_0_5_rate_delta"], delta["valid_native_box_rate_delta"], delta["near_full_image_rate_all_samples_delta"]))
    lines += ["", "## Selection", "", "Selected: **{}** (`{}`).".format(selected, conditions[selected]["checkpoint"]), "",
              "Rule: highest mean IoU; conditions within 0.005 are effectively tied, then higher strict IoU>.5, lower near-full rate, and non-collapsed valid-box rate.", "",
              "## Step-100 stopping rule", "", "Frozen-base mean IoU: `{:.6f}`; step-100: `{:.6f}`; delta: `{:+.6f}`.".format(stopping["frozen_base_mean_iou"], stopping["step100_mean_iou"], stopping["step100_delta"]), "",
              "Continuing beyond 150 justified by the existing step-100 rule: **{}**. No automatic action was taken.".format("yes" if step100_improved else "no"), "",
              "## Verdict", "", "**{}**".format(verdict), "", "## Reproducibility", "", "The initial combined command completed frozen base and steps 25/50/75, then encountered cross-condition CUDA allocator accumulation while starting step 100. Steps 100/125/150 were rerun successfully using the following fresh-process commands.", "", "Evaluation commands used:", "", "```bash", "\n\n".join(evaluation_commands), "```", "", "Assembly command:", "", "```bash", "{} {}".format(python, report.get("assembly_command")), "```", "", "Summarizer command:", "", "```bash", "{} {}".format(python, " ".join(sys.argv)), "```", ""]
    (output / "G8_CASEB_VALIDATION_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "report": str(output / 'G8_CASEB_VALIDATION_REPORT.md'), "selected": selected, "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
