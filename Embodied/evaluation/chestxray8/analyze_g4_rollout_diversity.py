#!/usr/bin/env python3
"""Analyze saved G=4 trajectory diversity without model execution."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = (
    HERE / "results/chestxray8_hybrid_grpo_native/training"
    / "two_gpu_18x18_production_500_train80_seed42_revalidated_20260808"
    / "two_gpu_g4_grpo_multistep_metrics.jsonl"
)
OUTPUT = (
    HERE / "results/chestxray8_hybrid_grpo_native/experiments"
    / "g8_caseb_localization_seed42_preparation_20260809"
)


def mean(values):
    return statistics.fmean(values) if values else None


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    pos = p * (len(values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    groups = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    all_pair_equal = []
    valid_pair_equal = []
    valid_pair_l1 = []
    valid_pair_l2 = []
    valid_pair_iou = []
    coordinate_stds = {name: [] for name in ("x1", "y1", "x2", "y2")}
    token_pair_equal = []
    token_unique_counts = []
    best_ious = []
    successes = []
    valid_box_count = 0
    trajectory_count = 0

    for group in groups:
        trajectories = group["trajectories"]
        assert len(trajectories) == 4
        boxes = [item["reward"]["final_box_norm_1000"] for item in trajectories]
        valid = [tuple(box) for box in boxes if box is not None and box[2] > box[0] and box[3] > box[1]]
        valid_box_count += len(valid)
        trajectory_count += 4
        for left, right in itertools.combinations(boxes, 2):
            all_pair_equal.append(left == right)
        for left, right in itertools.combinations(valid, 2):
            valid_pair_equal.append(left == right)
            diffs = [(a - b) / 1000.0 for a, b in zip(left, right)]
            valid_pair_l1.append(sum(abs(value) for value in diffs) / 4)
            valid_pair_l2.append(math.sqrt(sum(value * value for value in diffs)))
            valid_pair_iou.append(box_iou(left, right))
        if len(valid) >= 2:
            for coord_index, name in enumerate(("x1", "y1", "x2", "y2")):
                coordinate_stds[name].append(statistics.pstdev(box[coord_index] / 1000.0 for box in valid))

        checksums = [item["generated_token_ids_checksum"] for item in trajectories]
        token_unique_counts.append(len(set(checksums)))
        token_pair_equal.extend(left == right for left, right in itertools.combinations(checksums, 2))
        ious = [float(item["reward"]["final_iou"]) for item in trajectories]
        best_ious.append(max(ious))
        successes.extend(value > 0.5 for value in ious)

    bins = Counter()
    for value in best_ious:
        if value < 0.1:
            bins["0-.1"] += 1
        elif value < 0.25:
            bins[".1-.25"] += 1
        elif value < 0.4:
            bins[".25-.4"] += 1
        elif value <= 0.5:
            bins[".4-.5"] += 1
        else:
            bins[">.5"] += 1

    n_groups = len(groups)
    p = sum(successes) / len(successes)
    iid_mixed_g4 = 1 - (1 - p) ** 4 - p**4
    iid_mixed_g8 = 1 - (1 - p) ** 8 - p**8
    empirical_all_success_group_rate = sum(
        all(float(item["reward"]["final_iou"]) > 0.5 for item in group["trajectories"])
        for group in groups
    ) / n_groups
    empirical_mixed_g4 = sum(
        0 < sum(float(item["reward"]["final_iou"]) > 0.5 for item in group["trajectories"]) < 4
        for group in groups
    ) / n_groups
    paired_block_mixed_g8 = 2 * empirical_all_success_group_rate * (1 - empirical_all_success_group_rate)

    below_distances = [0.5 - value for value in best_ious if value <= 0.5]
    result = {
        "format": "saved_g4_rollout_diversity_v1",
        "source": str(SOURCE),
        "groups": n_groups,
        "trajectories": trajectory_count,
        "bbox": {
            "valid_box_count": valid_box_count,
            "valid_box_rate": valid_box_count / trajectory_count,
            "all_trajectory_pairs": len(all_pair_equal),
            "exact_duplicate_state_pair_rate_including_invalid_none": mean(all_pair_equal),
            "valid_valid_pairs": len(valid_pair_equal),
            "exact_duplicate_box_pair_rate_among_valid_pairs": mean(valid_pair_equal),
            "mean_pairwise_coordinate_l1_per_coordinate_norm01": mean(valid_pair_l1),
            "median_pairwise_coordinate_l1_per_coordinate_norm01": quantile(valid_pair_l1, 0.5),
            "mean_pairwise_coordinate_l2_norm01": mean(valid_pair_l2),
            "mean_pairwise_box_iou": mean(valid_pair_iou),
            "coordinate_population_std_norm01_mean_across_eligible_groups": {
                name: mean(values) for name, values in coordinate_stds.items()
            },
            "groups_with_at_least_two_valid_boxes": len(coordinate_stds["x1"]),
        },
        "token_sequences": {
            "evidence": "exact generated-token sequence equality via saved SHA-256 checksum; token IDs were not retained in the production JSONL",
            "trajectory_pairs": len(token_pair_equal),
            "exact_duplicate_sequence_pair_rate": mean(token_pair_equal),
            "mean_unique_sequences_per_group": mean(token_unique_counts),
            "groups_with_four_unique_sequences": sum(value == 4 for value in token_unique_counts),
            "groups_with_any_exact_sequence_duplicate": sum(value < 4 for value in token_unique_counts),
        },
        "best_raw_iou": {
            "mean": mean(best_ious),
            "median": quantile(best_ious, 0.5),
            "maximum": max(best_ious),
            "q90": quantile(best_ious, 0.9),
            "q95": quantile(best_ious, 0.95),
            "distance_below_0_5_for_nonwinning_groups": {
                "mean": mean(below_distances),
                "median": quantile(below_distances, 0.5),
                "minimum_closest_miss": min(below_distances),
            },
            "bins": {
                name: {"count": bins[name], "fraction": bins[name] / n_groups}
                for name in ("0-.1", ".1-.25", ".25-.4", ".4-.5", ">.5")
            },
        },
        "binary_spatial": {
            "successful_trajectories": sum(successes),
            "trajectory_success_rate": p,
            "observed_mixed_g4_groups": int(empirical_mixed_g4 * n_groups),
            "observed_mixed_g4_group_rate": empirical_mixed_g4,
            "all_success_g4_groups": int(empirical_all_success_group_rate * n_groups),
            "all_zero_g4_groups": sum(not any(float(item["reward"]["final_iou"]) > 0.5 for item in group["trajectories"]) for group in groups),
        },
        "g8_estimate": {
            "pooled_iid_assumption": {
                "mixed_g4_probability": iid_mixed_g4,
                "mixed_g8_probability": iid_mixed_g8,
                "absolute_increase": iid_mixed_g8 - iid_mixed_g4,
                "expected_mixed_groups_in_150_attempts_g8": 150 * iid_mixed_g8,
            },
            "heterogeneity_aware_two_observed_g4_blocks": {
                "mixed_g8_probability": paired_block_mixed_g8,
                "expected_mixed_groups_in_150_attempts": 150 * paired_block_mixed_g8,
                "caveat": "combines observed all-success/all-failure G4 block frequencies; prompts and policy steps are not replicated",
            },
            "diagnosis": "unlikely_to_materially_increase_mixed_binary_spatial_groups_under_observed_distribution",
            "reason": "successes were rare and perfectly group-clustered; doubling samples cannot create mixed rewards when per-prompt success probability is near 0 or 1",
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "saved_g4_rollout_diversity.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "best_iou_bins": result["best_raw_iou"]["bins"], "g8_estimate": result["g8_estimate"]}, indent=2))


if __name__ == "__main__":
    main()
