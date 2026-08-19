"""Read-only rollout-group diagnostics for localization reward density.

These helpers consume already-computed traces and reward components.  They do
not participate in reward construction, advantage normalization, replay, or
optimization.
"""

from __future__ import annotations

import statistics
from itertools import combinations
from typing import Any, Dict, Mapping, Sequence


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _nonzero_variance(values: Sequence[float]) -> bool:
    return bool(values) and max(values) != min(values)


def _coordinate_std(boxes: Sequence[Sequence[float]]) -> float | None:
    if len(boxes) < 2:
        return None
    coordinate_stds = [
        statistics.pstdev(float(box[index]) for box in boxes) for index in range(4)
    ]
    return float(statistics.fmean(coordinate_stds))


def _mean_pairwise_iou(boxes: Sequence[Sequence[float]]) -> float | None:
    if len(boxes) < 2:
        return None
    values = [_bbox_iou(first, second) for first, second in combinations(boxes, 2)]
    return float(statistics.fmean(values))


def build_group_spatial_diagnostic(
    traces,
    components,
    *,
    expected_group_size: int,
) -> Dict[str, Any]:
    """Build the required per-attempt arrays and group-level measurements."""
    if len(traces) != expected_group_size or len(components) != expected_group_size:
        raise RuntimeError("spatial diagnostic received the wrong group size")

    raw_ious = [float(component.final_iou) for component in components]
    spatial = [float(component.spatial_reward) for component in components]
    format_rewards = [float(component.format_reward) for component in components]
    semantic = [float(component.semantic_reward) for component in components]
    totals = [float(component.total_reward) for component in components]

    valid_boxes = []
    valid_flags = []
    near_full_flags = []
    area_fractions = []
    for trace, component in zip(traces, components):
        box = getattr(trace, "committed_final_box_norm_1000", None)
        valid = bool(
            component.format_valid
            and component.geometry_valid
            and component.has_unambiguous_committed_box
            and box is not None
        )
        valid_flags.append(float(valid))
        if not valid:
            near_full_flags.append(0.0)
            continue
        normalized_box = [float(value) for value in box]
        valid_boxes.append(normalized_box)
        width = max(0.0, (normalized_box[2] - normalized_box[0]) / 1000.0)
        height = max(0.0, (normalized_box[3] - normalized_box[1]) / 1000.0)
        near_full_flags.append(float(width >= 0.9 and height >= 0.9))
        area_fractions.append(width * height)

    positives = sum(value == 1.0 for value in spatial)
    iou_range = max(raw_ious) - min(raw_ious)
    group_class = (
        "all_spatial_0"
        if positives == 0
        else "all_spatial_1"
        if positives == expected_group_size
        else "mixed_spatial_0_1"
    )
    return {
        "raw_ious": raw_ious,
        "binary_spatial_rewards": spatial,
        "format_rewards": format_rewards,
        "semantic_rewards": semantic,
        "total_rewards": totals,
        "group_spatial_class": group_class,
        "contains_iou_gt_0_5": bool(positives),
        "group_max_raw_iou": max(raw_ious),
        "within_group_raw_iou_range": iou_range,
        "raw_iou_range_gt_0_05_but_identical_spatial_reward": bool(
            iou_range > 0.05 and not _nonzero_variance(spatial)
        ),
        "mean_pairwise_bbox_iou_within_group": _mean_pairwise_iou(valid_boxes),
        "mean_coordinate_std_norm_1000": _coordinate_std(valid_boxes),
        "valid_box_rate": float(statistics.fmean(valid_flags)),
        "near_full_box_rate": float(statistics.fmean(near_full_flags)),
        "mean_predicted_box_area_fraction": (
            float(statistics.fmean(area_fractions)) if area_fractions else 0.0
        ),
        "predicted_box_area_fractions": area_fractions,
        "valid_committed_boxes": valid_boxes,
        "reward_nonzero_variance": {
            "format": _nonzero_variance(format_rewards),
            "spatial": _nonzero_variance(spatial),
            "semantic": _nonzero_variance(semantic),
            "total_reward": _nonzero_variance(totals),
        },
    }


def aggregate_spatial_density_window(
    groups: Sequence[Mapping[str, Any]],
    *,
    optimizer_step_end: int,
    interval: int,
) -> Dict[str, Any]:
    """Aggregate all attempted groups since the previous update boundary."""
    if not groups:
        raise RuntimeError("cannot aggregate an empty spatial-density window")
    if optimizer_step_end <= 0 or optimizer_step_end % interval:
        raise RuntimeError("spatial-density window must end on its update interval")

    def mean(values: Sequence[float]) -> float:
        return float(statistics.fmean(values)) if values else 0.0

    def fraction(predicate) -> float:
        return mean([float(predicate(group)) for group in groups])

    all_ious = [float(value) for group in groups for value in group["raw_ious"]]
    pairwise = [
        float(group["mean_pairwise_bbox_iou_within_group"])
        for group in groups
        if group["mean_pairwise_bbox_iou_within_group"] is not None
    ]
    coordinate_stds = [
        float(group["mean_coordinate_std_norm_1000"])
        for group in groups
        if group["mean_coordinate_std_norm_1000"] is not None
    ]
    area_fractions = [
        float(value)
        for group in groups
        for value in group["predicted_box_area_fractions"]
    ]
    reward_names = ("format", "spatial", "semantic", "total_reward")
    spatial_density = {
        "fraction_all_spatial_0": fraction(
            lambda group: group["group_spatial_class"] == "all_spatial_0"
        ),
        "fraction_mixed_spatial_0_1": fraction(
            lambda group: group["group_spatial_class"] == "mixed_spatial_0_1"
        ),
        "fraction_all_spatial_1": fraction(
            lambda group: group["group_spatial_class"] == "all_spatial_1"
        ),
        "rollout_p_iou_gt_0_5": mean([float(value > 0.5) for value in all_ious]),
        "p_group_contains_ge_1_iou_gt_0_5": fraction(
            lambda group: group["contains_iou_gt_0_5"]
        ),
        "mean_group_max_raw_iou": mean(
            [float(group["group_max_raw_iou"]) for group in groups]
        ),
        "mean_within_group_raw_iou_range": mean(
            [float(group["within_group_raw_iou_range"]) for group in groups]
        ),
        "fraction_groups_raw_iou_range_gt_0_05_but_identical_spatial_reward": fraction(
            lambda group: group[
                "raw_iou_range_gt_0_05_but_identical_spatial_reward"
            ]
        ),
    }
    return {
        "event": "spatial_density_25_update_window",
        "optimizer_step_start_exclusive": optimizer_step_end - interval,
        "optimizer_step_end_inclusive": optimizer_step_end,
        "attempted_group_count_in_window": len(groups),
        "rollout_count_in_window": len(all_ious),
        "spatial_density": spatial_density,
        "exploration": {
            "mean_pairwise_bbox_iou_within_group": mean(pairwise),
            "groups_with_pairwise_bbox_iou": len(pairwise),
            "mean_coordinate_std_norm_1000": mean(coordinate_stds),
            "groups_with_coordinate_std": len(coordinate_stds),
            "valid_box_rate": mean(
                [float(group["valid_box_rate"]) for group in groups]
            ),
            "near_full_box_rate": mean(
                [float(group["near_full_box_rate"]) for group in groups]
            ),
            "mean_predicted_box_area_fraction": mean(area_fractions),
        },
        "reward_variation": {
            "fraction_groups_nonzero_variance": {
                name: fraction(
                    lambda group, reward_name=name: group["reward_nonzero_variance"][
                        reward_name
                    ]
                )
                for name in reward_names
            }
        },
        "reference_context": {
            "previous_ntp_g8_mixed_spatial": 0.07,
            "previous_ntp_g8_rollout_p_iou_gt_0_5": 0.014,
            "frozen_base_exploration_mixed_spatial": 0.20,
            "frozen_base_exploration_rollout_p_iou_gt_0_5": 0.05,
            "hard_pass_threshold_applied": False,
            "mixed_spatial_delta_vs_previous": (
                spatial_density["fraction_mixed_spatial_0_1"] - 0.07
            ),
            "rollout_p_iou_gt_0_5_delta_vs_previous": (
                spatial_density["rollout_p_iou_gt_0_5"] - 0.014
            ),
        },
    }


def validate_spatial_density_config(config: Mapping[str, Any], *, group_size: int) -> int:
    configured = config.get("diagnostics", {}).get("spatial_density")
    if not configured or not bool(configured.get("enabled", False)):
        return 0
    expected = {
        "enabled": True,
        "aggregate_every_optimizer_updates": 25,
        "log_every_attempted_group": True,
        "expected_group_size": 8,
    }
    if configured != expected:
        raise RuntimeError("spatial-density diagnostic config differs from contract")
    if group_size != expected["expected_group_size"]:
        raise RuntimeError("spatial-density diagnostics require G=8")
    return int(expected["aggregate_every_optimizer_updates"])
