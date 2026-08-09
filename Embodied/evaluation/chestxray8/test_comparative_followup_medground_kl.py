#!/usr/bin/env python3
"""CPU oracles for the three-way comparative follow-up evaluator."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from eval_comparative_followup_medground_kl import (  # noqa: E402
    analyze_training_metrics,
    generic_box_collapse,
    paired_comparison,
)


def _row(index: int, box, value: float):
    return {
        "sample_index": index,
        "image_index": f"image-{index}",
        "seed": 42 + index,
        "valid_native_box": float(box is not None),
        "committed_final_bbox_norm_1000": box,
        "iou": value,
        "iou_gt_0_5": float(value > 0.5),
        "semantic_reward": value / 10.0,
        "total_reward": value,
    }


def test_generic_box_collapse_statistics() -> None:
    rows = [
        _row(0, [0, 0, 1000, 1000], 0.1),
        _row(1, [0, 0, 1000, 1000], 0.2),
        _row(2, [100, 200, 400, 600], 0.3),
        _row(3, None, 0.0),
    ]
    result = generic_box_collapse(rows)
    assert result["valid_box_count"] == 3
    assert math.isclose(result["width"]["mean"], (1.0 + 1.0 + 0.3) / 3)
    assert result["near_full_image_count"] == 2
    assert math.isclose(result["near_full_image_rate_among_valid"], 2 / 3)
    assert result["repeated_exact_coordinate_tuple_count"] == 1
    assert math.isclose(result["repeated_exact_coordinate_rate_among_valid"], 2 / 3)
    assert result["top_exact_coordinates"][0]["box"] == [0, 0, 1000, 1000]


def test_paired_comparison_direction_and_alignment() -> None:
    left = [_row(0, [1, 2, 3, 4], 0.2), _row(1, None, 0.4)]
    right = [_row(0, [1, 2, 3, 4], 0.5), _row(1, [5, 6, 7, 8], 0.8)]
    result = paired_comparison("left", "right", left, right, 100, 7)
    assert result["definition"] == "right - left"
    assert math.isclose(result["paired_deltas"]["mean_iou"]["point"], 0.35)
    assert math.isclose(
        result["paired_deltas"]["valid_native_box_rate"]["point"], 0.5
    )
    broken = [dict(right[0]), dict(right[1])]
    broken[0]["seed"] += 1
    try:
        paired_comparison("left", "right", left, broken, 10, 7)
    except RuntimeError as exc:
        assert "alignment" in str(exc)
    else:
        raise AssertionError("misaligned paired inputs were accepted")


def _metric_record(step: int) -> dict:
    trajectories = []
    branches = ("pbd", "ntp_fallback", "none", "pbd")
    for index, branch in enumerate(branches):
        trajectories.append(
            {
                "committed_branch": branch,
                "reference_gradients_present": False,
                "reward": {
                    "format_reward": float(index % 2 == 0),
                    "spatial_reward": float(index == 0),
                },
            }
        )
    return {
        "global_step": step,
        "attempted_group_count": step,
        "effective_kl_config": {"enabled": True, "beta": 0.04},
        "loss_components": {
            "mean_kl_value": step / 1000.0,
            "beta": 0.04,
            "mean_kl_loss_contribution": 0.04 * step / 1000.0,
            "mean_grpo_loss": -step / 100.0,
            "mean_total_loss": -step / 100.0 + 0.04 * step / 1000.0,
        },
        "global_grad_norm_before_clip": float(step),
        "global_grad_norm_after_clip": min(float(step), 1.0),
        "clipping_applied": step > 1,
        "trajectories": trajectories,
    }


def test_training_metrics_exact_steps_and_windows() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "metrics.jsonl"
        path.write_text(
            "".join(json.dumps(_metric_record(step)) + "\n" for step in range(1, 501)),
            encoding="utf-8",
        )
        result = analyze_training_metrics(path)
    assert result["record_count"] == 500
    assert len(result["per_step"]) == 500
    assert len(result["windows"]) == 10
    assert result["windows"][0]["start_step"] == 1
    assert result["windows"][-1]["end_step"] == 500
    assert math.isclose(result["overall"]["format_reward_positive_frequency"], 0.5)
    assert math.isclose(result["overall"]["spatial_reward_positive_frequency"], 0.25)
    assert math.isclose(result["overall"]["branch_rates"]["pbd"], 0.5)


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"COMPARATIVE FOLLOW-UP CPU TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
