#!/usr/bin/env python3
"""Apply the predeclared step-100 stopping flag to internal-validation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean_iou(report, label):
    return float(report["conditions"][label]["aggregate"]["metrics"]["mean_iou"]["point"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-aggregate", required=True)
    parser.add_argument("--step100-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_report = json.loads(Path(args.base_aggregate).read_text())
    step_report = json.loads(Path(args.step100_aggregate).read_text())
    if base_report["split_label"] != "internal_validation" or step_report["split_label"] != "internal_validation":
        raise RuntimeError("stopping rule may only consume internal-validation results")
    base = mean_iou(base_report, "frozen_base")
    step100 = mean_iou(step_report, "step_100")
    improved = step100 > base
    result = {
        "format": "g8_caseb_step100_stopping_rule_v1",
        "rule": "if step-100 validation mean IoU has not improved over frozen base, flag unlikely to benefit from continuing to 500",
        "frozen_base_mean_iou": base,
        "step100_mean_iou": step100,
        "delta": step100 - base,
        "improved_strictly": improved,
        "flag_unlikely_to_benefit_from_continuing_to_500": not improved,
        "automatic_stop_or_continue": False,
        "heldout_test_used": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
