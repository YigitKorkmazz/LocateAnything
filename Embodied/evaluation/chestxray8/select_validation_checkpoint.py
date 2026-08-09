#!/usr/bin/env python3
"""Select a checkpoint strictly by internal-validation mean IoU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregate-json", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source = Path(args.aggregate_json).resolve(); report = json.loads(source.read_text(encoding="utf-8"))
    conditions = report["conditions"]
    candidates = [(label, item) for label, item in conditions.items() if item.get("checkpoint")]
    if not candidates: raise RuntimeError("no checkpoint conditions available for validation selection")
    label, item = max(candidates, key=lambda x: x[1]["aggregate"]["metrics"]["mean_iou"]["point"])
    result = {"format": "chestxray8_validation_checkpoint_selection_v1", "selection_rule": "maximize internal-validation mean_iou; total reward and malformed rate are diagnostics only",
              "validation_aggregate": str(source), "validation_manifest": report["manifest"], "validation_manifest_sha256": report["manifest_sha256"],
              "selected_label": label, "selected_checkpoint": item["checkpoint"],
              "selected_mean_iou": item["aggregate"]["metrics"]["mean_iou"]["point"],
              "secondary_diagnostics": {"total_reward_mean": item["aggregate"]["metrics"]["total_reward_mean"]["point"],
                                        "malformed_output_rate": item["aggregate"]["metrics"]["malformed_output_rate"]["point"]},
              "heldout_test_used_for_selection": False}
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"selection_manifest": str(output), "selected_checkpoint": result["selected_checkpoint"]}))


if __name__ == "__main__": main()
