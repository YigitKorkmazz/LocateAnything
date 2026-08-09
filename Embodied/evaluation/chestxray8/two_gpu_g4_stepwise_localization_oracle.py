#!/usr/bin/env python3
"""CPU-only update-6/7 localization oracle for exact live-cache GRPO runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from rl.inprocess_short_fixture_oracle import normalize_ab_by_a12_baseline
from two_gpu_exactness_oracle import _write_report_safely
from two_gpu_g4_resume_equivalence_oracle import (
    _control_projection, _exact_comparison, _identity_projection, _load,
    _optimizer_scale_aware, _records, _state_compare,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a6", required=True); p.add_argument("--a7", required=True)
    p.add_argument("--b6", required=True); p.add_argument("--b7", required=True)
    p.add_argument("--r6", required=True); p.add_argument("--r7", required=True)
    p.add_argument("--a-metrics", required=True); p.add_argument("--b-metrics", required=True); p.add_argument("--r-metrics", required=True)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _trajectory_at(metrics: str, step: int) -> Dict[str, Any]:
    records = _records(metrics)
    row = records.get(step)
    if row is None: return {"status": "missing"}
    return {"status": "ok", "control": _control_projection(row), "policy": _identity_projection(row)}


def _update_report(a_path: str, b_path: str, r_path: str) -> Dict[str, Any]:
    a, b, r = _load(a_path), _load(b_path), _load(r_path)
    ab, ar = _state_compare(a, b), _state_compare(a, r)
    lora_normalized = normalize_ab_by_a12_baseline(ar["lora"]["global"], ab["lora"]["global"])
    optimizer = _optimizer_scale_aware(a, b, r)
    return {"checkpoint_paths": {"A": a_path, "B": b_path, "R": r_path}, "A_vs_B_baseline": ab,
            "A_vs_R": ar, "lora_baseline_normalized_A_vs_R": lora_normalized,
            "optimizer_scale_aware_A_vs_R": optimizer,
            "acceptance": {"lora_no_worse_than_matched_A_B": bool(lora_normalized["ab_no_worse_than_a12_baseline"]),
                           "optimizer_no_state_exceeds_matched_raw_and_scale_baseline": optimizer["no_state_exceeds_matched_ab_raw_and_scale_baseline"],
                           "adamw_step_fields_exact": optimizer["step_fields_exact"],
                           "all_optimizer_state_finite": bool(ar["optimizer"]["global"]["both_sides_all_finite"])}}


def main() -> None:
    args = parse_args(); output = Path(args.output_json).resolve(); report: Dict[str, Any] = {"status": "running"}
    try:
        update6 = _update_report(args.a6, args.b6, args.r6)
        update7 = _update_report(args.a7, args.b7, args.r7)
        trajectories = {}
        for step in (6, 7):
            a, b, r = _trajectory_at(args.a_metrics, step), _trajectory_at(args.b_metrics, step), _trajectory_at(args.r_metrics, step)
            trajectories[str(step)] = {
                "A_vs_B_control": _exact_comparison(a.get("control"), b.get("control"), f"update_{step}.A_B.control"),
                "A_vs_R_control": _exact_comparison(a.get("control"), r.get("control"), f"update_{step}.A_R.control"),
                "A_vs_B_policy": _exact_comparison(a.get("policy"), b.get("policy"), f"update_{step}.A_B.policy"),
                "A_vs_R_policy": _exact_comparison(a.get("policy"), r.get("policy"), f"update_{step}.A_R.policy"),
            }
        u6_ok = all(update6["acceptance"].values())
        u7_policy = trajectories["7"]["A_vs_R_policy"]["equal"]
        if u6_ok:
            interpretation = "A: R6 is no worse than A6/B6; resume restore/first resumed update is consistent with the matched baseline."
            if not u7_policy:
                interpretation += " Update-7 policy divergence is reported as autoregressive amplification after baseline-scale numerical state differences."
        else:
            interpretation = "B: R6 exceeds the matched A6/B6 state baseline; inspect optimizer_scale_aware_A_vs_R.worst_raw_max_abs_state."
        report.update({"mode": "update_5_6_7_localization", "update_6": update6, "update_7": update7,
                       "trajectory_identity": trajectories, "interpretation": interpretation,
                       "acceptance": {"update6_matched_baseline": u6_ok,
                                      "control_state_exact_updates_6_7": all(trajectories[str(s)]["A_vs_R_control"]["equal"] for s in (6, 7)),
                                      "update6_policy_identity_exact": trajectories["6"]["A_vs_R_policy"]["equal"]},
                       "status": "passed" if u6_ok else "localization_failed_update6"})
    except Exception as exc:
        report.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
    _write_report_safely(output, report); print(json.dumps({"status": report["status"], "output": str(output)}))
    if report["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__": main()
