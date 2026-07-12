#!/usr/bin/env python3
"""
Build results/chestxray8_prompt_comparison.xlsx from bare_label and
radiology_context intermediate JSONL / Excel outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from openpyxl import Workbook

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sample_rows(jsonl_path: Path) -> Dict[Tuple[str, str], dict]:
    rows: Dict[Tuple[str, str], dict] = {}
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample = obj["sample_row"]
            key = (sample["Image"], sample["Disease"])
            rows[key] = sample
    return rows


def no_pred_rate(samples: List[dict]) -> float:
    if not samples:
        return 0.0
    return sum(1 for s in samples if int(s.get("Number of Predicted Boxes") or 0) == 0) / len(samples)


def recall_at(samples: List[dict], thresh: float) -> float:
    """GT-box-level recall approximated from per-sample Recall@t * n_gt / total_gt."""
    total_gt = sum(int(s.get("Number of GT Boxes") or 0) for s in samples)
    if total_gt == 0:
        return 0.0
    # Prefer reconstructing from Matched IoUs / GT if available via Mean isn't enough.
    # Use per-sample Recall@t * n_gt.
    key = {0.1: "Recall@0.1", 0.3: "Recall@0.3", 0.5: "Recall@0.5"}[thresh]
    matched = 0.0
    for s in samples:
        matched += float(s.get(key) or 0.0) * int(s.get("Number of GT Boxes") or 0)
    return matched / total_gt


def mean_iou_from_samples(samples: List[dict]) -> float:
    """Mean IoU over all GT boxes (unmatched GT count as 0)."""
    total = 0.0
    n_gt = 0
    for s in samples:
        n = int(s.get("Number of GT Boxes") or 0)
        n_gt += n
        matched = []
        raw = s.get("Matched IoUs")
        if raw:
            try:
                matched = json.loads(raw) if isinstance(raw, str) else list(raw)
            except json.JSONDecodeError:
                matched = []
        total += sum(float(v) for v in matched)
    if n_gt == 0:
        return 0.0
    return total / n_gt


def median_iou_from_samples(samples: List[dict]) -> float:
    ious: List[float] = []
    for s in samples:
        n = int(s.get("Number of GT Boxes") or 0)
        matched = []
        raw = s.get("Matched IoUs")
        if raw:
            try:
                matched = json.loads(raw) if isinstance(raw, str) else list(raw)
            except json.JSONDecodeError:
                matched = []
        ious.extend(float(v) for v in matched)
        ious.extend([0.0] * max(0, n - len(matched)))
    if not ious:
        return 0.0
    return float(np.median(ious))


def build_comparison(
    bare_rows: Dict[Tuple[str, str], dict],
    rad_rows: Dict[Tuple[str, str], dict],
) -> Tuple[List[dict], List[dict], List[dict]]:
    keys = sorted(set(bare_rows) | set(rad_rows), key=lambda x: (x[0], x[1]))
    bare_list = [bare_rows[k] for k in keys if k in bare_rows]
    rad_list = [rad_rows[k] for k in keys if k in rad_rows]

    overall = []
    for strategy, samples in [("bare_label", bare_list), ("radiology_context", rad_list)]:
        times = [
            float(s["Inference Time"])
            for s in samples
            if s.get("Status") == "ok" and s.get("Inference Time") is not None
        ]
        overall.append({
            "Prompt Strategy": strategy,
            "Total Image-Disease Pairs": len(samples),
            "Total GT Boxes": sum(int(s.get("Number of GT Boxes") or 0) for s in samples),
            "Total Predicted Boxes": sum(int(s.get("Number of Predicted Boxes") or 0) for s in samples),
            "No-Prediction Rate": no_pred_rate(samples),
            "Mean IoU": mean_iou_from_samples(samples),
            "Median IoU": median_iou_from_samples(samples),
            "Recall@0.1": recall_at(samples, 0.1),
            "Recall@0.3": recall_at(samples, 0.3),
            "Recall@0.5": recall_at(samples, 0.5),
            "Mean Inference Time": float(np.mean(times)) if times else 0.0,
            "Errors": sum(1 for s in samples if s.get("Status") != "ok"),
        })

    disease_rows = []
    diseases = sorted({k[1] for k in keys})
    for disease in diseases:
        for strategy, mapping in [("bare_label", bare_rows), ("radiology_context", rad_rows)]:
            samples = [mapping[k] for k in keys if k[1] == disease and k in mapping]
            disease_rows.append({
                "Disease": disease,
                "Prompt Strategy": strategy,
                "Number of Images": len({s["Image"] for s in samples}),
                "Number of GT Boxes": sum(int(s.get("Number of GT Boxes") or 0) for s in samples),
                "Number of Predicted Boxes": sum(
                    int(s.get("Number of Predicted Boxes") or 0) for s in samples
                ),
                "No-Prediction Rate": no_pred_rate(samples),
                "Mean IoU": mean_iou_from_samples(samples),
                "Median IoU": median_iou_from_samples(samples),
                "Recall@0.1": recall_at(samples, 0.1),
                "Recall@0.3": recall_at(samples, 0.3),
                "Recall@0.5": recall_at(samples, 0.5),
            })

    paired = []
    for key in keys:
        b = bare_rows.get(key)
        r = rad_rows.get(key)
        if b is None or r is None:
            continue
        bare_mean = float(b.get("Mean Matched IoU") or 0.0)
        # Use Maximum IoU; for mean over GT boxes prefer reconstructing
        bare_max = float(b.get("Maximum IoU") or 0.0)
        rad_mean = float(r.get("Mean Matched IoU") or 0.0)
        rad_max = float(r.get("Maximum IoU") or 0.0)
        # Better paired mean: mean over GT boxes (including unmatched=0)
        bare_gt_mean = mean_iou_from_samples([b])
        rad_gt_mean = mean_iou_from_samples([r])
        paired.append({
            "Image": key[0],
            "Disease": key[1],
            "Bare Label Prompt": b.get("Prompt Phrase") or b.get("Final Rendered User Query"),
            "Bare Label Number of Predictions": int(b.get("Number of Predicted Boxes") or 0),
            "Bare Label Mean IoU": bare_gt_mean,
            "Bare Label Maximum IoU": bare_max,
            "Radiology Context Prompt": r.get("Prompt Phrase") or r.get("Final Rendered User Query"),
            "Radiology Context Number of Predictions": int(r.get("Number of Predicted Boxes") or 0),
            "Radiology Context Mean IoU": rad_gt_mean,
            "Radiology Context Maximum IoU": rad_max,
            "Mean IoU Difference": rad_gt_mean - bare_gt_mean,
            "Maximum IoU Difference": rad_max - bare_max,
        })

    return overall, disease_rows, paired


def write_comparison_xlsx(
    out_path: Path,
    overall: List[dict],
    disease_rows: List[dict],
    paired: List[dict],
) -> None:
    wb = Workbook()

    ws1 = wb.active
    ws1.title = "overall_comparison"
    cols1 = [
        "Prompt Strategy", "Total Image-Disease Pairs", "Total GT Boxes",
        "Total Predicted Boxes", "No-Prediction Rate", "Mean IoU", "Median IoU",
        "Recall@0.1", "Recall@0.3", "Recall@0.5", "Mean Inference Time", "Errors",
    ]
    ws1.append(cols1)
    for row in overall:
        ws1.append([row.get(c) for c in cols1])

    ws2 = wb.create_sheet("disease_comparison")
    cols2 = [
        "Disease", "Prompt Strategy", "Number of Images", "Number of GT Boxes",
        "Number of Predicted Boxes", "No-Prediction Rate", "Mean IoU", "Median IoU",
        "Recall@0.1", "Recall@0.3", "Recall@0.5",
    ]
    ws2.append(cols2)
    for row in disease_rows:
        ws2.append([row.get(c) for c in cols2])

    ws3 = wb.create_sheet("paired_sample_comparison")
    cols3 = [
        "Image", "Disease",
        "Bare Label Prompt", "Bare Label Number of Predictions",
        "Bare Label Mean IoU", "Bare Label Maximum IoU",
        "Radiology Context Prompt", "Radiology Context Number of Predictions",
        "Radiology Context Mean IoU", "Radiology Context Maximum IoU",
        "Mean IoU Difference", "Maximum IoU Difference",
    ]
    ws3.append(cols3)
    for row in paired:
        ws3.append([row.get(c) for c in cols3])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def print_report(overall: List[dict], disease_rows: List[dict], paired: List[dict]) -> None:
    print("\n" + "=" * 78)
    print("20-PAIR PROMPT COMPARISON REPORT")
    print("=" * 78)
    by_strat = {r["Prompt Strategy"]: r for r in overall}
    for strat in ("bare_label", "radiology_context"):
        r = by_strat[strat]
        n_pairs = r["Total Image-Disease Pairs"]
        n_no = int(round(r["No-Prediction Rate"] * n_pairs))
        print(f"\n[{strat}]")
        print(f"  pairs: {n_pairs}")
        print(f"  samples with no prediction: {n_no} ({r['No-Prediction Rate']:.3f})")
        print(f"  mean IoU: {r['Mean IoU']:.4f}")
        print(f"  median IoU: {r['Median IoU']:.4f}")
        print(f"  Recall@0.1: {r['Recall@0.1']:.4f}")
        print(f"  Recall@0.3: {r['Recall@0.3']:.4f}")
        print(f"  Recall@0.5: {r['Recall@0.5']:.4f}")

    print("\nPer-disease:")
    diseases = sorted({r["Disease"] for r in disease_rows})
    for disease in diseases:
        print(f"  {disease}:")
        for strat in ("bare_label", "radiology_context"):
            row = next(
                d for d in disease_rows
                if d["Disease"] == disease and d["Prompt Strategy"] == strat
            )
            print(
                f"    {strat}: n={row['Number of Images']} "
                f"no_pred={row['No-Prediction Rate']:.2f} "
                f"meanIoU={row['Mean IoU']:.4f} "
                f"R@0.1={row['Recall@0.1']:.3f} "
                f"R@0.3={row['Recall@0.3']:.3f} "
                f"R@0.5={row['Recall@0.5']:.3f}"
            )

    # Consistency of improvement
    improved_mean = sum(1 for p in paired if p["Mean IoU Difference"] > 1e-9)
    worsened_mean = sum(1 for p in paired if p["Mean IoU Difference"] < -1e-9)
    tied_mean = len(paired) - improved_mean - worsened_mean
    improved_max = sum(1 for p in paired if p["Maximum IoU Difference"] > 1e-9)

    pneumonia = [p for p in paired if p["Disease"] == "Pneumonia"]
    pneum_improved = all(p["Mean IoU Difference"] > 1e-9 for p in pneumonia) if pneumonia else False
    non_pneum = [p for p in paired if p["Disease"] != "Pneumonia"]
    non_pneum_improved = sum(1 for p in non_pneum if p["Mean IoU Difference"] > 1e-9)

    print("\nDoes context prompting consistently improve results?")
    print(f"  pairs with Mean IoU improvement: {improved_mean}/{len(paired)}")
    print(f"  pairs with Mean IoU regression:  {worsened_mean}/{len(paired)}")
    print(f"  pairs tied:                      {tied_mean}/{len(paired)}")
    print(f"  pairs with Max IoU improvement:  {improved_max}/{len(paired)}")
    if pneumonia:
        print(
            f"  Pneumonia pairs improved (mean IoU): "
            f"{sum(1 for p in pneumonia if p['Mean IoU Difference'] > 1e-9)}/{len(pneumonia)}"
        )
    print(
        f"  Non-Pneumonia pairs improved (mean IoU): "
        f"{non_pneum_improved}/{len(non_pneum)}"
    )
    if improved_mean == len(paired):
        verdict = "YES — radiology_context improved Mean IoU on every pair."
    elif improved_mean > worsened_mean and non_pneum_improved > 0:
        verdict = (
            "PARTIAL — radiology_context helps on average / on multiple diseases, "
            "not only the pneumonia example."
        )
    elif pneum_improved and non_pneum_improved == 0:
        verdict = (
            "NO (narrow) — improvement appears limited to pneumonia-like cases; "
            "non-Pneumonia pairs did not improve."
        )
    elif improved_mean == 0:
        verdict = "NO — radiology_context did not improve Mean IoU on any pair."
    else:
        verdict = "MIXED — some pairs improve, some do not; not a consistent gain."
    print(f"  Verdict: {verdict}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bare-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "bare_label"),
    )
    p.add_argument(
        "--rad-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "radiology_context"),
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "results" / "chestxray8_prompt_comparison.xlsx"),
    )
    args = p.parse_args()

    bare = load_sample_rows(Path(args.bare_dir) / "intermediate_results.jsonl")
    rad = load_sample_rows(Path(args.rad_dir) / "intermediate_results.jsonl")
    overall, disease_rows, paired = build_comparison(bare, rad)
    out = Path(args.output)
    write_comparison_xlsx(out, overall, disease_rows, paired)
    print_report(overall, disease_rows, paired)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
