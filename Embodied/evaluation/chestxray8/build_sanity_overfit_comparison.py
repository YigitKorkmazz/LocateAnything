#!/usr/bin/env python3
"""Build results/sanity_overfit/chestxray8_overfit_capacity_comparison.xlsx

Reads existing evaluation / training artifacts only. Does not run training or eval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CHEST_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = REPO_ROOT / "results" / "sanity_overfit"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _pick_overall(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not summary:
        return None
    rows = summary.get("overall") or []
    for r in rows:
        if r.get("Note") == "not_evaluated":
            continue
        if r.get("Run") in ("lora_projector", "lora") or "Mean IoU" in r:
            return r
    return rows[0] if rows else None


def _final_loss_ma(train_dir: Path) -> Optional[float]:
    sel = _load_json(train_dir / "checkpoint_selection.json") or {}
    if sel.get("final_moving_average_loss") is not None:
        return float(sel["final_moving_average_loss"])
    hist = _load_json(train_dir / "loss_history.json") or []
    if not hist:
        return None
    # Prefer explicit loss_ma; else mean of last 20 raw losses.
    if "loss_ma" in hist[-1]:
        return float(hist[-1]["loss_ma"])
    window = hist[-20:]
    return float(sum(float(h["loss"]) for h in window) / len(window))


def _optimizer_steps(train_dir: Path) -> Optional[int]:
    sel = _load_json(train_dir / "checkpoint_selection.json") or {}
    if sel.get("global_step") is not None:
        return int(sel["global_step"])
    hist = _load_json(train_dir / "loss_history.json") or []
    if hist:
        return int(hist[-1].get("step", len(hist)))
    return None


def _checkpoint_path(train_dir: Path) -> Optional[str]:
    p = train_dir / "final_checkpoint.txt"
    if p.is_file():
        return p.read_text().strip().splitlines()[0]
    sel = _load_json(train_dir / "checkpoint_selection.json") or {}
    return sel.get("path")


def _eval_summary(eval_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_json(eval_dir / "finetuning_eval_summary.json")


def build_workbook(
    results_dir: Path,
    out_xlsx: Path,
) -> None:
    integrity = _load_json(results_dir / "split_integrity_report.json") or {}

    # Expected layout after manual runs:
    #   overfit_20/                 training run
    #   overfit_20/eval_on_20/      eval summary
    #   overfit_50/                 training run
    #   overfit_50/eval_on_50/
    #   overfit_50/eval_on_20/
    #   overfit_100/ ...
    experiments = {
        "train_20_eval_20": {
            "train_dir": results_dir / "overfit_20",
            "eval_dir": results_dir / "overfit_20" / "eval_on_20",
            "train_size_label": "20%",
            "eval_size_label": "20%",
        },
        "train_50_eval_20": {
            "train_dir": results_dir / "overfit_50",
            "eval_dir": results_dir / "overfit_50" / "eval_on_20",
            "train_size_label": "50%",
            "eval_size_label": "20%",
        },
        "train_100_eval_20": {
            "train_dir": results_dir / "overfit_100",
            "eval_dir": results_dir / "overfit_100" / "eval_on_20",
            "train_size_label": "100%",
            "eval_size_label": "20%",
        },
        "train_50_eval_50": {
            "train_dir": results_dir / "overfit_50",
            "eval_dir": results_dir / "overfit_50" / "eval_on_50",
            "train_size_label": "50%",
            "eval_size_label": "50%",
        },
        "train_100_eval_100": {
            "train_dir": results_dir / "overfit_100",
            "eval_dir": results_dir / "overfit_100" / "eval_on_100",
            "train_size_label": "100%",
            "eval_size_label": "100%",
        },
    }

    overall_rows: List[Dict[str, Any]] = []
    for name, cfg in experiments.items():
        summary = _eval_summary(cfg["eval_dir"])
        overall = _pick_overall(summary) or {}
        train_dir = cfg["train_dir"]
        overall_rows.append(
            {
                "Experiment": name,
                "Training dataset size": cfg["train_size_label"],
                "Evaluation dataset size": cfg["eval_size_label"],
                "Number of pairs": overall.get("Total Image-Disease Pairs"),
                "Mean IoU": overall.get("Mean IoU"),
                "Median IoU": overall.get("Median IoU"),
                "Recall@0.1": overall.get("Recall@0.1"),
                "Recall@0.3": overall.get("Recall@0.3"),
                "Recall@0.5": overall.get("Recall@0.5"),
                "Valid-output rate": overall.get("Valid Structured-Output Rate"),
                "Parsed-box rate": overall.get("Parsed-Box Rate"),
                "No-prediction rate": overall.get("No-Prediction Rate"),
                "Final moving-average training loss": _final_loss_ma(train_dir),
                "Number of optimizer steps": _optimizer_steps(train_dir),
                "Checkpoint path": _checkpoint_path(train_dir),
                "Eval summary path": str(cfg["eval_dir"] / "finetuning_eval_summary.json"),
                "Eval found": bool(summary),
            }
        )

    # Classwise for train_100_eval_100
    classwise_rows: List[Dict[str, Any]] = []
    s100 = _eval_summary(experiments["train_100_eval_100"]["eval_dir"])
    disease_src = (s100 or {}).get("disease") or []
    if disease_src:
        for row in disease_src:
            classwise_rows.append(
                {
                    "Disease": row.get("Disease"),
                    "Number of pairs": row.get("Number of Examples")
                    or row.get("Total Image-Disease Pairs"),
                    "Mean IoU": row.get("Mean IoU"),
                    "Median IoU": row.get("Median IoU"),
                    "Recall@0.1": row.get("Recall@0.1"),
                    "Recall@0.3": row.get("Recall@0.3"),
                    "Recall@0.5": row.get("Recall@0.5"),
                }
            )
    else:
        xlsx_cands = list(
            (results_dir / "overfit_100" / "eval_on_100").glob("*.xlsx")
        )
        disease_loaded = False
        for xp in xlsx_cands:
            try:
                df = pd.read_excel(xp, sheet_name="disease_comparison")
                for _, row in df.iterrows():
                    classwise_rows.append(
                        {
                            "Disease": row.get("Disease"),
                            "Number of pairs": row.get("Number of Examples")
                            or row.get("Total Image-Disease Pairs"),
                            "Mean IoU": row.get("Mean IoU"),
                            "Median IoU": row.get("Median IoU"),
                            "Recall@0.1": row.get("Recall@0.1"),
                            "Recall@0.3": row.get("Recall@0.3"),
                            "Recall@0.5": row.get("Recall@0.5"),
                        }
                    )
                disease_loaded = True
                break
            except Exception:
                continue
        if not disease_loaded:
            classwise_rows.append(
                {
                    "Disease": None,
                    "Number of pairs": None,
                    "Mean IoU": None,
                    "Median IoU": None,
                    "Recall@0.1": None,
                    "Recall@0.3": None,
                    "Recall@0.5": None,
                    "Note": "disease rows not found for train_100_eval_100",
                }
            )

    orig20_rows = []
    for name in ("train_20_eval_20", "train_50_eval_20", "train_100_eval_20"):
        row = next(r for r in overall_rows if r["Experiment"] == name)
        orig20_rows.append(
            {
                "Trained on": {
                    "train_20_eval_20": "20%",
                    "train_50_eval_20": "50%",
                    "train_100_eval_20": "100%",
                }[name],
                "Eval set": "original fixed 20%",
                "Number of pairs": row["Number of pairs"],
                "Mean IoU": row["Mean IoU"],
                "Median IoU": row["Median IoU"],
                "Recall@0.1": row["Recall@0.1"],
                "Recall@0.3": row["Recall@0.3"],
                "Recall@0.5": row["Recall@0.5"],
                "Valid-output rate": row["Valid-output rate"],
                "Parsed-box rate": row["Parsed-box rate"],
                "No-prediction rate": row["No-prediction rate"],
                "Checkpoint path": row["Checkpoint path"],
            }
        )

    train_cfg_rows = []
    for stage in ("overfit_20", "overfit_50", "overfit_100"):
        args = _load_json(results_dir / stage / "training_arguments.json") or {}
        train_cfg_rows.append({"stage": stage, **args})

    integrity_rows = [{"key": "raw_report", "value": json.dumps(integrity)}]
    if integrity:
        for k, v in (integrity.get("checksums") or {}).items():
            integrity_rows.append({"key": f"checksum.{k}", "value": v})
        for k, v in (integrity.get("paths") or {}).items():
            integrity_rows.append({"key": f"path.{k}", "value": v})
        for label, s in (integrity.get("summaries") or {}).items():
            integrity_rows.append(
                {
                    "key": f"summary.{label}",
                    "value": json.dumps(s),
                }
            )
        for k, v in (integrity.get("integrity") or {}).items():
            integrity_rows.append({"key": f"check.{k}", "value": v})

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(overall_rows).to_excel(
            writer, sheet_name="overall_progression", index=False
        )
        pd.DataFrame(classwise_rows).to_excel(
            writer, sheet_name="classwise_full_dataset", index=False
        )
        pd.DataFrame(orig20_rows).to_excel(
            writer, sheet_name="original_20_progression", index=False
        )
        pd.DataFrame(train_cfg_rows).to_excel(
            writer, sheet_name="training_configuration", index=False
        )
        pd.DataFrame(integrity_rows).to_excel(
            writer, sheet_name="split_integrity", index=False
        )

    print(f"Wrote {out_xlsx}")
    missing = [r["Experiment"] for r in overall_rows if not r["Eval found"]]
    if missing:
        print("WARNING: missing eval summaries for:", ", ".join(missing))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS))
    p.add_argument(
        "--output-xlsx",
        type=str,
        default=str(DEFAULT_RESULTS / "chestxray8_overfit_capacity_comparison.xlsx"),
    )
    args = p.parse_args()
    build_workbook(Path(args.results_dir), Path(args.output_xlsx))


if __name__ == "__main__":
    main()
