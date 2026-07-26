#!/usr/bin/env python3
"""Audit joint-overfit checkpoints and compare fixed-20% evaluation predictions.

This script is read-only with respect to model/evaluation artifacts. It writes
CSV/XLSX/JSON audit outputs under --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch
from safetensors.torch import load_file


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_state_diff(
    left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    left_keys = set(left)
    right_keys = set(right)
    common = sorted(left_keys & right_keys)
    n_differ = 0
    max_abs = 0.0
    sum_abs = 0.0
    n_elements = 0
    differing_names: List[str] = []
    for name in common:
        a = left[name].detach().cpu()
        b = right[name].detach().cpu()
        if tuple(a.shape) != tuple(b.shape):
            differing_names.append(name)
            n_differ += 1
            continue
        if not torch.equal(a, b):
            n_differ += 1
            differing_names.append(name)
        delta = (a.float() - b.float()).abs()
        if delta.numel():
            max_abs = max(max_abs, float(delta.max().item()))
            sum_abs += float(delta.double().sum().item())
            n_elements += delta.numel()
    return {
        "left_tensor_count": len(left_keys),
        "right_tensor_count": len(right_keys),
        "common_tensor_count": len(common),
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "differing_tensor_count": n_differ,
        "identical_tensor_count": len(common) - n_differ,
        "max_abs_parameter_difference": max_abs,
        "mean_abs_parameter_difference": sum_abs / n_elements
        if n_elements
        else 0.0,
        "compared_parameter_elements": n_elements,
        "differing_tensor_names": differing_names,
    }


def load_projector(path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected projector state dict at {path}, got {type(state)}")
    return state


def parse_boxes(value: Any) -> List[List[float]]:
    if pd.isna(value):
        return []
    boxes = json.loads(str(value))
    return [[float(v) for v in box] for box in boxes]


def flatten_boxes(boxes: List[List[float]]) -> List[float]:
    return [v for box in boxes for v in box]


def compare_box_lists(
    left: List[List[float]], right: List[List[float]], atol: float
) -> bool:
    if len(left) != len(right):
        return False
    if any(len(a) != len(b) for a, b in zip(left, right)):
        return False
    return all(
        abs(a - b) <= atol
        for left_box, right_box in zip(left, right)
        for a, b in zip(left_box, right_box)
    )


def coordinate_differences(
    left: List[List[float]], right: List[List[float]]
) -> List[float]:
    if len(left) != len(right):
        return []
    if any(len(a) != len(b) for a, b in zip(left, right)):
        return []
    return [
        abs(a - b)
        for left_box, right_box in zip(left, right)
        for a, b in zip(left_box, right_box)
    ]


def load_samples(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="sample_results")
    required = {
        "Image",
        "Disease",
        "User Query",
        "GT Boxes",
        "Pred Boxes",
        "Raw Answer",
        "Status",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing sample columns {sorted(missing)}")
    df = df.copy()
    df["stable_key"] = (
        df["Image"].astype(str)
        + "||"
        + df["Disease"].astype(str)
        + "||"
        + df["User Query"].astype(str)
    )
    if df["stable_key"].duplicated().any():
        raise RuntimeError(f"{path}: stable key is not unique")
    df = df.set_index("stable_key", drop=False)
    rename = {
        "Pred Boxes": f"pred_boxes_{label}",
        "GT Boxes": f"gt_boxes_{label}",
        "Raw Answer": f"raw_answer_{label}",
        "Status": f"status_{label}",
    }
    return df.rename(columns=rename)


def compare_predictions(
    workbooks: Dict[str, Path],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    labels = list(workbooks)
    frames = {label: load_samples(path, label) for label, path in workbooks.items()}
    base_label = labels[0]
    base = frames[base_label]
    key_sets = {label: set(df.index) for label, df in frames.items()}
    same_keys = all(keys == key_sets[base_label] for keys in key_sets.values())
    same_order = all(
        list(df.index) == list(base.index) for label, df in frames.items()
    )

    merged = pd.DataFrame(index=base.index)
    merged["stable_key"] = base.index
    merged["Image"] = base["Image"]
    merged["Disease"] = base["Disease"]
    merged["User Query"] = base["User Query"]
    for label, df in frames.items():
        merged[f"gt_boxes_{label}"] = df.reindex(base.index)[f"gt_boxes_{label}"]
        merged[f"pred_boxes_{label}"] = df.reindex(base.index)[
            f"pred_boxes_{label}"
        ]
        merged[f"raw_answer_{label}"] = df.reindex(base.index)[
            f"raw_answer_{label}"
        ]
        merged[f"status_{label}"] = df.reindex(base.index)[f"status_{label}"]

    pairwise: List[Dict[str, Any]] = []
    all_coordinate_deltas: List[float] = []
    for i, left_label in enumerate(labels):
        for right_label in labels[i + 1 :]:
            exact_count = 0
            close_1e6_count = 0
            close_1e4_count = 0
            text_diff_same_box = 0
            deltas: List[float] = []
            prefix = f"{left_label}_vs_{right_label}"
            exact_values: List[bool] = []
            close_1e6_values: List[bool] = []
            close_1e4_values: List[bool] = []
            max_delta_values: List[float | None] = []
            mean_delta_values: List[float | None] = []
            text_diff_values: List[bool] = []
            for _, row in merged.iterrows():
                left_boxes = parse_boxes(row[f"pred_boxes_{left_label}"])
                right_boxes = parse_boxes(row[f"pred_boxes_{right_label}"])
                exact = left_boxes == right_boxes
                close_1e6 = compare_box_lists(left_boxes, right_boxes, 1e-6)
                close_1e4 = compare_box_lists(left_boxes, right_boxes, 1e-4)
                row_deltas = coordinate_differences(left_boxes, right_boxes)
                text_diff = (
                    str(row[f"raw_answer_{left_label}"])
                    != str(row[f"raw_answer_{right_label}"])
                )
                text_diff_same = text_diff and exact
                exact_count += int(exact)
                close_1e6_count += int(close_1e6)
                close_1e4_count += int(close_1e4)
                text_diff_same_box += int(text_diff_same)
                deltas.extend(row_deltas)
                exact_values.append(exact)
                close_1e6_values.append(close_1e6)
                close_1e4_values.append(close_1e4)
                max_delta_values.append(max(row_deltas) if row_deltas else None)
                mean_delta_values.append(
                    sum(row_deltas) / len(row_deltas) if row_deltas else None
                )
                text_diff_values.append(text_diff_same)
            n = len(merged)
            merged[f"{prefix}_exact"] = exact_values
            merged[f"{prefix}_close_atol_1e-6"] = close_1e6_values
            merged[f"{prefix}_close_atol_1e-4"] = close_1e4_values
            merged[f"{prefix}_max_abs_coordinate_diff"] = max_delta_values
            merged[f"{prefix}_mean_abs_coordinate_diff"] = mean_delta_values
            merged[f"{prefix}_text_diff_same_box"] = text_diff_values
            pairwise.append(
                {
                    "comparison": prefix,
                    "row_count": n,
                    "exact_identical_box_rows": exact_count,
                    "exact_identical_box_percent": 100.0 * exact_count / n,
                    "close_atol_1e-6_rows": close_1e6_count,
                    "close_atol_1e-6_percent": 100.0 * close_1e6_count / n,
                    "close_atol_1e-4_rows": close_1e4_count,
                    "close_atol_1e-4_percent": 100.0 * close_1e4_count / n,
                    "max_abs_coordinate_difference": max(deltas) if deltas else 0.0,
                    "mean_abs_coordinate_difference": sum(deltas) / len(deltas)
                    if deltas
                    else 0.0,
                    "different_text_same_parsed_box_rows": text_diff_same_box,
                }
            )
            all_coordinate_deltas.extend(deltas)

    gt_identical = True
    for label in labels[1:]:
        gt_identical = gt_identical and all(
            parse_boxes(a) == parse_boxes(b)
            for a, b in zip(
                merged[f"gt_boxes_{base_label}"], merged[f"gt_boxes_{label}"]
            )
        )
    integrity = {
        "stable_key_sets_identical": same_keys,
        "row_order_identical": same_order,
        "ground_truth_boxes_identical": gt_identical,
        "row_counts": {label: len(df) for label, df in frames.items()},
        "all_max_abs_coordinate_difference": max(all_coordinate_deltas)
        if all_coordinate_deltas
        else 0.0,
        "all_mean_abs_coordinate_difference": (
            sum(all_coordinate_deltas) / len(all_coordinate_deltas)
            if all_coordinate_deltas
            else 0.0
        ),
    }
    return merged.reset_index(drop=True), pairwise, integrity


def hash_artifacts(paths: Iterable[Tuple[str, str, Path]]) -> List[Dict[str, Any]]:
    rows = []
    for run, kind, path in paths:
        rows.append(
            {
                "run": run,
                "artifact_kind": kind,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/sanity_overfit_joint"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sanity_overfit_joint/audit"),
    )
    p.add_argument(
        "--fresh-rerun-xlsx",
        type=Path,
        default=None,
        help="Optional fresh 100%-checkpoint eval workbook to compare to prior output.",
    )
    p.add_argument(
        "--fresh-rerun-provenance",
        type=Path,
        default=None,
        help="Optional provenance JSON associated with --fresh-rerun-xlsx.",
    )
    args = p.parse_args()
    root = args.results_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    runs = {
        "20": {
            "checkpoint": root / "overfit_20/checkpoint-2000",
            "xlsx": root / "overfit_20/eval_on_20_gpu2/eval_on_20_gpu2.xlsx",
            "json": root / "overfit_20/eval_on_20_gpu2/finetuning_eval_summary.json",
        },
        "50": {
            "checkpoint": root / "overfit_50/checkpoint-3000",
            "xlsx": root / "overfit_50/eval_on_20/eval_on_20.xlsx",
            "json": root / "overfit_50/eval_on_20/finetuning_eval_summary.json",
        },
        "100": {
            "checkpoint": root / "overfit_100/checkpoint-7500",
            "xlsx": root / "overfit_100/eval_on_20/eval_on_20.xlsx",
            "json": root / "overfit_100/eval_on_20/finetuning_eval_summary.json",
        },
    }
    for run, cfg in runs.items():
        cfg["adapter_model"] = cfg["checkpoint"] / "adapter/adapter_model.safetensors"
        cfg["adapter_config"] = cfg["checkpoint"] / "adapter/adapter_config.json"
        cfg["adapter_readme"] = cfg["checkpoint"] / "adapter/README.md"
        cfg["projector"] = cfg["checkpoint"] / "mlp1.pt"
        for key, path in cfg.items():
            if key == "checkpoint":
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Run {run}: missing {key}: {path}")

    artifacts: List[Tuple[str, str, Path]] = []
    for run, cfg in runs.items():
        for kind in (
            "adapter_model",
            "adapter_config",
            "adapter_readme",
            "projector",
            "xlsx",
            "json",
        ):
            artifacts.append((run, kind, cfg[kind]))
    hash_rows = hash_artifacts(artifacts)

    row_comparison, pairwise_predictions, dataset_integrity = compare_predictions(
        {run: cfg["xlsx"] for run, cfg in runs.items()}
    )
    fresh_rerun: Dict[str, Any] | None = None
    if args.fresh_rerun_xlsx:
        if not args.fresh_rerun_xlsx.is_file():
            raise FileNotFoundError(args.fresh_rerun_xlsx)
        fresh_rows, fresh_pairs, fresh_integrity = compare_predictions(
            {
                "prior_100": runs["100"]["xlsx"],
                "fresh_100": args.fresh_rerun_xlsx,
            }
        )
        fresh_rerun = {
            "prediction_comparison": fresh_pairs[0],
            "dataset_integrity": fresh_integrity,
            "prior_xlsx_sha256": sha256_file(runs["100"]["xlsx"]),
            "fresh_xlsx_sha256": sha256_file(args.fresh_rerun_xlsx),
            "fresh_xlsx_path": str(args.fresh_rerun_xlsx),
        }
        if args.fresh_rerun_provenance:
            fresh_rerun["provenance"] = json.loads(
                args.fresh_rerun_provenance.read_text()
            )

    adapter_states = {
        run: load_file(str(cfg["adapter_model"]), device="cpu")
        for run, cfg in runs.items()
    }
    projector_states = {
        run: load_projector(cfg["projector"]) for run, cfg in runs.items()
    }
    parameter_rows = []
    labels = list(runs)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            parameter_rows.append(
                {
                    "kind": "LoRA adapter",
                    "comparison": f"{left}_vs_{right}",
                    **tensor_state_diff(adapter_states[left], adapter_states[right]),
                }
            )
            parameter_rows.append(
                {
                    "kind": "mlp1 projector",
                    "comparison": f"{left}_vs_{right}",
                    **tensor_state_diff(
                        projector_states[left], projector_states[right]
                    ),
                }
            )

    hash_df = pd.DataFrame(hash_rows)
    pred_df = pd.DataFrame(pairwise_predictions)
    param_df = pd.DataFrame(parameter_rows)
    integrity_df = pd.DataFrame(
        [{"check": k, "value": json.dumps(v) if isinstance(v, dict) else v}
         for k, v in dataset_integrity.items()]
    )
    row_comparison.to_csv(out / "row_level_prediction_comparison.csv", index=False)
    with pd.ExcelWriter(
        out / "joint_overfit_evaluation_audit.xlsx", engine="openpyxl"
    ) as writer:
        pred_df.to_excel(writer, sheet_name="prediction_summary", index=False)
        row_comparison.to_excel(writer, sheet_name="row_level_comparison", index=False)
        hash_df.to_excel(writer, sheet_name="artifact_hashes", index=False)
        param_df.to_excel(writer, sheet_name="parameter_differences", index=False)
        integrity_df.to_excel(writer, sheet_name="dataset_integrity", index=False)
        if fresh_rerun is not None:
            pd.DataFrame(
                [
                    {
                        "key": key,
                        "value": json.dumps(value)
                        if isinstance(value, (dict, list))
                        else value,
                    }
                    for key, value in fresh_rerun.items()
                ]
            ).to_excel(writer, sheet_name="fresh_rerun", index=False)

    report = {
        "runs": {
            run: {key: str(value) for key, value in cfg.items()}
            for run, cfg in runs.items()
        },
        "artifact_hashes": hash_rows,
        "prediction_comparisons": pairwise_predictions,
        "dataset_integrity": dataset_integrity,
        "parameter_comparisons": parameter_rows,
        "fresh_rerun": fresh_rerun,
        "outputs": {
            "csv": str(out / "row_level_prediction_comparison.csv"),
            "xlsx": str(out / "joint_overfit_evaluation_audit.xlsx"),
        },
    }
    (out / "audit_data.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
