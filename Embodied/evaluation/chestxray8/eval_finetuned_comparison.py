#!/usr/bin/env python3
"""
Evaluate base / full-SFT / LoRA / LoRA+projector LocateAnything checkpoints on
the held-out ChestX-ray8 test split and write a comparison Excel workbook.

Does NOT touch results/full_radiology_context/.
Outputs under results/finetuning/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CHEST_DIR))

from eval_locateanything_bbox import (  # noqa: E402
    BOX_RE,
    PROMPT_STRATEGIES,
    build_final_user_query,
    compute_pair_metrics,
    convert_boxes_to_pixels,
    init_worker,
    parse_normalized_boxes,
    run_inference,
    to_jsonable_box,
)
from sft_common import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    PROMPT_STRATEGY,
    collect_reproducibility_info,
    read_jsonl,
    set_global_seed,
)

PINNED_MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("finetune_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(output_dir / "eval_finetuned.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _tensor_fingerprint(t) -> Dict[str, Any]:
    import torch

    x = t.detach().float().cpu().reshape(-1)
    n = min(4096, int(x.numel()))
    # Stable lightweight fingerprint for logging.
    payload = x[:n].numpy().tobytes()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "norm": float(x.norm().item()),
        "sum0": float(x[:n].sum().item()),
        "sha1_prefix": hashlib.sha1(payload).hexdigest()[:12],
    }


def _mlp1_state_fingerprints(module) -> Dict[str, Dict[str, Any]]:
    return {name: _tensor_fingerprint(param) for name, param in module.state_dict().items()}


def load_base_worker_pinned(
    base_model: str,
    device: str,
    revision: str = PINNED_MODEL_REVISION,
    logger: Optional[logging.Logger] = None,
):
    """Load LocateAnything at a pinned HF revision (worker defaults do not pin)."""
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from locateanything_worker import LocateAnythingWorker

    log = logger.info if logger is not None else print
    log(f"[load] base_model={base_model} revision={revision} device={device}")

    worker = LocateAnythingWorker.__new__(LocateAnythingWorker)
    worker.device = device
    worker.dtype = torch.bfloat16
    worker.use_batch_runtime = False
    worker.tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True, revision=revision
    )
    worker.processor = AutoProcessor.from_pretrained(
        base_model, trust_remote_code=True, revision=revision
    )
    worker.model = (
        AutoModel.from_pretrained(
            base_model,
            revision=revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    return worker


def _attach_lora_adapter(worker, adapter_dir: Path, logger: Optional[logging.Logger] = None):
    from peft import PeftModel

    log = logger.info if logger is not None else print
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.is_file():
        raise FileNotFoundError(f"adapter_config.json missing under {adapter_dir}")

    model = worker.model
    if hasattr(model, "language_model"):
        model.language_model = PeftModel.from_pretrained(
            model.language_model, str(adapter_dir)
        )
        model.use_llm_lora = True
    else:
        worker.model = PeftModel.from_pretrained(model, str(adapter_dir))
        worker.model.use_llm_lora = True

    lm = worker.model.language_model
    active = list(getattr(lm, "active_adapters", []) or [])
    n_lora = sum(1 for n, _ in lm.named_parameters() if "lora_" in n.lower())
    log(
        f"[load] LoRA adapter active={active} n_lora_param_tensors={n_lora} "
        f"use_llm_lora={getattr(worker.model, 'use_llm_lora', None)} path={adapter_dir}"
    )
    if not active and n_lora == 0:
        raise RuntimeError(f"LoRA adapter loaded from {adapter_dir} but no LoRA params found")
    return worker


def load_projector_into_mlp1(
    worker,
    projector_path: Path,
    logger: logging.Logger,
    device: str,
) -> Dict[str, Any]:
    """Strictly load mlp1.pt into worker.model.mlp1 with shape/key validation."""
    import torch

    projector_path = Path(projector_path)
    if not projector_path.is_file():
        raise FileNotFoundError(f"Projector checkpoint not found: {projector_path}")
    if not hasattr(worker.model, "mlp1"):
        raise RuntimeError("Loaded model has no mlp1 module")

    mlp1 = worker.model.mlp1
    expected = mlp1.state_dict()
    before = _mlp1_state_fingerprints(mlp1)

    logger.info("[load] projector checkpoint path=%s", projector_path)
    state = torch.load(str(projector_path), map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(
            f"Projector checkpoint must be a state_dict dict, got {type(state)}"
        )

    expected_keys = set(expected.keys())
    loaded_keys = set(state.keys())
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Projector state_dict key mismatch for mlp1.\n"
            f"  missing ({len(missing)}): {missing}\n"
            f"  unexpected ({len(unexpected)}): {unexpected}"
        )

    # Strict shape checks before load.
    shape_errors = []
    for key in sorted(expected_keys):
        exp_t = expected[key]
        got_t = state[key]
        if tuple(exp_t.shape) != tuple(got_t.shape):
            shape_errors.append(
                f"{key}: expected {tuple(exp_t.shape)}, got {tuple(got_t.shape)}"
            )
    if shape_errors:
        raise RuntimeError(
            "Projector tensor shape mismatch:\n  " + "\n  ".join(shape_errors)
        )

    incompatible = mlp1.load_state_dict(state, strict=True)
    # strict=True raises on mismatch; keep this for older torch behaviors.
    if getattr(incompatible, "missing_keys", None) or getattr(
        incompatible, "unexpected_keys", None
    ):
        raise RuntimeError(f"mlp1.load_state_dict incompatible: {incompatible}")

    # Move tensors to model device/dtype.
    mlp1.to(device=next(worker.model.parameters()).device)
    after = _mlp1_state_fingerprints(mlp1)

    changed = []
    unchanged = []
    for key in sorted(expected_keys):
        if (
            before[key]["norm"] != after[key]["norm"]
            or before[key]["sha1_prefix"] != after[key]["sha1_prefix"]
            or before[key]["sum0"] != after[key]["sum0"]
        ):
            changed.append(key)
        else:
            unchanged.append(key)

    if not changed:
        raise RuntimeError(
            "Projector load appeared successful but mlp1 weights are identical to "
            "the base model (no checksum/norm change). Refusing to continue."
        )

    report = {
        "projector_path": str(projector_path),
        "n_loaded_tensors": len(expected_keys),
        "parameter_names": sorted(expected_keys),
        "fingerprints_before": before,
        "fingerprints_after": after,
        "n_changed_vs_base": len(changed),
        "changed_keys": changed,
        "unchanged_keys": unchanged,
        "mlp1_differs_from_base": True,
    }
    logger.info(
        "[load] loaded %d projector tensors into mlp1; changed_vs_base=%d/%d",
        report["n_loaded_tensors"],
        report["n_changed_vs_base"],
        report["n_loaded_tensors"],
    )
    for name in report["parameter_names"]:
        b = before[name]
        a = after[name]
        logger.info(
            "  mlp1.%s before(norm=%.6f,sha1=%s) after(norm=%.6f,sha1=%s)",
            name,
            b["norm"],
            b["sha1_prefix"],
            a["norm"],
            a["sha1_prefix"],
        )
    return report


def load_lora_worker(
    base_model: str,
    adapter_dir: Path,
    device: str,
    revision: str = PINNED_MODEL_REVISION,
    logger: Optional[logging.Logger] = None,
):
    """Load base LocateAnything and attach a PEFT adapter (LoRA-only)."""
    worker = load_base_worker_pinned(
        base_model, device=device, revision=revision, logger=logger
    )
    worker = _attach_lora_adapter(worker, Path(adapter_dir), logger=logger)
    worker.model.eval()
    return worker


def load_lora_projector_worker(
    base_model: str,
    adapter_dir: Path,
    projector_path: Path,
    device: str,
    revision: str = PINNED_MODEL_REVISION,
    logger: Optional[logging.Logger] = None,
):
    """Load base + LoRA adapter + fully fine-tuned mlp1 projector."""
    if logger is None:
        raise ValueError("logger is required for lora_projector loading")

    worker = load_base_worker_pinned(
        base_model, device=device, revision=revision, logger=logger
    )
    worker = _attach_lora_adapter(worker, Path(adapter_dir), logger=logger)
    projector_report = load_projector_into_mlp1(
        worker, Path(projector_path), logger=logger, device=device
    )

    lm = worker.model.language_model
    active = list(getattr(lm, "active_adapters", []) or [])
    logger.info(
        "[load] lora_projector ready: active_adapters=%s use_llm_lora=%s "
        "mlp1_differs_from_base=%s",
        active,
        getattr(worker.model, "use_llm_lora", None),
        projector_report["mlp1_differs_from_base"],
    )
    if not getattr(worker.model, "use_llm_lora", False):
        raise RuntimeError("LoRA adapter is not marked active after load")

    worker.model.eval()
    worker._projector_load_report = projector_report  # type: ignore[attr-defined]
    return worker


def resolve_adapter_dir(lora_checkpoint: Path) -> Path:
    """Accept either .../checkpoint-N or .../checkpoint-N/adapter."""
    path = Path(lora_checkpoint)
    if (path / "adapter_config.json").is_file():
        return path
    if (path / "adapter" / "adapter_config.json").is_file():
        return path / "adapter"
    raise FileNotFoundError(
        f"Could not find adapter_config.json under {path} or {path / 'adapter'}"
    )


def resolve_model_loader(
    run_name: str,
    model_path: str,
    base_model: str,
    device: str,
    logger: logging.Logger,
    projector_checkpoint: Optional[str] = None,
    revision: str = PINNED_MODEL_REVISION,
):
    if run_name == "lora_projector":
        if not projector_checkpoint:
            raise ValueError("lora_projector run requires --projector-checkpoint")
        adapter_dir = resolve_adapter_dir(Path(model_path))
        return load_lora_projector_worker(
            base_model=base_model,
            adapter_dir=adapter_dir,
            projector_path=Path(projector_checkpoint),
            device=device,
            revision=revision,
            logger=logger,
        )

    path = Path(model_path)
    adapter = path / "adapter"
    if run_name == "lora" or adapter.is_dir() or (path / "adapter_config.json").exists():
        adapter_dir = resolve_adapter_dir(path)
        return load_lora_worker(
            base_model, adapter_dir, device, revision=revision, logger=logger
        )
    return init_worker(model_path, device)


def summarize_rows(sample_rows: List[dict], box_rows: List[dict], run_name: str) -> Dict[str, Any]:
    ok_rows = [r for r in sample_rows if r.get("Status") == "ok"]
    times = [float(r["Inference Time"]) for r in ok_rows if r.get("Inference Time") is not None]
    all_box_ious = [float(b["IoU"]) for b in box_rows]
    n_no_pred = sum(1 for r in sample_rows if int(r.get("Number of Predicted Boxes") or 0) == 0)
    n_valid_struct = sum(1 for r in sample_rows if r.get("Valid Structured Output"))
    n_parsed = sum(1 for r in sample_rows if int(r.get("Number of Predicted Boxes") or 0) > 0)

    def overall_recall(thresh: float) -> float:
        if not box_rows:
            return 0.0
        return sum(1 for b in box_rows if float(b["IoU"]) >= thresh) / len(box_rows)

    return {
        "Run": run_name,
        "Total Images": len({r["Image"] for r in sample_rows}),
        "Total Image-Disease Pairs": len(sample_rows),
        "Total GT Boxes": sum(int(r["Number of GT Boxes"]) for r in sample_rows),
        "Total Predicted Boxes": sum(int(r.get("Number of Predicted Boxes") or 0) for r in sample_rows),
        "Valid Structured-Output Rate": (
            n_valid_struct / len(sample_rows) if sample_rows else 0.0
        ),
        "Parsed-Box Rate": n_parsed / len(sample_rows) if sample_rows else 0.0,
        "No-Prediction Rate": n_no_pred / len(sample_rows) if sample_rows else 0.0,
        "Mean IoU": float(np.mean(all_box_ious)) if all_box_ious else 0.0,
        "Median IoU": float(np.median(all_box_ious)) if all_box_ious else 0.0,
        "Recall@0.1": overall_recall(0.1),
        "Recall@0.3": overall_recall(0.3),
        "Recall@0.5": overall_recall(0.5),
        "Mean Inference Time": float(np.mean(times)) if times else 0.0,
        "Errors": sum(1 for r in sample_rows if r.get("Status") != "ok"),
    }


def per_disease_rows(sample_rows: List[dict], box_rows: List[dict], run_name: str) -> List[dict]:
    by_disease: Dict[str, List[dict]] = defaultdict(list)
    boxes_by_disease: Dict[str, List[dict]] = defaultdict(list)
    for r in sample_rows:
        by_disease[r["Disease"]].append(r)
    for b in box_rows:
        boxes_by_disease[b["Disease"]].append(b)
    out = []
    for disease in sorted(by_disease):
        rows = by_disease[disease]
        brows = boxes_by_disease.get(disease, [])
        s = summarize_rows(rows, brows, run_name)
        s["Disease"] = disease
        s["Number of Examples"] = s["Total Image-Disease Pairs"]
        out.append(s)
    return out


def evaluate_split(
    worker,
    pairs: Sequence[dict],
    run_name: str,
    prompt_strategy: str,
    logger: logging.Logger,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    if limit is not None:
        pairs = list(pairs)[:limit]
    sample_rows: List[dict] = []
    box_rows: List[dict] = []

    for pair in tqdm(pairs, desc=f"eval:{run_name}"):
        image_name = pair["image_index"]
        disease = pair["disease"]
        phrase, final_query = build_final_user_query(disease, prompt_strategy)
        # Always use the active strategy (do not prefer stale jsonl prompt text).
        gt_boxes = pair["gt_boxes_xyxy_px"]
        width = int(pair["image_width"])
        height = int(pair["image_height"])
        status = "ok"
        raw_answer = ""
        elapsed = None
        pred_boxes: List[List[float]] = []
        err = ""
        try:
            image = Image.open(pair["image_path"]).convert("RGB")
            width, height = image.size
            raw_answer, elapsed = run_inference(
                worker,
                image,
                phrase,
                prompt_strategy=prompt_strategy,
                final_query=final_query,
            )
            norm = parse_normalized_boxes(raw_answer)
            pred_boxes = convert_boxes_to_pixels(norm, width, height)
        except Exception as e:  # noqa: BLE001
            status = "error"
            err = str(e)
            logger.exception("Failed %s %s", image_name, disease)

        metrics = compute_pair_metrics(gt_boxes, pred_boxes)
        valid_structured = bool(BOX_RE.search(raw_answer or ""))
        sample_row = {
            "Run": run_name,
            "Image": image_name,
            "Disease": disease,
            "Patient ID": pair.get("patient_id"),
            "Prompt Strategy": prompt_strategy,
            "User Query": final_query,
            "Number of GT Boxes": len(gt_boxes),
            "Number of Predicted Boxes": len(pred_boxes),
            "Valid Structured Output": valid_structured,
            "Mean Matched IoU": metrics["mean_matched_iou"],
            "Recall@0.1": metrics["recall_0_1"],
            "Recall@0.3": metrics["recall_0_3"],
            "Recall@0.5": metrics["recall_0_5"],
            "Inference Time": elapsed,
            "Status": status,
            "Error": err,
            "Raw Answer": raw_answer,
            "GT Boxes": json.dumps([to_jsonable_box(b) for b in gt_boxes]),
            "Pred Boxes": json.dumps([to_jsonable_box(b) for b in pred_boxes]),
        }
        sample_rows.append(sample_row)
        for gt_idx, iou in enumerate(metrics["gt_ious"]):
            box_rows.append(
                {
                    "Run": run_name,
                    "Image": image_name,
                    "Disease": disease,
                    "GT Index": gt_idx,
                    "IoU": float(iou),
                }
            )

    overall = summarize_rows(sample_rows, box_rows, run_name)
    disease = per_disease_rows(sample_rows, box_rows, run_name)
    return {
        "overall": overall,
        "disease": disease,
        "samples": sample_rows,
        "boxes": box_rows,
    }


def write_comparison_xlsx(
    path: Path,
    overall_rows: List[dict],
    disease_rows: List[dict],
    sample_rows: List[dict],
    training_config_rows: List[dict],
    parameter_rows: List[dict],
) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(overall_rows).to_excel(writer, sheet_name="overall_comparison", index=False)
        pd.DataFrame(disease_rows).to_excel(writer, sheet_name="disease_comparison", index=False)
        sample_df = pd.DataFrame(sample_rows)
        if "Raw Answer" in sample_df.columns:
            sample_df["Raw Answer"] = sample_df["Raw Answer"].astype(str).str.slice(0, 500)
        sample_df.to_excel(writer, sheet_name="sample_results", index=False)
        pd.DataFrame(training_config_rows).to_excel(
            writer, sheet_name="training_configuration", index=False
        )
        pd.DataFrame(parameter_rows).to_excel(
            writer, sheet_name="parameter_summary", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-file",
        type=str,
        default=str(CHEST_DIR / "splits" / f"test_pairs_seed{DEFAULT_SEED}.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt-strategy", type=str, default=PROMPT_STRATEGY)
    parser.add_argument("--base-model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-revision",
        type=str,
        default=PINNED_MODEL_REVISION,
        help="Pinned HF revision for base / LoRA / projector loads.",
    )
    parser.add_argument(
        "--full-sft-checkpoint",
        type=str,
        default=None,
        help="Path to full-SFT checkpoint dir (optional if training did not fit).",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default=None,
        help="Path to LoRA checkpoint dir (checkpoint-N or checkpoint-N/adapter).",
    )
    parser.add_argument(
        "--projector-checkpoint",
        type=str,
        default=None,
        help="Path to mlp1.pt for lora_projector evaluation. When set, the "
        "fine-tuned run is labeled lora_projector and requires --lora-checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "finetuning"),
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default=str(
            REPO_ROOT / "results" / "finetuning" / "chestxray8_finetuning_comparison.xlsx"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Debug: evaluate first N test pairs.")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--omit-missing-run-placeholders",
        action="store_true",
        help="Do not insert empty base/full_sft/lora placeholder rows into the Excel.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--full-sft-config",
        type=str,
        default=None,
        help="Optional training_arguments.json for full SFT.",
    )
    parser.add_argument(
        "--lora-config",
        type=str,
        default=None,
        help="Optional training_arguments.json / reports for LoRA or lora_projector.",
    )
    args = parser.parse_args()
    set_global_seed(args.seed)

    output_dir = Path(args.output_dir)
    logger = setup_logger(output_dir)
    pairs = read_jsonl(Path(args.split_file))
    logger.info("Loaded %d held-out test pairs from %s", len(pairs), args.split_file)
    logger.info("Model revision pin: %s", args.model_revision)
    if args.prompt_strategy not in PROMPT_STRATEGIES:
        raise ValueError(args.prompt_strategy)

    if args.projector_checkpoint and not args.lora_checkpoint:
        raise ValueError("--projector-checkpoint requires --lora-checkpoint")

    runs: List[Tuple[str, str]] = []
    if not args.skip_base:
        runs.append(("base_zero_shot", args.base_model))
    if args.full_sft_checkpoint:
        runs.append(("full_sft", args.full_sft_checkpoint))
    if args.lora_checkpoint and args.projector_checkpoint:
        runs.append(("lora_projector", args.lora_checkpoint))
    elif args.lora_checkpoint:
        runs.append(("lora", args.lora_checkpoint))
    if not runs:
        raise RuntimeError("No runs specified")

    overall_rows: List[dict] = []
    disease_rows: List[dict] = []
    sample_rows: List[dict] = []
    training_config_rows: List[dict] = []
    parameter_rows: List[dict] = []
    projector_load_reports: Dict[str, Any] = {}

    # Attach configs if present
    config_labels = ["base_zero_shot", "full_sft", "lora", "lora_projector"]
    for label in config_labels:
        row: Dict[str, Any] = {"Run": label}
        cfg_path = None
        if label == "full_sft":
            cfg_path = args.full_sft_config
        elif label in ("lora", "lora_projector"):
            cfg_path = args.lora_config
        if cfg_path and Path(cfg_path).is_file():
            row.update(json.loads(Path(cfg_path).read_text()))
        if label == "lora_projector" and args.projector_checkpoint:
            row["projector_checkpoint"] = args.projector_checkpoint
            row["lora_checkpoint"] = args.lora_checkpoint
        training_config_rows.append(row)

    for run_name, model_path in runs:
        logger.info("Starting run=%s model=%s", run_name, model_path)
        worker = resolve_model_loader(
            run_name,
            model_path,
            args.base_model,
            args.device,
            logger=logger,
            projector_checkpoint=args.projector_checkpoint
            if run_name == "lora_projector"
            else None,
            revision=args.model_revision,
        )
        if run_name == "lora_projector":
            projector_load_reports[run_name] = getattr(
                worker, "_projector_load_report", None
            )
        result = evaluate_split(
            worker,
            pairs,
            run_name=run_name,
            prompt_strategy=args.prompt_strategy,
            logger=logger,
            limit=args.limit,
        )
        overall_rows.append(result["overall"])
        disease_rows.extend(result["disease"])
        sample_rows.extend(result["samples"])
        # Free GPU before next model
        del worker
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

        # Parameter summary files if available near checkpoint
        ckpt = Path(model_path)
        for cand in (
            ckpt / "parameter_summary.json",
            ckpt.parent / "parameter_summary.json",
            ckpt / "trainable_parameter_report.json",
            ckpt.parent / "trainable_parameter_report.json",
        ):
            if cand.is_file():
                payload = json.loads(cand.read_text())
                payload = {"Run": run_name, **payload}
                parameter_rows.append(payload)
                break
        else:
            parameter_rows.append({"Run": run_name, "note": "no parameter report found"})

        if run_name == "lora_projector" and projector_load_reports.get(run_name):
            parameter_rows.append(
                {
                    "Run": run_name,
                    "projector_load": projector_load_reports[run_name],
                }
            )

    # Ensure placeholder rows exist for missing experiments
    if not args.omit_missing_run_placeholders:
        present = {r["Run"] for r in overall_rows}
        for required in ("base_zero_shot", "full_sft", "lora", "lora_projector"):
            if required not in present:
                overall_rows.append(
                    {
                        "Run": required,
                        "Note": "not_evaluated",
                        "Mean IoU": None,
                    }
                )

    excel_path = Path(args.excel_path)
    write_comparison_xlsx(
        excel_path,
        overall_rows,
        disease_rows,
        sample_rows,
        training_config_rows,
        parameter_rows,
    )
    summary_path = output_dir / "finetuning_eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "split_file": args.split_file,
                "overall": overall_rows,
                "disease": disease_rows,
                "projector_load_reports": projector_load_reports,
                "reproducibility": collect_reproducibility_info(args.base_model),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    logger.info("Wrote %s", excel_path)
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
