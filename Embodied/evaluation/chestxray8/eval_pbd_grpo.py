#!/usr/bin/env python3
"""Evaluate RL-native PBD and ordinary LocateAnything decoding modes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from eval_locateanything_bbox import box_iou  # noqa: E402
from rl.rewards import (  # noqa: E402
    COMPLETION_PARSERS,
    is_valid_geometry,
    resolve_parser_name,
)
from rl.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    append_jsonl,
    assert_new_output_dir,
    build_policy,
    generate_rollout_group,
    load_resolved_config,
    load_verified_pairs,
    tokenize_rl_pair,
    write_json,
)
from train_chestxray8_sft import load_lora_adapter_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_checkpoint(model, checkpoint: Path, device: torch.device) -> None:
    load_lora_adapter_weights(model, checkpoint / "adapter")
    model.mlp1.load_state_dict(
        torch.load(checkpoint / "mlp1.pt", map_location=device)
    )


def _score_completion(
    completion: str,
    pair: Dict[str, Any],
    *,
    mode: str,
    rollout_index: int,
    parser_name: str = "chain_of_box",
) -> Dict[str, Any]:
    parsed = COMPLETION_PARSERS[parser_name](completion)
    valid = bool(
        parsed.format_valid
        and parsed.final_box_norm_1000 is not None
        and is_valid_geometry(parsed.final_box_norm_1000)
    )
    iou = 0.0
    if valid:
        iou = max(
            (
                box_iou(parsed.final_box_norm_1000, gt)
                for gt in pair["gt_boxes_norm_1000"]
            ),
            default=0.0,
        )
    return {
        "mode": mode,
        "rollout_index": rollout_index,
        "image_index": pair["image_index"],
        "patient_id": pair["patient_id"],
        "disease": pair["disease"],
        "completion": completion,
        "final_box_norm_1000": parsed.final_box_norm_1000,
        "valid_output": valid,
        "iou": float(iou),
        "recall_0_5": float(iou >= 0.5),
        "parse_error": parsed.error,
    }


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ious = [record["iou"] for record in records]
    return {
        "supported": True,
        "n_predictions": len(records),
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "median_iou": statistics.median(ious) if ious else 0.0,
        "recall_at_0_5": (
            sum(record["recall_0_5"] for record in records) / len(records)
            if records
            else 0.0
        ),
        "valid_output_rate": (
            sum(record["valid_output"] for record in records) / len(records)
            if records
            else 0.0
        ),
    }


def _ordinary_completion(
    model,
    tokenizer,
    inputs: Dict[str, Any],
    config: Dict[str, Any],
    mode: str,
) -> str:
    prompt_length = inputs["input_ids"].size(1)
    with torch.no_grad():
        generated = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs["image_grid_hws"],
            tokenizer=tokenizer,
            n_future_tokens=6,
            max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
            use_cache=True,
            generation_mode=mode,
            temperature=float(config["evaluation"]["ordinary_temperature"]),
            top_k=None,
            top_p=1.0,
            repetition_penalty=1.0,
        )
    return tokenizer.decode(
        generated[0, prompt_length:].tolist(), skip_special_tokens=False
    )


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    output_dir = assert_new_output_dir(args.output_dir)
    pairs = load_verified_pairs(config, "test")
    device = torch.device(args.device)
    model, tokenizer, processor, revision = build_policy(config, device)
    if args.checkpoint:
        _load_checkpoint(model, Path(args.checkpoint), device)

    mode_records: Dict[str, List[Dict[str, Any]]] = {
        "rl_native_stochastic_pbd": [],
        "fast_mtp": [],
        "slow_ntp": [],
        "hybrid": [],
    }
    mode_support: Dict[str, Dict[str, Any]] = {}
    ordinary_modes = {
        "fast_mtp": "fast",
        "slow_ntp": "slow",
        "hybrid": "hybrid",
    }

    for sample_index, pair in enumerate(pairs):
        inputs = tokenize_rl_pair(processor, pair, device, config=config)
        parser_name = resolve_parser_name(config)
        traces = generate_rollout_group(
            model,
            tokenizer,
            inputs,
            config,
            sample_seed=int(config["evaluation"]["seed"]) + sample_index,
        )
        mode_records["rl_native_stochastic_pbd"].extend(
            _score_completion(
                trace.decoded_text or "",
                pair,
                mode="rl_native_stochastic_pbd",
                rollout_index=group_index,
                parser_name=parser_name,
            )
            for group_index, trace in enumerate(traces)
        )
        for report_name, api_mode in ordinary_modes.items():
            if mode_support.get(report_name, {}).get("supported") is False:
                continue
            try:
                completion = _ordinary_completion(
                    model, tokenizer, inputs, config, api_mode
                )
            except (AssertionError, NotImplementedError) as error:
                mode_support[report_name] = {
                    "supported": False,
                    "reason": f"{type(error).__name__}: {error}",
                }
                continue
            mode_records[report_name].append(
                _score_completion(
                    completion,
                    pair,
                    mode=report_name,
                    rollout_index=0,
                    parser_name=parser_name,
                )
            )

    report = {
        "model_revision": revision,
        "checkpoint": args.checkpoint,
        "test_split_sha256": config["data"]["test_sha256"],
        "modes": {},
    }
    for mode, records in mode_records.items():
        if mode in mode_support and not mode_support[mode]["supported"]:
            report["modes"][mode] = mode_support[mode]
        else:
            report["modes"][mode] = _summarize(records)
        append_jsonl(output_dir / f"{mode}_predictions.jsonl", records)
    write_json(output_dir / "evaluation_report.json", report)
    (output_dir / "evaluation_report.compact.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
