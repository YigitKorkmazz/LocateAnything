#!/usr/bin/env python3
"""Held-out evaluation for ChestX-ray8 PBD-RL checkpoints."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from eval_locateanything_bbox import box_iou  # noqa: E402
from rl.pbd_rl import StochasticPBDRLDecoder  # noqa: E402
from rl.prompt import build_rl_user_text, resolve_prompt_mode  # noqa: E402
from rl.rewards import (  # noqa: E402
    COMPLETION_PARSERS,
    is_valid_geometry,
    resolve_parser_name,
)
from rl.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    assert_new_output_dir,
    load_resolved_config,
    load_verified_pairs,
    sampling_from_config,
    tokenize_rl_pair,
    write_json,
)
from sft_common import LLM_LORA_TARGET_MODULES  # noqa: E402
from train_chestxray8_sft import (  # noqa: E402
    apply_llm_lora,
    load_locateanything_model,
    load_lora_adapter_weights,
    load_tokenizer_and_processor,
    unfreeze_mlp1,
)


SUPPORTED_MODES = {
    "rl_native_stochastic_pbd": "supported",
    "fast_mtp": "supported",
    "slow_ntp": "supported",
    "hybrid": "supported",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=None,
        help="Subset of evaluation modes; defaults to resolved YAML list.",
    )
    return parser.parse_args()


def load_eval_policy(config: Dict[str, Any], checkpoint: Optional[str], device: torch.device):
    model_cfg = config["model"]
    tokenizer, processor = load_tokenizer_and_processor(
        model_cfg["name_or_path"],
        revision=model_cfg["revision"],
        max_seq_length=int(config["rollout"]["max_sequence_length"]),
    )
    model, revision = load_locateanything_model(
        model_cfg["name_or_path"],
        tokenizer,
        revision=model_cfg["revision"],
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    if checkpoint:
        ckpt = Path(checkpoint)
        apply_llm_lora(
            model,
            rank=int(model_cfg["lora"]["r"]),
            alpha=int(model_cfg["lora"]["alpha"]),
            dropout=float(model_cfg["lora"]["dropout"]),
            target_modules=LLM_LORA_TARGET_MODULES,
        )
        unfreeze_mlp1(model)
        adapter = ckpt / "adapter"
        if adapter.is_dir():
            load_lora_adapter_weights(model, adapter)
        mlp1_path = ckpt / "mlp1.pt"
        if mlp1_path.is_file():
            model.mlp1.load_state_dict(torch.load(mlp1_path, map_location=device))
    model.eval()
    return model, tokenizer, processor, revision


def final_box_from_text(
    text: str,
    *,
    parser_name: str = "chain_of_box",
) -> Optional[Tuple[int, int, int, int]]:
    parsed = COMPLETION_PARSERS[parser_name](text)
    if not parsed.format_valid or parsed.final_box_norm_1000 is None:
        return None
    if not is_valid_geometry(parsed.final_box_norm_1000):
        return None
    return parsed.final_box_norm_1000


def pair_metrics(
    pred_box: Optional[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    if pred_box is None:
        return {
            "valid_output": False,
            "iou": 0.0,
            "recall_at_0_5": 0.0,
        }
    ious = [box_iou(pred_box, gt) for gt in gt_boxes]
    best = max(ious) if ious else 0.0
    return {
        "valid_output": True,
        "iou": float(best),
        "recall_at_0_5": float(best > 0.5),
    }


def summarize_mode(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n_pairs": 0,
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "recall_at_0_5": 0.0,
            "valid_output_rate": 0.0,
        }
    ious = [row["iou"] for row in rows]
    return {
        "n_pairs": len(rows),
        "mean_iou": float(statistics.fmean(ious)),
        "median_iou": float(statistics.median(ious)),
        "recall_at_0_5": float(statistics.fmean(row["recall_at_0_5"] for row in rows)),
        "valid_output_rate": float(
            statistics.fmean(float(row["valid_output"]) for row in rows)
        ),
    }


@torch.no_grad()
def evaluate_rl_native(
    model,
    tokenizer,
    processor,
    pairs: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    decoder = StochasticPBDRLDecoder(
        model, tokenizer, sampling=sampling_from_config(config)
    )
    rows = []
    for index, pair in enumerate(pairs):
        inputs = tokenize_rl_pair(processor, pair, device, config=config)
        trace = decoder.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            image_grid_hws=inputs["image_grid_hws"],
            max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
            seed=int(config["evaluation"]["seed"]) + index,
            force_first_box_block=False,
        )
        pred = final_box_from_text(
            trace.decoded_text or "",
            parser_name=resolve_parser_name(config),
        )
        metrics = pair_metrics(pred, pair["gt_boxes_norm_1000"])
        rows.append(
            {
                "image_index": pair["image_index"],
                "disease": pair["disease"],
                "mode": "rl_native_stochastic_pbd",
                "completion": trace.decoded_text,
                "pred_box_norm_1000": list(pred) if pred else None,
                **metrics,
            }
        )
    return rows


@torch.no_grad()
def evaluate_ordinary_mode(
    model,
    tokenizer,
    processor,
    pairs: Sequence[Dict[str, Any]],
    *,
    generation_mode: str,
    mode_name: str,
    config: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Use the unchanged LocateAnything generate() path (fast/slow/hybrid)."""
    rows = []
    max_new_tokens = int(config["evaluation"]["max_new_tokens"])
    temperature = float(config["evaluation"]["ordinary_temperature"])
    for pair in pairs:
        image = Image.open(pair["image_path"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": build_rl_user_text(
                            pair, prompt_mode=resolve_prompt_mode(config)
                        ),
                    },
                ],
            }
        ]
        text = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = processor.process_vision_info(messages)
        inputs = processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(device)
        response = model.generate(
            pixel_values=inputs["pixel_values"].to(model.language_model.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws"),
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=1.0,
            top_k=None,
            repetition_penalty=1.0,
            use_cache=True,
        )
        answer = tokenizer.decode(
            response[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=False,
        )
        pred = final_box_from_text(
            answer, parser_name=resolve_parser_name(config)
        )
        metrics = pair_metrics(pred, pair["gt_boxes_norm_1000"])
        rows.append(
            {
                "image_index": pair["image_index"],
                "disease": pair["disease"],
                "mode": mode_name,
                "completion": answer,
                "pred_box_norm_1000": list(pred) if pred else None,
                **metrics,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    pairs = load_verified_pairs(config, "test")
    output_dir = assert_new_output_dir(args.output_dir)
    modes = list(args.modes or config["evaluation"]["modes"])
    device = torch.device(args.device)

    mode_status = {
        mode: SUPPORTED_MODES.get(mode, "unsupported") for mode in modes
    }
    write_json(output_dir / "mode_support.json", mode_status)

    model, tokenizer, processor, revision = load_eval_policy(
        config, args.checkpoint, device
    )
    summaries: Dict[str, Any] = {"revision": revision, "modes": {}}
    all_rows: Dict[str, List[Dict[str, Any]]] = {}

    ordinary = {
        "fast_mtp": "fast",
        "slow_ntp": "slow",
        "hybrid": "hybrid",
    }
    for mode_name in modes:
        status = mode_status.get(mode_name, "unsupported")
        if status != "supported":
            summaries["modes"][mode_name] = {
                "status": "unsupported",
                "reason": "mode not supported by existing LocateAnything APIs",
            }
            continue
        if mode_name == "rl_native_stochastic_pbd":
            rows = evaluate_rl_native(
                model, tokenizer, processor, pairs, config, device
            )
        else:
            rows = evaluate_ordinary_mode(
                model,
                tokenizer,
                processor,
                pairs,
                generation_mode=ordinary[mode_name],
                mode_name=mode_name,
                config=config,
                device=device,
            )
        all_rows[mode_name] = rows
        summaries["modes"][mode_name] = {
            "status": "supported",
            **summarize_mode(rows),
        }

    for mode, rows in all_rows.items():
        write_json(output_dir / f"predictions_{mode}.json", rows)
    write_json(output_dir / "eval_summary.json", summaries)
    print(f"Evaluation artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
