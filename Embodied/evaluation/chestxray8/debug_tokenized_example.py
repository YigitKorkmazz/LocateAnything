#!/usr/bin/env python3
"""
Print one complete supervised training example for ChestX-ray8 LocateAnything SFT.

Shows user prompt, assistant target, rendered chat template, input IDs, labels,
masked positions, normalized boxes, and decoded target text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CHEST_DIR))

from sft_common import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    IGNORE_INDEX,
    read_jsonl,
    set_global_seed,
)
from train_chestxray8_sft import (  # noqa: E402
    build_assistant_only_labels,
    load_tokenizer_and_processor,
    messages_from_pair,
    tokenize_messages,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-file",
        type=str,
        default=str(CHEST_DIR / "splits" / f"train_pairs_seed{DEFAULT_SEED}.jsonl"),
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(REPO_ROOT / "results" / "finetuning" / "debug_tokenized_example.json"),
    )
    args = parser.parse_args()
    set_global_seed(args.seed)

    pairs = read_jsonl(Path(args.split_file))
    if not pairs:
        raise RuntimeError(f"No pairs in {args.split_file}")
    pair = pairs[args.sample_index % len(pairs)]

    tokenizer, processor = load_tokenizer_and_processor(
        args.model_path, revision=args.model_revision, max_seq_length=4096
    )
    messages = messages_from_pair(pair)
    batch = tokenize_messages(processor, messages)
    input_ids = batch["input_ids"][0]
    labels = build_assistant_only_labels(input_ids, tokenizer)
    masked_positions = (labels == IGNORE_INDEX).nonzero(as_tuple=False).flatten().tolist()
    supervised_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=False).flatten().tolist()

    rendered = processor.py_apply_chat_template(messages, tokenize=False)
    decoded_all = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
    target_ids = [int(labels[i]) for i in supervised_positions]
    decoded_target = tokenizer.decode(target_ids, skip_special_tokens=False)

    report: Dict[str, Any] = {
        "sample_index": args.sample_index,
        "image_index": pair["image_index"],
        "disease": pair["disease"],
        "patient_id": pair["patient_id"],
        "user_prompt": pair["user_query"],
        "assistant_target": pair["assistant_target"],
        "normalized_boxes": pair["gt_boxes_norm_1000"],
        "rendered_chat_template": rendered,
        "input_token_ids": input_ids.tolist(),
        "labels": labels.tolist(),
        "n_tokens": int(input_ids.numel()),
        "n_masked_label_positions": len(masked_positions),
        "n_supervised_label_positions": len(supervised_positions),
        "masked_label_positions_preview": masked_positions[:64],
        "supervised_label_positions": supervised_positions,
        "decoded_full_sequence": decoded_all,
        "decoded_target_text": decoded_target,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 72)
    print("ONE COMPLETE SUPERVISED EXAMPLE")
    print("=" * 72)
    print(f"image_index : {pair['image_index']}")
    print(f"disease     : {pair['disease']}")
    print(f"user_prompt : {pair['user_query']}")
    print(f"assistant   : {pair['assistant_target']}")
    print(f"norm boxes  : {pair['gt_boxes_norm_1000']}")
    print("-" * 72)
    print("rendered chat template:")
    print(rendered)
    print("-" * 72)
    print(f"n_tokens={report['n_tokens']}  supervised={report['n_supervised_label_positions']}  masked={report['n_masked_label_positions']}")
    print("decoded target text:")
    print(decoded_target)
    print(f"saved: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
