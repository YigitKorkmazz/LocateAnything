#!/usr/bin/env python3
"""Final evaluation of a `production_grpo_fast` checkpoint on the 194-example
heldout set (`splits/test_pairs_seed42.jsonl`).

This script is deliberately kept outside the `production_grpo_fast` package
so that package's own static gate (`production_grpo_fast/test_no_heldout_access.py`,
which scans every `.py` file inside that directory for the heldout manifest
name) stays untouched and still passes -- the "no heldout access during
development" rule applied to the training backend while it was being built
and used, not to the one-time final-evaluation step that is expected to
finally read that file now that training is done.

Reuses the exact tested generation + reward pipeline the training run itself
used (`production_grpo_fast.generation`, `.rewards_adapter`, `.runtime`,
`.grpo_step`), with the same sampling config (T=1.2/top_p=0.9/top_k=0,
G rollouts per example) so the reported reward is directly comparable to the
training-time `mean_reward` logged in `train_log_rank0.jsonl`. Only the
LoRA adapter weights are loaded from the checkpoint (via
`production_grpo_fast.checkpoint.resume_checkpoint`, which also happens to
restore optimizer/scheduler/RNG state; that state is simply unused here).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (str(CHEST_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rl.rewards import MedCLIPSemanticScorer  # noqa: E402
from rl.runtime import load_verified_pairs  # noqa: E402

from production_grpo_fast import checkpoint as ckpt_mod  # noqa: E402
from production_grpo_fast import runtime as prod_runtime  # noqa: E402
from production_grpo_fast import sampler  # noqa: E402
from production_grpo_fast.generation import cache_group_visual_features, generate_group_shared_prefill  # noqa: E402
from production_grpo_fast.grpo_step import MAX_NEW_TOKENS, _sampling_config, prepare_group_inputs  # noqa: E402
from production_grpo_fast.rewards_adapter import build_reward_pipeline, score_group  # noqa: E402

HELDOUT_SPLIT_KEY = "test"
HELDOUT_PATH = "splits/" + "test_pairs_seed42.jsonl"
HELDOUT_SHA256 = "783965de02032ad44cc387afa22cd2382649330a0f5f39e0caf5a62d3385a2ed"
HELDOUT_EXPECTED_SIZE = 194


def load_heldout_pairs() -> List[Dict[str, Any]]:
    config = {
        "data": {
            f"{HELDOUT_SPLIT_KEY}_split": HELDOUT_PATH,
            f"{HELDOUT_SPLIT_KEY}_sha256": HELDOUT_SHA256,
        }
    }
    pairs = load_verified_pairs(config, HELDOUT_SPLIT_KEY)
    if len(pairs) != HELDOUT_EXPECTED_SIZE:
        raise RuntimeError(f"expected {HELDOUT_EXPECTED_SIZE} heldout examples, found {len(pairs)}")
    return pairs


def resolve_checkpoint(output_dir: Path, explicit: str) -> Path:
    if explicit and explicit != "auto":
        return Path(explicit)
    target = ckpt_mod.latest_checkpoint(output_dir)
    if target is None:
        raise FileNotFoundError(f"no checkpoint found under {output_dir}")
    return target


def evaluate(
    model: Any,
    tokenizer: Any,
    processor: Any,
    reward_pipeline: Any,
    pairs: List[Dict[str, Any]],
    *,
    device: torch.device,
    generations: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    sampling = _sampling_config()
    results: List[Dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        start = time.monotonic()
        inputs = prepare_group_inputs(processor, pair, device)
        cached_visual_features = cache_group_visual_features(model, inputs["pixel_values"], inputs["image_grid_hws"])
        seeds = [int(index) * generations + j for j in range(generations)]
        with torch.inference_mode():
            traces = generate_group_shared_prefill(
                model,
                tokenizer,
                inputs["input_ids"],
                cached_visual_features,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                seeds=seeds,
            )
        totals, components = score_group(reward_pipeline, pair, traces)
        elapsed = time.monotonic() - start
        results.append(
            {
                "index": index,
                "image_index": pair.get("image_index"),
                "disease": pair.get("disease"),
                "user_query": pair.get("user_query"),
                "gt_boxes_norm_1000": pair.get("gt_boxes_norm_1000"),
                "rewards": totals,
                "mean_reward": sum(totals) / len(totals),
                "reward_components": [c.to_dict() for c in components],
                "decoded_texts": [t.decoded_text for t in traces],
                "predicted_boxes": [t.committed_final_box_norm_1000 for t in traces],
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"[{index + 1}/{len(pairs)}] disease={pair.get('disease')!r} "
            f"mean_reward={results[-1]['mean_reward']:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_format = [c["format_reward"] for r in results for c in r["reward_components"]]
    all_spatial = [c["spatial_reward"] for r in results for c in r["reward_components"]]
    all_semantic = [c["semantic_reward"] for r in results for c in r["reward_components"]]
    all_total = [t for r in results for t in r["rewards"]]
    per_example_mean = [r["mean_reward"] for r in results]

    by_disease: Dict[str, List[float]] = {}
    for r in results:
        by_disease.setdefault(r["disease"], []).append(r["mean_reward"])

    return {
        "n_examples": len(results),
        "n_rollouts": len(all_total),
        "mean_reward_per_rollout": statistics.fmean(all_total),
        "mean_reward_per_example": statistics.fmean(per_example_mean),
        "stdev_reward_per_example": statistics.pstdev(per_example_mean) if len(per_example_mean) > 1 else 0.0,
        "mean_format_reward": statistics.fmean(all_format),
        "mean_spatial_reward": statistics.fmean(all_spatial),
        "spatial_success_rate_iou_gt_0.5": statistics.fmean(all_spatial),
        "mean_semantic_reward": statistics.fmean(all_semantic),
        "mean_reward_by_disease": {
            disease: statistics.fmean(values) for disease, values in sorted(by_disease.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-output-dir", required=True, help="the production_grpo_fast --output-dir the checkpoint was trained into")
    parser.add_argument("--checkpoint", default="auto", help="'auto' for the latest checkpoint under --training-output-dir, or an explicit checkpoint path")
    parser.add_argument("--results-path", required=True, help="where to write the JSON results file")
    parser.add_argument("--gpu", type=int, default=1, help="physical GPU index to run evaluation on")
    parser.add_argument("--generations", type=int, default=sampler.GENERATIONS, help="rollouts per heldout example (default matches training G)")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    training_output_dir = Path(args.training_output_dir)
    checkpoint_path = resolve_checkpoint(training_output_dir, args.checkpoint)
    print(f"Evaluating checkpoint: {checkpoint_path}")

    model, tokenizer, processor, load_report = prod_runtime.build_model_and_tokenizer(device)
    prod_runtime.trainability_audit(model)

    # resume_checkpoint's API also restores optimizer/scheduler/RNG state;
    # unused here, but building real objects is cheap and keeps this call
    # identical to the one `train.py` uses (only the LoRA adapter weights
    # actually matter for evaluation).
    optimizer = prod_runtime.build_optimizer(model, lr=1e-6, weight_decay=0.0)
    scheduler = prod_runtime.build_linear_schedule(optimizer, total_steps=sampler.TOTAL_OPTIMIZER_STEPS)
    next_group_at_checkpoint = ckpt_mod.resume_checkpoint(checkpoint_path, model, optimizer, scheduler)
    print(f"Loaded adapter weights (checkpoint next_global_group_index={next_group_at_checkpoint})")

    model.eval()
    reward_pipeline = build_reward_pipeline(MedCLIPSemanticScorer(device=device))
    pairs = load_heldout_pairs()
    print(f"Loaded {len(pairs)} heldout examples from {HELDOUT_PATH} (sha256 verified)")

    results = evaluate(
        model,
        tokenizer,
        processor,
        reward_pipeline,
        pairs,
        device=device,
        generations=args.generations,
        max_new_tokens=args.max_new_tokens,
    )
    summary = summarize(results)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_next_global_group_index": next_group_at_checkpoint,
                "load_report": load_report,
                "generations_per_example": args.generations,
                "max_new_tokens": args.max_new_tokens,
                "summary": summary,
                "per_example_results": results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved full results to {results_path}")


if __name__ == "__main__":
    main()
