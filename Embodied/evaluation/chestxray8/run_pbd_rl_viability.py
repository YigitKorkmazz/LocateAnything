#!/usr/bin/env python3
"""Exactly-100-sample, rollout-only ChestX-ray8 PBD-RL viability CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.grpo import group_relative_advantages  # noqa: E402
from rl.rewards import (  # noqa: E402
    MedCLIPSemanticScorer,
    build_reward_pipeline_from_config,
)
from rl.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_HYBRID_NATIVE_CONFIG,
    DEFAULT_NATIVE_CONFIG,
    append_jsonl,
    assert_new_output_dir,
    build_policy,
    fixed_sample_indices,
    generate_rollout_group,
    load_resolved_config,
    load_verified_pairs,
    summarize_rewards,
    tokenize_rl_pair,
    trace_record,
    write_json,
)


def build_viability_plan(config: Dict[str, Any], population_size: int) -> Dict[str, Any]:
    viability = config["viability"]
    count = int(viability["sample_count"])
    seed = int(viability["selection_seed"])
    indices = fixed_sample_indices(population_size, count, seed)
    if config["viability"]["optimizer_updates"] != 0:
        raise RuntimeError("viability must perform zero optimizer updates")
    return {
        "sample_count": count,
        "selection_seed": seed,
        "sample_indices": indices,
        "group_size": int(config["objective"]["group_size"]),
        "optimizer_updates": 0,
        "save_checkpoints": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "100-sample rollout-only PBD-RL viability. "
            "Select prompt/reward mode via --config YAML "
            "(Chain-of-Box or native LocateAnything)."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=(
            "Resolved experiment YAML. Use "
            f"{DEFAULT_NATIVE_CONFIG.name} for PBD-only native format, or "
            f"{DEFAULT_HYBRID_NATIVE_CONFIG.name} for Hybrid-aware native."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate fixed selection/configuration without loading any model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    pairs = load_verified_pairs(config, "train")
    plan = build_viability_plan(config, len(pairs))
    output_dir = assert_new_output_dir(args.output_dir)
    selected = [pairs[index] for index in plan["sample_indices"]]
    write_json(output_dir / "viability_plan.json", plan)
    write_json(
        output_dir / "selected_samples.json",
        [
            {
                "sample_index": index,
                "image_index": pair["image_index"],
                "patient_id": pair["patient_id"],
                "disease": pair["disease"],
            }
            for index, pair in zip(plan["sample_indices"], selected)
        ],
    )
    write_json(
        output_dir / "resolved_config.json",
        {
            "config_path": config["_config_path"],
            "prompt_mode": config["prompt"]["mode"],
            "reward_parser": config["rewards"]["parser"],
            "rollout_path": config["rollout"].get("path"),
            "reward_weights": {
                "format": config["rewards"]["format"]["weight"],
                "spatial": config["rewards"]["spatial"]["weight"],
                "semantic": config["rewards"]["semantic"]["weight"],
            },
            "experiment": config.get("experiment"),
        },
    )
    if args.dry_run:
        print(f"DRY RUN OK: {output_dir / 'viability_plan.json'}")
        return

    device = torch.device(args.device)
    model, tokenizer, processor, revision = build_policy(config, device)
    scorer = MedCLIPSemanticScorer(device=device)
    rewards = build_reward_pipeline_from_config(config, scorer)
    all_records: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    trace_path = output_dir / "rollout_traces.jsonl"

    for ordinal, (sample_index, pair) in enumerate(
        zip(plan["sample_indices"], selected)
    ):
        inputs = tokenize_rl_pair(processor, pair, device, config=config)
        traces = generate_rollout_group(
            model,
            tokenizer,
            inputs,
            config,
            sample_seed=plan["selection_seed"] + sample_index,
        )
        components = [
            rewards.score_from_trace(trace, pair) for trace in traces
        ]
        advantages = group_relative_advantages(
            [component.total_reward for component in components]
        )
        records = [
            trace_record(
                trace,
                sample_index=sample_index,
                group_index=group_index,
                pair=pair,
                reward=component.to_dict(),
                advantage=float(advantages[group_index]),
            )
            for group_index, (trace, component) in enumerate(
                zip(traces, components)
            )
        ]
        append_jsonl(trace_path, records)
        all_records.extend(records)
        if ordinal < int(config["viability"]["representative_examples"]):
            examples.extend(records)

    summary = {
        **summarize_rewards(all_records),
        "sample_count": 100,
        "group_size": 4,
        "optimizer_updates": 0,
        "checkpoint_saving": False,
        "model_revision": revision,
        "config_path": config["_config_path"],
        "prompt_mode": config["prompt"]["mode"],
        "reward_parser": config["rewards"]["parser"],
    }
    write_json(output_dir / "viability_metrics.json", summary)
    write_json(output_dir / "representative_completions.json", examples)
    print(f"Viability artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
