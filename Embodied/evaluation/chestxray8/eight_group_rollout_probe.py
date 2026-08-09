#!/usr/bin/env python3
"""Fresh-start rollout-only groups on the production shard (eight by default)."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fresh_start_regression_audit import DEFAULT_AUDIT_ROOT  # noqa: E402
from rl.rewards import MedCLIPSemanticScorer, build_reward_pipeline_from_config  # noqa: E402
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    build_policy_two_gpu_live_cache,
    generate_rollout_group,
    load_resolved_config,
    load_verified_pairs,
    tokenize_rl_pair,
)
from two_gpu_g4_grpo_multistep_smoke import (  # noqa: E402
    FRESH_START_GUARD_GROUPS,
    GROUP_SIZE,
    _attempt_schedule,
    _atomic_write_json,
    _memory,
    _reset_peaks,
    _validate_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG))
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--groups", type=int, default=FRESH_START_GUARD_GROUPS)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()
    if args.groups <= 0:
        raise ValueError("--groups must be positive")
    output_name = args.output_file or (
        "eight_group_rollout_only_probe.json"
        if args.groups == FRESH_START_GUARD_GROUPS
        else f"{args.groups}_group_rollout_only_probe.json"
    )
    output = Path(args.audit_root).resolve() / output_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("requires exactly two visible GPUs")
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    config = load_resolved_config(args.config)
    _validate_config(config)
    required_gpu_name = (config.get("hardware") or {}).get("required_gpu_name_substring")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if required_gpu_name and any(required_gpu_name not in name for name in gpu_names):
        raise RuntimeError(
            f"requires two {required_gpu_name} GPUs; visible devices are {gpu_names}"
        )
    seed = int(config["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    model, tokenizer, processor, revision, shard = build_policy_two_gpu_live_cache(config)
    model.eval()
    pairs = load_verified_pairs(config, "train")
    scorer = MedCLIPSemanticScorer(device=devices[0])
    rewards = build_reward_pipeline_from_config(config, scorer)
    groups = []
    branch_counts: Counter[str] = Counter()
    valid_boxes = 0
    nontruncated = 0
    _reset_peaks(devices)
    sample_cursor = 0
    for attempted in range(1, args.groups + 1):
        schedule = _attempt_schedule(
            seed=seed,
            attempted_group_count=attempted,
            sample_cursor=sample_cursor,
            sample_count=len(pairs),
        )
        pair = pairs[schedule["sample_index"]]
        inputs = tokenize_rl_pair(processor, pair, devices[0], config=config)
        events = []
        with torch.no_grad():
            traces = generate_rollout_group(
                model,
                tokenizer,
                inputs,
                config,
                sample_seed=schedule["attempt_seed"],
                diagnostic_observer=events.append,
            )
        if hasattr(scorer, "model"):
            scorer.model.to(devices[0])
        components = [rewards.score_from_trace(trace, pair) for trace in traces]
        if hasattr(scorer, "model"):
            scorer.model.to("cpu")
        trajectories = []
        for index, (trace, component) in enumerate(zip(traces, components)):
            branch = str(trace.reward_branch)
            branch_counts[branch] += 1
            valid_boxes += int(trace.has_unambiguous_committed_box)
            nontruncated += int(not trace.truncated)
            rollout_seed = schedule["rollout_seeds"][index]
            trajectories.append(
                {
                    "group_index": index,
                    "rollout_seed": rollout_seed,
                    "trace": trace.to_dict(),
                    "reward": component.to_dict(),
                    "observer_events": [
                        event for event in events if event.get("seed") == rollout_seed
                    ],
                }
            )
        groups.append(
            {
                "attempted_group_count": attempted,
                "schedule": schedule,
                "sample_index": schedule["sample_index"],
                "sample": dict(pair),
                "rendered_prompt": inputs["rendered_prompt"],
                "prompt_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
                "image_grid_hws": inputs["image_grid_hws"].detach().cpu().tolist(),
                "trajectories": trajectories,
            }
        )
        sample_cursor += 1
        del traces, components, inputs, events
        gc.collect()
        torch.cuda.empty_cache()
    acceptance = {
        "exactly_requested_groups": len(groups) == args.groups,
        "exactly_requested_trajectories": (
            sum(len(group["trajectories"]) for group in groups)
            == args.groups * GROUP_SIZE
        ),
    }
    if args.groups == FRESH_START_GUARD_GROUPS:
        acceptance.update(
            {
                "exactly_eight_groups": len(groups) == FRESH_START_GUARD_GROUPS,
                "exactly_32_trajectories": (
                    sum(len(group["trajectories"]) for group in groups) == 32
                ),
                "at_least_one_valid_committed_bbox": valid_boxes >= 1,
                "at_least_one_nontruncated_trajectory": nontruncated >= 1,
                "branch_distribution_not_entirely_none": branch_counts.get("none", 0) < 32,
            }
        )
    acceptance["passed"] = all(acceptance.values())
    report = {
        "format": "fresh_start_rollout_probe_v2",
        "requested_groups": args.groups,
        "status": "passed" if acceptance["passed"] else "failed",
        "model_revision": revision,
        "visible_gpu_names": gpu_names,
        "config_path": config["_config_path"],
        "shard": shard,
        "seed": seed,
        "branch_distribution": dict(branch_counts),
        "valid_committed_bbox_count": valid_boxes,
        "nontruncated_trajectory_count": nontruncated,
        "peak_memory": _memory(devices),
        "groups": groups,
        "acceptance": acceptance,
    }
    _atomic_write_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output)}))
    if not acceptance["passed"]:
        raise RuntimeError("eight-group rollout acceptance failed")


if __name__ == "__main__":
    main()
