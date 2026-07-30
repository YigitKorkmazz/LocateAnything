#!/usr/bin/env python3
"""Full reward-only GRPO training for native stochastic PBD-RL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.pbd_rl import PBDRolloutReplayer  # noqa: E402
from rl.policy_state import (  # noqa: E402
    PolicySnapshot,
    assert_approved_trainable_parameters,
    use_policy_snapshot,
)
from rl.rewards import MedCLIPSemanticScorer, ProductionRewardPipeline  # noqa: E402
from rl.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    append_jsonl,
    assert_new_output_dir,
    build_policy,
    generate_rollout_group,
    load_resolved_config,
    load_verified_pairs,
    projector_state_cpu,
    save_rl_checkpoint,
    tokenize_rl_pair,
    trace_record,
    write_json,
)
from train_chestxray8_sft import build_optimizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=None,
        help="Optional override of training.max_optimizer_steps",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Disabled by default; pass an explicit checkpoint path only when intended.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    if config["objective"]["loss_total"] != "L_GRPO":
        raise RuntimeError("training requires L_total=L_GRPO")
    if config["objective"]["supervised_losses"]:
        raise RuntimeError("training must not include supervised losses")
    if args.resume_from_checkpoint or config["training"].get("resume_from_checkpoint"):
        raise RuntimeError(
            "resume_from_checkpoint is disabled unless explicitly implemented later"
        )

    train_cfg = config["training"]
    max_steps = int(
        args.max_optimizer_steps
        if args.max_optimizer_steps is not None
        else train_cfg["max_optimizer_steps"]
    )
    output_dir = assert_new_output_dir(args.output_dir)
    write_json(output_dir / "resolved_config.json", config)

    pairs = load_verified_pairs(config, "train")
    device = torch.device(args.device)
    model, tokenizer, processor, revision = build_policy(config, device)
    trainability = assert_approved_trainable_parameters(model)
    reference_projector = projector_state_cpu(model)
    old_snapshot = PolicySnapshot.capture(model, optimizer_step=0)

    optimizer = build_optimizer(
        model,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        use_8bit_adam=bool(train_cfg["use_8bit_adam"]),
        projector_lr=float(train_cfg["projector_learning_rate"]),
    )
    scorer = MedCLIPSemanticScorer(device=device)
    reward_cfg = config["rewards"]
    rewards = ProductionRewardPipeline(
        scorer,
        format_weight=reward_cfg["format"]["weight"],
        spatial_weight=reward_cfg["spatial"]["weight"],
        semantic_weight=reward_cfg["semantic"]["weight"],
        iou_threshold=reward_cfg["spatial"]["iou_threshold"],
    )
    replayer = PBDRolloutReplayer(model, tokenizer)
    clip_epsilon = float(config["objective"]["ppo_clip_epsilon"])
    sync_interval = int(config["objective"]["old_policy_sync_interval_optimizer_steps"])
    if sync_interval != 1:
        raise RuntimeError("old-policy sync interval must be 1")

    metrics_path = output_dir / "train_metrics.jsonl"
    traces_path = output_dir / "rollout_traces.jsonl"
    optimizer_step = 0
    epoch = 0
    write_json(
        output_dir / "train_init.json",
        {
            "revision": revision,
            "trainable_tensor_count": len(trainability["trainable_names"]),
            "reference_projector_keys": sorted(reference_projector),
            "max_optimizer_steps": max_steps,
            "loss_total": "L_GRPO",
            "supervised_losses": [],
        },
    )

    while optimizer_step < max_steps:
        epoch += 1
        for sample_index, pair in enumerate(pairs):
            if optimizer_step >= max_steps:
                break
            inputs = tokenize_rl_pair(processor, pair, device)
            with use_policy_snapshot(model, old_snapshot):
                with torch.no_grad():
                    traces = generate_rollout_group(
                        model,
                        tokenizer,
                        inputs,
                        config,
                        sample_seed=int(train_cfg["seed"]) + sample_index + epoch * 100000,
                    )
            components = [
                rewards.score(trace.decoded_text or "", pair) for trace in traces
            ]
            advantages = group_relative_advantages(
                [component.total_reward for component in components]
            ).to(device)
            with torch.no_grad(), use_policy_snapshot(model, old_snapshot):
                old_logps = torch.stack(
                    [
                        replayer.score(
                            trace,
                            pixel_values=inputs["pixel_values"],
                            input_ids=inputs["input_ids"],
                            image_grid_hws=inputs["image_grid_hws"],
                        )[0]
                        for trace in traces
                    ]
                )
            current_logps = torch.stack(
                [
                    replayer.score(
                        trace,
                        pixel_values=inputs["pixel_values"],
                        input_ids=inputs["input_ids"],
                        image_grid_hws=inputs["image_grid_hws"],
                    )[0]
                    for trace in traces
                ]
            )
            loss = grpo_clipped_loss(
                current_logps,
                old_logps,
                advantages,
                clip_epsilon=clip_epsilon,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(train_cfg["max_grad_norm"]),
            )
            optimizer.step()
            optimizer_step += 1
            old_snapshot = PolicySnapshot.capture(
                model, optimizer_step=optimizer_step
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
            append_jsonl(traces_path, records)
            append_jsonl(
                metrics_path,
                [
                    {
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "sample_index": sample_index,
                        "loss_grpo": float(loss.detach().cpu()),
                        "loss_total": float(loss.detach().cpu()),
                        "supervised_loss": None,
                        "mean_reward": float(
                            sum(c.total_reward for c in components) / len(components)
                        ),
                        "mean_advantage": float(advantages.detach().cpu().mean()),
                        "old_policy_synced": True,
                    }
                ],
            )
            if optimizer_step % int(train_cfg["checkpoint_interval"]) == 0:
                save_rl_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    step=optimizer_step,
                    config=config,
                    reference_projector=reference_projector,
                )

    write_json(
        output_dir / "train_summary.json",
        {
            "optimizer_steps": optimizer_step,
            "epochs_completed": epoch,
            "loss_total": "L_GRPO",
            "supervised_losses": [],
            "reference_projector_unchanged": True,
        },
    )
    print(f"Training artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
