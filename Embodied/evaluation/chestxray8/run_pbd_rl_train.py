#!/usr/bin/env python3
"""Full reward-only GRPO training for native stochastic PBD/Hybrid-RL."""

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

from rl.cuda_memory import (  # noqa: E402
    current_allocated_mb,
    peak_reserved_mb,
    record_stage_peak,
    release_cuda_temporaries,
    reset_peak_memory,
)
from rl.grpo import group_relative_advantages  # noqa: E402
from rl.grpo_train_step import (  # noqa: E402
    assert_production_cached_replay_checkpointing_disabled,
    memory_safe_grpo_optimizer_step,
    replay_semantics_report,
)
from rl.policy_state import (  # noqa: E402
    PolicySnapshot,
    assert_approved_trainable_parameters,
    use_policy_snapshot,
)
from rl.rewards import (  # noqa: E402
    MedCLIPSemanticScorer,
    build_reward_pipeline_from_config,
)
from rl.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    append_jsonl,
    assert_new_output_dir,
    build_policy,
    build_rollout_replayer,
    decoder_inputs,
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
        "--smoke-one-step",
        action="store_true",
        help="Run exactly one optimizer step then exit.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Disabled by default; pass an explicit checkpoint path only when intended.",
    )
    return parser.parse_args()


def _training_memory_knobs(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "replay_microbatch_size": int(train_cfg.get("replay_microbatch_size", 1)),
        "gradient_replay_use_cache": bool(
            train_cfg.get("gradient_replay_use_cache", False)
        ),
        "gradient_checkpointing": bool(
            train_cfg.get("gradient_checkpointing", False)
        ),
    }


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
    memory_knobs = _training_memory_knobs(train_cfg)
    assert_production_cached_replay_checkpointing_disabled(
        replay_backend="live_production_cached_autograd",
        gradient_checkpointing=memory_knobs["gradient_checkpointing"],
    )
    max_steps = int(
        args.max_optimizer_steps
        if args.max_optimizer_steps is not None
        else train_cfg["max_optimizer_steps"]
    )
    if args.smoke_one_step:
        max_steps = 1
    output_dir = assert_new_output_dir(args.output_dir)
    write_json(output_dir / "resolved_config.json", config)

    pairs = load_verified_pairs(config, "train")
    device = torch.device(args.device)
    reset_peak_memory(device)
    model, tokenizer, processor, revision = build_policy(config, device)
    trainability = assert_approved_trainable_parameters(model)
    semantics = replay_semantics_report(model, config)
    reference_projector = projector_state_cpu(model)
    old_snapshot = PolicySnapshot.capture(model, optimizer_step=0)
    memory_stages: Dict[str, Any] = {"replay_semantics": semantics}
    record_stage_peak(memory_stages, "after_policy_load", device)

    optimizer = build_optimizer(
        model,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        use_8bit_adam=bool(train_cfg["use_8bit_adam"]),
        projector_lr=float(train_cfg["projector_learning_rate"]),
    )
    scorer = MedCLIPSemanticScorer(device=device)
    rewards = build_reward_pipeline_from_config(config, scorer)
    replayer = build_rollout_replayer(model, tokenizer, config)
    clip_epsilon = float(config["objective"]["ppo_clip_epsilon"])
    group_size = int(config["objective"]["group_size"])
    sync_interval = int(config["objective"]["old_policy_sync_interval_optimizer_steps"])
    if sync_interval != 1:
        raise RuntimeError("old-policy sync interval must be 1")
    record_stage_peak(memory_stages, "after_aux_init", device)

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
            "memory_knobs": memory_knobs,
            "replay_semantics": semantics,
            "reference_is_metric_only": True,
            "reference_in_loss": False,
        },
    )

    while optimizer_step < max_steps:
        epoch += 1
        for sample_index, pair in enumerate(pairs):
            if optimizer_step >= max_steps:
                break
            step_stages: Dict[str, Any] = dict(memory_stages)
            inputs = tokenize_rl_pair(processor, pair, device, config=config)
            reset_peak_memory(device)
            with use_policy_snapshot(model, old_snapshot):
                with torch.no_grad():
                    traces = generate_rollout_group(
                        model,
                        tokenizer,
                        inputs,
                        config,
                        sample_seed=int(train_cfg["seed"])
                        + sample_index
                        + epoch * 100000,
                    )
            record_stage_peak(step_stages, "after_rollout_generation", device)

            components = [
                rewards.score_from_trace(trace, pair) for trace in traces
            ]
            advantages = group_relative_advantages(
                [component.total_reward for component in components]
            ).to(device)
            record_stage_peak(step_stages, "after_rewards_and_advantages", device)

            # Free MedCLIP GPU residency before gradient-bearing replay.
            if hasattr(scorer, "model"):
                scorer.model.to("cpu")
                release_cuda_temporaries(empty_cache=True, device=device)
                record_stage_peak(step_stages, "after_medclip_cpu_offload", device)

            step_result = memory_safe_grpo_optimizer_step(
                model=model,
                optimizer=optimizer,
                replayer=replayer,
                traces=traces,
                advantages=advantages,
                decoder_kwargs=decoder_inputs(inputs),
                old_snapshot=old_snapshot,
                reference_snapshot=None,
                clip_epsilon=clip_epsilon,
                max_grad_norm=float(train_cfg["max_grad_norm"]),
                group_size=group_size,
                replay_microbatch_size=memory_knobs["replay_microbatch_size"],
                gradient_replay_use_cache=memory_knobs["gradient_replay_use_cache"],
                gradient_checkpointing=memory_knobs["gradient_checkpointing"],
                score_reference=False,
                assert_init_ratios=(optimizer_step == 0),
                memory_stages=step_stages,
                empty_cache_at_stage_boundaries=False,
            )
            optimizer_step += 1
            old_snapshot = PolicySnapshot.capture(
                model, optimizer_step=optimizer_step
            )

            # Restore MedCLIP for the next reward pass.
            if hasattr(scorer, "model"):
                scorer.model.to(device)

            peak_values = [
                float(v.get("peak_allocated_mb", 0.0))
                for v in step_stages.values()
                if isinstance(v, dict) and "peak_allocated_mb" in v
            ]
            step_stages["overall_peak_allocated_mb"] = (
                max(peak_values) if peak_values else 0.0
            )
            step_stages["peak_reserved_mb"] = peak_reserved_mb(device)
            step_stages["current_allocated_mb_end"] = current_allocated_mb(device)
            write_json(output_dir / "cuda_memory_stages.json", step_stages)

            records = [
                {
                    **trace_record(
                        trace,
                        sample_index=sample_index,
                        group_index=group_index,
                        pair=pair,
                        reward=component.to_dict(),
                        advantage=float(advantages[group_index]),
                    ),
                    "optimizer_step": optimizer_step,
                    "old_log_prob": step_result["old_logps"][group_index],
                    "current_log_prob": step_result["current_logps"][group_index],
                    "loss_grpo": step_result["per_rollout_losses"][group_index],
                    "ppo_ratio": step_result["per_rollout_ratios"][group_index],
                    "supervised_loss": None,
                }
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
                        "loss_grpo": step_result["loss_grpo"],
                        "loss_total": step_result["loss_total"],
                        "supervised_loss": None,
                        "mean_reward": float(
                            sum(c.total_reward for c in components) / len(components)
                        ),
                        "mean_advantage": float(advantages.detach().cpu().mean()),
                        "old_policy_synced": True,
                        "activation_dtype": step_result["activation_dtype"],
                        "replay_microbatch_size": 1,
                        "gradient_replay_use_cache": False,
                        "gradient_checkpointing": memory_knobs[
                            "gradient_checkpointing"
                        ],
                        "reference_in_loss": False,
                        "one_optimizer_step_completed": True,
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
            "reference_is_metric_only": True,
            "memory_knobs": memory_knobs,
            "replay_semantics": semantics,
        },
    )
    print(f"Training artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
