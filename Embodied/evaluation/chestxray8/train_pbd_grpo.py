#!/usr/bin/env python3
"""Full reward-only GRPO training entrypoint for stochastic native PBD/Hybrid."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.cuda_memory import record_stage_peak  # noqa: E402
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
from train_chestxray8_sft import (  # noqa: E402
    build_optimizer,
    load_lora_adapter_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Disabled by default; use only when explicitly requested.",
    )
    return parser.parse_args()


def _load_resume(
    checkpoint: Path,
    model,
    optimizer,
    device: torch.device,
) -> int:
    load_lora_adapter_weights(model, checkpoint / "adapter")
    model.mlp1.load_state_dict(
        torch.load(checkpoint / "mlp1.pt", map_location=device)
    )
    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location=device)
    )
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    return int(state["optimizer_step"])


def _reward_pipeline(config: Dict[str, Any], device: torch.device):
    return build_reward_pipeline_from_config(
        config, MedCLIPSemanticScorer(device=device)
    )


def main() -> None:
    args = parse_args()
    config = load_resolved_config(args.config)
    output_dir = assert_new_output_dir(args.output_dir)
    write_json(output_dir / "resolved_config.json", config)
    pairs = load_verified_pairs(config, "train")
    device = torch.device(args.device)
    model, tokenizer, processor, revision = build_policy(config, device)
    trainability = assert_approved_trainable_parameters(model)
    semantics = replay_semantics_report(model, config)

    reference_snapshot = PolicySnapshot.capture(model, optimizer_step=0)
    reference_projector = projector_state_cpu(model)
    training_cfg = config["training"]
    optimizer = build_optimizer(
        model,
        lr=float(training_cfg["learning_rate"]),
        projector_lr=float(training_cfg["projector_learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        use_8bit_adam=bool(training_cfg["use_8bit_adam"]),
    )
    optimizer_step = 0
    if args.resume_from_checkpoint:
        optimizer_step = _load_resume(
            Path(args.resume_from_checkpoint), model, optimizer, device
        )
    old_snapshot = PolicySnapshot.capture(model, optimizer_step=optimizer_step)
    rewards = _reward_pipeline(config, device)
    replayer = build_rollout_replayer(model, tokenizer, config)
    trace_path = output_dir / "train_rollout_traces.jsonl"
    metric_path = output_dir / "train_metrics.jsonl"
    epsilon = float(config["objective"]["ppo_clip_epsilon"])
    group_size = int(config["objective"]["group_size"])
    max_steps = int(training_cfg["max_optimizer_steps"])
    checkpoint_interval = int(training_cfg["checkpoint_interval"])
    seed = int(training_cfg["seed"])
    final_checkpoint = None
    memory_knobs = {
        "replay_microbatch_size": int(
            training_cfg.get("replay_microbatch_size", 1)
        ),
        "gradient_replay_use_cache": bool(
            training_cfg.get("gradient_replay_use_cache", False)
        ),
        "gradient_checkpointing": bool(
            training_cfg.get("gradient_checkpointing", False)
        ),
    }
    assert_production_cached_replay_checkpointing_disabled(
        replay_backend="live_production_cached_autograd",
        gradient_checkpointing=memory_knobs["gradient_checkpointing"],
    )

    for epoch in range(int(training_cfg["num_epochs"])):
        order = list(range(len(pairs)))
        random.Random(seed + epoch).shuffle(order)
        for sample_index in order:
            if optimizer_step >= max_steps:
                break
            pair = pairs[sample_index]
            inputs = tokenize_rl_pair(processor, pair, device, config=config)
            step_stages: Dict[str, Any] = {"replay_semantics": semantics}

            with use_policy_snapshot(model, old_snapshot), torch.no_grad():
                traces = generate_rollout_group(
                    model,
                    tokenizer,
                    inputs,
                    config,
                    sample_seed=seed + epoch * len(pairs) + sample_index,
                )
            record_stage_peak(step_stages, "after_rollout_generation", device)
            components = [
                rewards.score_from_trace(trace, pair) for trace in traces
            ]
            advantages = group_relative_advantages(
                [component.total_reward for component in components]
            ).to(device)
            record_stage_peak(step_stages, "after_rewards_and_advantages", device)

            step_result = memory_safe_grpo_optimizer_step(
                model=model,
                optimizer=optimizer,
                replayer=replayer,
                traces=traces,
                advantages=advantages,
                decoder_kwargs=decoder_inputs(inputs),
                old_snapshot=old_snapshot,
                reference_snapshot=reference_snapshot,
                clip_epsilon=epsilon,
                max_grad_norm=float(training_cfg["max_grad_norm"]),
                group_size=group_size,
                replay_microbatch_size=memory_knobs["replay_microbatch_size"],
                gradient_replay_use_cache=memory_knobs[
                    "gradient_replay_use_cache"
                ],
                gradient_checkpointing=memory_knobs["gradient_checkpointing"],
                score_reference=True,
                assert_init_ratios=(optimizer_step == 0),
                memory_stages=step_stages,
            )
            optimizer_step += 1
            old_snapshot = PolicySnapshot.capture(
                model, optimizer_step=optimizer_step
            )
            write_json(output_dir / "cuda_memory_stages.json", step_stages)
            reference_logps = step_result["reference_logps"] or [None] * group_size
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
                    "reference_log_prob": reference_logps[group_index],
                    "loss_grpo": step_result["per_rollout_losses"][group_index],
                    "supervised_loss": None,
                }
                for group_index, (trace, component) in enumerate(
                    zip(traces, components)
                )
            ]
            append_jsonl(trace_path, records)
            append_jsonl(
                metric_path,
                [
                    {
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "sample_index": sample_index,
                        "mean_reward": sum(
                            component.total_reward for component in components
                        )
                        / group_size,
                        "mean_grpo_loss": step_result["loss_grpo"],
                        "loss_total": "L_GRPO",
                        "supervised_loss": None,
                        "reference_in_loss": False,
                    }
                ],
            )
            if optimizer_step % checkpoint_interval == 0:
                final_checkpoint = save_rl_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    step=optimizer_step,
                    config=config,
                    reference_projector=reference_projector,
                )
        if optimizer_step >= max_steps:
            break

    if final_checkpoint is None or final_checkpoint.name != f"checkpoint-{optimizer_step}":
        final_checkpoint = save_rl_checkpoint(
            output_dir,
            model,
            optimizer,
            step=optimizer_step,
            config=config,
            reference_projector=reference_projector,
        )
    write_json(
        output_dir / "training_summary.json",
        {
            "optimizer_steps": optimizer_step,
            "model_revision": revision,
            "final_checkpoint": str(final_checkpoint),
            "loss_total": "L_GRPO",
            "supervised_losses": [],
            "trainable_tensor_count": len(trainability["trainable_names"]),
            "old_policy_sync_interval": 1,
            "reference_projector_fixed": True,
            "reference_is_metric_only": True,
            "memory_knobs": memory_knobs,
            "replay_semantics": semantics,
        },
    )


if __name__ == "__main__":
    main()
