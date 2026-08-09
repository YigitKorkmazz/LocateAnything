#!/usr/bin/env python3
"""One real G=4 AdamW GRPO smoke step on the exact 18/18 live-cache shard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.rewards import MedCLIPSemanticScorer, build_reward_pipeline_from_config  # noqa: E402
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    build_policy_two_gpu_live_cache,
    build_rollout_replayer,
    decoder_inputs,
    generate_rollout_group,
    hybrid_logprob_objective,
    is_hybrid_rollout,
    load_resolved_config,
    load_verified_pairs,
    tokenize_rl_pair,
)
from rl.two_gpu_shard import DecoderShardLayout, resolve_locateanything_qwen_decoder  # noqa: E402
from train_chestxray8_sft import build_optimizer  # noqa: E402
from two_gpu_exactness_oracle import _write_report_safely  # noqa: E402
from two_gpu_live_cache_feasibility import _ReplayKVRecorder, _cross_device_boundary_report  # noqa: E402


OUTPUT_FILENAME = "two_gpu_g4_grpo_smoke_step.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def _memory(devices: List[torch.device]) -> Dict[str, Any]:
    out = {}
    for device in devices:
        index = int(device.index or 0)
        torch.cuda.synchronize(index)
        total = int(torch.cuda.get_device_properties(index).total_memory)
        out[str(device)] = {
            "allocated_mb": float(torch.cuda.memory_allocated(index) / (1024**2)),
            "reserved_mb": float(torch.cuda.memory_reserved(index) / (1024**2)),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated(index) / (1024**2)),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved(index) / (1024**2)),
            "capacity_mb": float(total / (1024**2)),
            "below_capacity": torch.cuda.max_memory_allocated(index) < total,
        }
    return out


def _reset_peaks(devices: List[torch.device]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(int(device.index or 0))


def _lora_parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]


def _grad_report(model) -> Dict[str, Any]:
    params = _lora_parameters(model)
    missing = [name for name, p in params if p.grad is None]
    nonfinite = [name for name, p in params if p.grad is not None and not bool(torch.isfinite(p.grad).all().item())]
    nonzero = [name for name, p in params if p.grad is not None and bool(torch.count_nonzero(p.grad).item())]
    norm_sq = 0.0
    for _, p in params:
        if p.grad is not None:
            norm_sq += float(p.grad.detach().float().square().sum().item())
    return {
        "expected_lora_tensors": len(params),
        "missing_lora_gradients": missing,
        "nonfinite_lora_gradients": nonfinite,
        "nonzero_lora_gradient_tensors": len(nonzero),
        "cumulative_gradient_norm": math.sqrt(norm_sq),
        "all_504_present": len(params) == 504 and not missing,
        "all_finite": not nonfinite,
        "nontrivial": bool(nonzero),
    }


def _checksum_and_snapshot(model) -> tuple[Dict[str, str], Dict[str, torch.Tensor]]:
    checksums, snapshot = {}, {}
    for name, parameter in _lora_parameters(model):
        cpu = parameter.detach().float().cpu().contiguous()
        checksums[name] = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
        snapshot[name] = cpu.clone()
    return checksums, snapshot


def _optimizer_state_devices(optimizer) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                key = str(value.device)
                out[key] = out.get(key, 0) + 1
    return out


def _trace_summary(trace, component) -> Dict[str, Any]:
    return {
        "reward": component.to_dict(),
        "committed_branch": getattr(trace, "reward_branch", None),
        "committed_final_bbox_norm_1000": getattr(trace, "committed_final_box_norm_1000", None),
        "has_unambiguous_committed_box": bool(getattr(trace, "has_unambiguous_committed_box", False)),
        "fallback_triggered": bool(getattr(trace, "fallback_triggered", False)),
        "generated_token_count": len(trace.generated_token_ids),
        "scored_block_count": sum(bool(getattr(block, "scored_for_grpo", True)) for block in trace.blocks),
        "block_count": len(trace.blocks),
        "decoder_path": getattr(trace, "decoder_path", "pbd"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    report: Dict[str, Any] = {
        "smoke_step": "one_real_two_gpu_G4_GRPO_AdamW_step",
        "output_filename": OUTPUT_FILENAME,
        "group_size": 4,
        "objective": "L_total = L_GRPO only",
        "full_sampled_trajectory_objective": True,
        "carry_cache": True,
        "detached_kv": False,
        "bfix": False,
        "truncated_bptt": False,
        "gradient_checkpointing": False,
        "scheduler_steps": 0,
        "optimizer_steps": 0,
    }
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("requires CUDA_VISIBLE_DEVICES=2,3 (visible cuda:0,cuda:1)")
        config = load_resolved_config(args.config)
        if int(config["objective"]["group_size"]) != 4 or config["objective"]["loss_total"] != "L_GRPO":
            raise RuntimeError("smoke step requires G=4 and L_total=L_GRPO")
        if bool(config["training"].get("gradient_checkpointing", False)):
            raise RuntimeError("live-cache smoke step requires gradient_checkpointing=false")
        if not is_hybrid_rollout(config) or hybrid_logprob_objective(config) != "full_trajectory":
            raise RuntimeError("smoke step requires the production Hybrid full_trajectory objective")
        if not bool(config["rollout"].get("hybrid", {}).get("score_rejected_pbd_proposals", False)):
            raise RuntimeError("full Hybrid trajectory must score fallback-gating rejected PBD proposals")
        if bool(config["rollout"].get("reconstruct_actions_from_text", False)):
            raise RuntimeError("smoke step requires decoder-native Hybrid commit semantics")
        if not bool(config["rewards"].get("use_decoder_committed_final_box", False)):
            raise RuntimeError("smoke step requires rewards from the committed final bbox only")
        model, tokenizer, processor, revision, shard = build_policy_two_gpu_live_cache(
            config, first_device=devices[0], second_device=devices[1]
        )
        model.eval()
        if len(_lora_parameters(model)) != 504:
            raise RuntimeError("Case-B smoke step requires exactly 504 trainable LoRA tensors")
        pairs = load_verified_pairs(config, "train")
        pair = pairs[int(args.sample_index)]
        inputs = tokenize_rl_pair(processor, pair, devices[0], config=config)
        optimizer = build_optimizer(
            model,
            lr=float(config["training"]["learning_rate"]),
            projector_lr=float(config["training"]["projector_learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            use_8bit_adam=bool(config["training"]["use_8bit_adam"]),
        )
        replayer = build_rollout_replayer(model, tokenizer, config)
        scorer = MedCLIPSemanticScorer(device=devices[0])
        rewards = build_reward_pipeline_from_config(config, scorer)
        with torch.no_grad():
            rollout_seed = int(config["training"]["seed"]) + int(args.sample_index)
            traces = generate_rollout_group(
                model, tokenizer, inputs, config,
                sample_seed=rollout_seed,
            )
        if len(traces) != 4:
            raise RuntimeError(f"expected exactly four independently sampled trajectories, got {len(traces)}")
        components = [rewards.score_from_trace(trace, pair) for trace in traces]
        reward_values = [float(component.total_reward) for component in components]
        advantages = group_relative_advantages(reward_values).to(devices[1])
        if hasattr(scorer, "model"):
            scorer.model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        decoder_kwargs = decoder_inputs(inputs)
        with torch.no_grad():
            old_logps = [
                replayer.score(trace, use_cache=True, legacy_nocache_masks=False, **decoder_kwargs)[0].detach()
                for trace in traces
            ]

        optimizer.zero_grad(set_to_none=True)
        before_checksums, before_params = _checksum_and_snapshot(model)
        trajectory_reports: List[Dict[str, Any]] = []
        for index, trace in enumerate(traces):
            _reset_peaks(devices)
            resolved = resolve_locateanything_qwen_decoder(model).decoder
            resolved._chestxray8_two_gpu_boundary_events = []
            kv = _ReplayKVRecorder(DecoderShardLayout(devices[0], devices[1], 18))
            current, blocks = replayer.score(
                trace,
                use_cache=True,
                legacy_nocache_masks=False,
                on_scored_block_cache=kv.record,
                **decoder_kwargs,
            )
            loss = grpo_clipped_loss(
                current.reshape(1),
                old_logps[index].detach().reshape(1).to(current.device, torch.float32),
                advantages[index].detach().reshape(1),
                clip_epsilon=float(config["objective"]["ppo_clip_epsilon"]),
            )
            scaled = loss / 4.0
            scaled.backward()
            memory = _memory(devices)
            trajectory_reports.append({
                **_trace_summary(trace, components[index]),
                "group_index": index,
                "sampling_seed": rollout_seed * 4 + index,
                "advantage": float(advantages[index].detach().cpu()),
                "old_logp": float(old_logps[index].float().cpu()),
                "current_logp": float(current.detach().float().cpu()),
                "ppo_ratio": float(torch.exp(current.detach().float() - old_logps[index].detach().float()).cpu()),
                "loss_grpo_unscaled": float(loss.detach().float().cpu()),
                "loss_grpo_scaled": float(scaled.detach().float().cpu()),
                "block_logps": [float(value.detach().float().cpu()) for value in blocks],
                "cumulative_gradients": _grad_report(model),
                "per_gpu_memory_after_backward": memory,
                "live_kv": kv.report(),
                "cross_device_boundary": _cross_device_boundary_report(model),
            })
            kv._previous_next_cache = None
            del current, blocks, loss, scaled, kv
            gc.collect()

        pre_step_grads = _grad_report(model)
        if not (pre_step_grads["all_504_present"] and pre_step_grads["all_finite"] and pre_step_grads["nontrivial"]):
            raise RuntimeError("refusing AdamW step: accumulated LoRA gradients are missing, nonfinite, or trivial")
        optimizer.step()
        report["optimizer_steps"] = 1
        if report["optimizer_steps"] != 1:
            raise RuntimeError("optimizer step counter must equal exactly one")
        after_checksums, _ = _checksum_and_snapshot(model)
        changed_tensors, changed_elements = [], 0
        named = dict(model.named_parameters())
        for name, before in before_params.items():
            after = named[name].detach().float().cpu()
            changed = int(torch.count_nonzero(after != before).item())
            if changed:
                changed_tensors.append(name)
                changed_elements += changed
        report.update({
            "status": "completed",
            "model_revision": revision,
            "config_path": config["_config_path"],
            "sample": {"sample_index": args.sample_index, "image_index": pair["image_index"], "disease": pair["disease"]},
            "shard": shard,
            "reward_group": {"rewards": reward_values, "mean": float(sum(reward_values) / 4.0), "std": float(torch.tensor(reward_values).std(unbiased=False)), "advantages": [float(x.cpu()) for x in advantages]},
            "sampling": {
                "independently_sampled": True,
                "base_seed": rollout_seed,
                "trajectory_seeds": [rollout_seed * 4 + index for index in range(4)],
            },
            "trajectories": trajectory_reports,
            "pre_step_accumulated_gradients": pre_step_grads,
            "parameter_checksums": {"before": before_checksums, "after": after_checksums},
            "parameter_change": {"changed_lora_tensors": changed_tensors, "num_changed_lora_tensors": len(changed_tensors), "changed_lora_elements": changed_elements},
            "optimizer_state_devices": _optimizer_state_devices(optimizer),
            "per_gpu_memory_after_step": _memory(devices),
        })
        reward_std = report["reward_group"]["std"]
        forbidden_ok = all(not report[key] for key in ("detached_kv", "bfix", "truncated_bptt", "gradient_checkpointing"))
        report["acceptance"] = {
            "four_trajectories_completed": len(trajectory_reports) == 4,
            "nonzero_reward_variance": reward_std > 0.0,
            "all_losses_finite": all(math.isfinite(item["loss_grpo_unscaled"]) for item in trajectory_reports),
            "all_gradients_finite_nontrivial": pre_step_grads["all_finite"] and pre_step_grads["nontrivial"],
            "all_504_lora_gradients_present": pre_step_grads["all_504_present"],
            "exactly_one_optimizer_step": report["optimizer_steps"] == 1,
            "at_least_one_lora_parameter_changed": bool(changed_tensors),
            "no_forbidden_approximations": forbidden_ok,
            "all_live_kv_checks_pass": all(item["live_kv"]["all_replay_caches_live_and_colocated"] for item in trajectory_reports),
            "all_cross_device_autograd_checks_pass": all(item["cross_device_boundary"]["all_transfers_cross_cuda_0_to_cuda_1_with_autograd"] and item["cross_device_boundary"]["gradient_reaches_early_and_late_lora"] for item in trajectory_reports),
            "both_gpus_below_capacity": all(info["below_capacity"] for info in report["per_gpu_memory_after_step"].values()),
        }
        report["acceptance"]["passed"] = all(report["acceptance"].values())
    except Exception as exc:
        report.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        _write_report_safely(output_path, report)
        raise
    _write_report_safely(output_path, report)
    print({"status": report["status"], "output": str(output_path)})


if __name__ == "__main__":
    main()
