#!/usr/bin/env python3
"""One exact production trajectory forward/backward on a 2x RTX 3090 shard.

This is deliberately a feasibility probe, not a training entrypoint: it makes
one no-grad rollout, replays that exact trajectory with ``carry_cache=True``,
backpropagates ``L_GRPO / G`` once, writes diagnostics, and never constructs or
steps an optimizer.
"""

from __future__ import annotations

import argparse
import json
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

from rl.grpo import grpo_clipped_loss  # noqa: E402
from rl.grpo_train_step import assert_grpo_loss_depends_only_on_trajectory_logprob  # noqa: E402
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    build_policy_two_gpu_live_cache,
    build_rollout_replayer,
    is_hybrid_rollout,
    load_resolved_config,
    load_verified_pairs,
    sampling_from_config,
    tokenize_rl_pair,
    write_json,
)
from rl.two_gpu_shard import (  # noqa: E402
    DecoderShardLayout,
    inspect_legacy_cache_devices,
    resolve_locateanything_qwen_decoder,
)
from rl.hybrid_rl import StochasticHybridRLDecoder  # noqa: E402
from rl.pbd_rl import StochasticPBDRLDecoder  # noqa: E402


OUTPUT_FILENAME = "two_gpu_live_cache_feasibility.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact one-trajectory two-GPU live-cache backward feasibility"
    )
    parser.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    return parser.parse_args()


def _memory(device: torch.device) -> Dict[str, float]:
    index = int(device.index or 0)
    return {
        "allocated_mb": float(torch.cuda.memory_allocated(index) / (1024**2)),
        "reserved_mb": float(torch.cuda.memory_reserved(index) / (1024**2)),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated(index) / (1024**2)),
        "peak_reserved_mb": float(torch.cuda.max_memory_reserved(index) / (1024**2)),
    }


def _reset_peaks(devices: List[torch.device]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(int(device.index or 0))


def _sync(devices: List[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(int(device.index or 0))


def _gradient_diagnostics(model: torch.nn.Module) -> Dict[str, Any]:
    result: Dict[str, Dict[str, Any]] = {}
    lora_nonzero = 0
    lora_total = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        device_key = str(parameter.device)
        record = result.setdefault(
            device_key,
            {
                "trainable_parameter_tensors": 0,
                "trainable_parameter_elements": 0,
                "gradient_tensors": 0,
                "finite_gradient_tensors": 0,
                "nonfinite_gradient_tensors": 0,
                "gradient_l2_squared": 0.0,
                "lora_gradient_tensors": 0,
                "lora_finite_gradient_tensors": 0,
            },
        )
        record["trainable_parameter_tensors"] += 1
        record["trainable_parameter_elements"] += int(parameter.numel())
        grad = parameter.grad
        if grad is None:
            continue
        record["gradient_tensors"] += 1
        finite = bool(torch.isfinite(grad).all().item())
        if finite:
            record["finite_gradient_tensors"] += 1
            record["gradient_l2_squared"] += float(grad.detach().float().square().sum().item())
        else:
            record["nonfinite_gradient_tensors"] += 1
        if "lora_" in name:
            lora_total += 1
            record["lora_gradient_tensors"] += 1
            if finite:
                lora_nonzero += int(bool(torch.count_nonzero(grad).item()))
                record["lora_finite_gradient_tensors"] += 1
    for record in result.values():
        record["gradient_l2"] = math.sqrt(record.pop("gradient_l2_squared"))
        record["all_present_gradients_finite"] = record["nonfinite_gradient_tensors"] == 0
    return {
        "per_device": result,
        "lora_gradient_tensors": lora_total,
        "lora_nonzero_gradient_tensors": lora_nonzero,
        "all_lora_gradients_present_and_finite": lora_total > 0
        and lora_total == sum(
            entry["lora_finite_gradient_tensors"] for entry in result.values()
        ),
    }


class _ReplayKVRecorder:
    """Observe actual cache returns from replay scorer blocks, never model hooks."""

    def __init__(self, layout: DecoderShardLayout) -> None:
        self.layout = layout
        self.records: List[Dict[str, Any]] = []
        self._previous_next_cache = None

    def record(self, scored_index, block, prior_cache, next_cache) -> None:
        returned = inspect_legacy_cache_devices(next_cache, self.layout)
        prior = (
            inspect_legacy_cache_devices(prior_cache, self.layout)
            if prior_cache is not None
            else None
        )
        carried_identity = (
            prior_cache is not None and prior_cache is self._previous_next_cache
        )
        prior_lengths = (
            [item.get("sequence_length") for item in prior["layers"]]
            if prior is not None
            else []
        )
        returned_lengths = [item.get("sequence_length") for item in returned["layers"]]
        self.records.append(
            {
                "replay_scored_block_index": int(scored_index),
                "trace_block_index": int(block.block_index),
                "prior_cache_present": prior_cache is not None,
                "prior_cache_is_previous_returned_cache": carried_identity,
                "prior_cache_sequence_lengths": prior_lengths,
                "returned_cache_sequence_lengths": returned_lengths,
                "returned_cache": returned,
            }
        )
        self._previous_next_cache = next_cache

    def report(self) -> Dict[str, Any]:
        returned = [record["returned_cache"] for record in self.records]
        all_layer_placement = bool(returned) and all(
            item["all_layers_live_and_colocated"] for item in returned
        )
        later_records = self.records[1:]
        later_carry = all(
            item["prior_cache_present"]
            and item["prior_cache_is_previous_returned_cache"]
            for item in later_records
        )
        all_layers = [layer for item in returned for layer in item["layers"]]
        early_live = any(
            layer.get("layer_index", 36) < 18
            and layer.get("key_requires_grad")
            and layer.get("key_grad_fn") is not None
            for layer in all_layers
        )
        late_live = any(
            layer.get("layer_index", -1) >= 18
            and layer.get("key_requires_grad")
            and layer.get("key_grad_fn") is not None
            for layer in all_layers
        )
        return {
            "replay_forward_count": len(self.records),
            "replay_block_forward_count": len(self.records),
            "later_replay_block_count": len(later_records),
            "later_replay_blocks_carried_prior_cache": later_carry,
            "early_trainable_path_has_live_kv": early_live,
            "late_trainable_path_has_live_kv": late_live,
            "all_replay_caches_live_and_colocated": (
                all_layer_placement and later_carry and early_live and late_live
            ),
            "replay_blocks": self.records,
        }


def _cross_device_boundary_report(model) -> Dict[str, Any]:
    resolved = resolve_locateanything_qwen_decoder(model)
    events = list(getattr(resolved.decoder, "_chestxray8_two_gpu_boundary_events", []))
    lora_by_side: Dict[str, Any] = {"early": None, "late": None}
    for name, parameter in model.named_parameters():
        if "lora_" not in name or parameter.grad is None:
            continue
        layer_index = next((index for index in range(36) if f"layers.{index}." in name), None)
        if layer_index is None:
            continue
        side = "early" if layer_index < 18 else "late"
        if lora_by_side[side] is None:
            lora_by_side[side] = {
                "name": name,
                "layer_index": layer_index,
                "device": str(parameter.device),
                "gradient_finite": bool(torch.isfinite(parameter.grad).all().item()),
            }
    return {
        "replay_boundary_transfer_count": len(events),
        "events": events,
        "all_transfers_cross_cuda_0_to_cuda_1_with_autograd": bool(events)
        and all(
            event["hidden_before_device"] == "cuda:0"
            and event["hidden_after_device"] == "cuda:1"
            and event["hidden_after_requires_grad"]
            and event["hidden_after_grad_fn"] is not None
            for event in events
        ),
        "early_lora_gradient": lora_by_side["early"],
        "late_lora_gradient": lora_by_side["late"],
        "gradient_reaches_early_and_late_lora": (
            lora_by_side["early"] is not None and lora_by_side["late"] is not None
        ),
    }


def _one_rollout(model, tokenizer, inputs, config: Dict[str, Any], seed: int, max_new_tokens: int):
    sampling = sampling_from_config(config)
    if is_hybrid_rollout(config):
        decoder = StochasticHybridRLDecoder(
            model,
            tokenizer,
            sampling=sampling,
        )
    else:
        decoder = StochasticPBDRLDecoder(model, tokenizer, sampling=sampling)
    return decoder.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_grid_hws=inputs["image_grid_hws"],
        max_new_tokens=max_new_tokens,
        seed=seed,
        force_first_box_block=False,
    )


def _decoder_argument_report(model) -> Dict[str, Any]:
    resolved = resolve_locateanything_qwen_decoder(model)
    decoder = resolved.decoder
    calls = list(getattr(decoder, "_chestxray8_two_gpu_argument_reports", []))
    return {
        "resolved_decoder_path": resolved.decoder_path,
        "original_signature": getattr(
            decoder, "_chestxray8_two_gpu_original_forward_signature", None
        ),
        "call_count": len(calls),
        "calls": calls,
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
        "probe": "one_production_trajectory_forward_backward_only",
        "output_filename": OUTPUT_FILENAME,
        "optimizer_constructed": False,
        "optimizer_step_called": False,
        "group_size_for_loss_scaling": 4,
        "objective": "L_total = L_GRPO only",
        "full_trajectory_objective": True,
        "carry_cache": True,
        "detached_kv": False,
        "truncated_bptt": False,
        "bfix": False,
        "gradient_checkpointing": False,
        "attn_implementation": "sdpa",
        "visible_devices_required": ["cuda:0", "cuda:1"],
    }
    recorder = None
    model = None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError(
                "expected exactly two visible CUDA devices; set CUDA_VISIBLE_DEVICES=2,3"
            )
        config = load_resolved_config(args.config)
        if config["objective"]["loss_total"] != "L_GRPO" or int(
            config["objective"]["group_size"]
        ) != 4:
            raise RuntimeError("probe requires L_total=L_GRPO and G=4")
        if not bool((config.get("rollout", {}).get("hybrid") or {}).get("score_rejected_pbd_proposals", True)) and is_hybrid_rollout(config):
            raise RuntimeError("Hybrid probe requires the exact full_trajectory objective")
        if bool(config["training"].get("gradient_checkpointing", False)):
            raise RuntimeError("production live-cache probe requires gradient_checkpointing=false")
        assert_grpo_loss_depends_only_on_trajectory_logprob()
        report["config_path"] = config["_config_path"]
        report["model_name"] = config["model"]["name_or_path"]
        report["model_revision"] = config["model"]["revision"]
        report["rollout_path"] = config["rollout"]["path"]
        report["hybrid_logprob_objective"] = (
            (config.get("rollout", {}).get("hybrid") or {}).get("logprob_objective")
        )

        def record_pre_forward_layout(layout_report: Dict[str, Any]) -> None:
            report["pre_forward_layout"] = layout_report

        model, tokenizer, processor, revision, shard_report = build_policy_two_gpu_live_cache(
            config,
            first_device=devices[0],
            second_device=devices[1],
            on_pre_forward_layout=record_pre_forward_layout,
        )
        model.eval()
        report["model_revision_loaded"] = revision
        report["shard"] = shard_report
        pairs = load_verified_pairs(config, "train")
        if args.sample_index < 0 or args.sample_index >= len(pairs):
            raise IndexError(f"sample index {args.sample_index} outside train split")
        pair = pairs[args.sample_index]
        inputs = tokenize_rl_pair(processor, pair, devices[0], config=config)
        max_new_tokens = (
            int(args.max_new_tokens)
            if args.max_new_tokens is not None
            else int(config["rollout"]["max_new_tokens"])
        )
        report["sample"] = {
            "sample_index": args.sample_index,
            "image_index": pair["image_index"],
            "patient_id": pair["patient_id"],
            "disease": pair["disease"],
            "max_new_tokens": max_new_tokens,
        }

        _reset_peaks(devices)
        trace = _one_rollout(model, tokenizer, inputs, config, args.seed, max_new_tokens)
        _sync(devices)
        report["decoder_forward_arguments_after_rollout"] = _decoder_argument_report(model)
        report["rollout"] = {
            "block_count": len(trace.blocks),
            "scored_block_count": sum(bool(block.scored_for_grpo) for block in trace.blocks),
            "generated_token_count": len(trace.generated_token_ids),
            "decoder_path": getattr(trace, "decoder_path", "pbd"),
        }
        report["memory_after_rollout"] = {str(device): _memory(device) for device in devices}

        replayer = build_rollout_replayer(model, tokenizer, config)
        decoder_kwargs = {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs["pixel_values"],
            "image_grid_hws": inputs["image_grid_hws"],
        }
        with torch.no_grad():
            old_logp, _ = replayer.score(
                trace, use_cache=True, legacy_nocache_masks=False, **decoder_kwargs
            )
        old_logp = old_logp.detach()
        _sync(devices)
        report["memory_after_old_policy_forward"] = {
            str(device): _memory(device) for device in devices
        }

        model.zero_grad(set_to_none=True)
        _reset_peaks(devices)
        layout = DecoderShardLayout(devices[0], devices[1], 18)
        resolved_decoder = resolve_locateanything_qwen_decoder(model).decoder
        # Count only the gradient-bearing replay forwards below, not rollout or
        # no-grad old-policy scoring.
        resolved_decoder._chestxray8_two_gpu_boundary_events = []
        recorder = _ReplayKVRecorder(layout)
        current_logp, _ = replayer.score(
            trace,
            use_cache=True,
            legacy_nocache_masks=False,
            on_scored_block_cache=recorder.record,
            **decoder_kwargs,
        )
        loss = grpo_clipped_loss(
            current_logp.reshape(1),
            old_logp.reshape(1).to(current_logp.device, torch.float32),
            torch.ones(1, device=current_logp.device, dtype=torch.float32),
            clip_epsilon=float(config["objective"]["ppo_clip_epsilon"]),
        )
        (loss / 4.0).backward()
        _sync(devices)
        recorder_report = recorder.report()
        recorder = None
        report["loss_grpo"] = float(loss.detach().float().cpu())
        report["current_logp"] = float(current_logp.detach().float().cpu())
        report["old_logp"] = float(old_logp.detach().float().cpu())
        report["memory_after_backward"] = {str(device): _memory(device) for device in devices}
        report["finite_gradients"] = _gradient_diagnostics(model)
        report["live_kv"] = recorder_report
        report["cross_device_boundary"] = _cross_device_boundary_report(model)
        report["status"] = (
            "passed"
            if recorder_report["all_replay_caches_live_and_colocated"]
            and report["finite_gradients"]["all_lora_gradients_present_and_finite"]
            and report["cross_device_boundary"]["all_transfers_cross_cuda_0_to_cuda_1_with_autograd"]
            and report["cross_device_boundary"]["gradient_reaches_early_and_late_lora"]
            else "failed_diagnostics"
        )
    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        if model is not None:
            try:
                report["decoder_forward_arguments_at_error"] = _decoder_argument_report(model)
            except Exception as report_exc:  # noqa: BLE001
                report["decoder_forward_argument_report_error"] = str(report_exc)
        if torch.cuda.is_available():
            report["memory_at_error"] = {str(device): _memory(device) for device in devices}
        write_json(output_path, report)
        raise
    write_json(output_path, report)
    print(json.dumps({"status": report["status"], "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
