#!/usr/bin/env python3
"""Replay-only exactness and memory oracle for functional NTP checkpointing.

This script never constructs an optimizer and never updates parameters.  It is
not imported by the production trainer.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from rl.exact_kv_checkpoint import functional_kv_layer_checkpointing  # noqa: E402
from rl.grpo import grpo_clipped_loss  # noqa: E402
from rl.nan_grad_diagnostics import compare_grad_dicts  # noqa: E402
from rl.ntp_rl import NTP_ONLY_DECODER_PATH, validate_ntp_only_trace  # noqa: E402
from rl.pbd_rl import (  # noqa: E402
    BlockTrace,
    RolloutTrace,
    SlotTrace,
    _cache_length,
    _forward_language_model,
    _truncate_legacy_cache,
    _unwrap_lm_output,
    build_filtered_categorical,
)
from rl.runtime import (  # noqa: E402
    build_policy_two_gpu_live_cache,
    build_rollout_decoder,
    build_rollout_replayer,
    decoder_inputs,
    load_verified_pairs,
    tokenize_rl_pair,
    write_json,
)
from two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_production import (  # noqa: E402
    DEFAULT_SPEC,
    validate_sampling_ablation_contract,
)
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--mode", choices=("exactness", "memory", "preflight"), required=True
    )
    parser.add_argument("--backend", choices=("baseline", "checkpoint"))
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--sample-index", type=int, default=2)
    return parser.parse_args()


def _sync(devices: List[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(int(device.index or 0))


def _reset_peaks(devices: List[torch.device]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(int(device.index or 0))


def _memory(devices: List[torch.device]) -> Dict[str, Any]:
    _sync(devices)
    return {
        str(device): {
            "allocated": int(torch.cuda.memory_allocated(int(device.index or 0))),
            "reserved": int(torch.cuda.memory_reserved(int(device.index or 0))),
            "max_memory_allocated": int(
                torch.cuda.max_memory_allocated(int(device.index or 0))
            ),
            "max_memory_reserved": int(
                torch.cuda.max_memory_reserved(int(device.index or 0))
            ),
            "capacity": int(torch.cuda.get_device_properties(device).total_memory),
        }
        for device in devices
    }


def _lora_parameters(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "lora_" in name.lower()
    ]


def _projector_parameters(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name == "mlp1" or name.startswith("mlp1.")
    ]


def _collect_lora_grads(model) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in _lora_parameters(model)
        if parameter.grad is not None
    }


def _gradient_contract(model) -> Dict[str, Any]:
    lora = _lora_parameters(model)
    projector = _projector_parameters(model)
    present = [(name, parameter.grad) for name, parameter in lora if parameter.grad is not None]
    return {
        "lora_parameter_tensors": len(lora),
        "lora_gradient_tensors_present": len(present),
        "lora_gradient_tensors_finite": sum(
            bool(torch.isfinite(gradient).all().item()) for _, gradient in present
        ),
        "projector_parameter_tensors": len(projector),
        "projector_trainable_tensors": sum(parameter.requires_grad for _, parameter in projector),
        "projector_gradient_tensors_present": sum(
            parameter.grad is not None for _, parameter in projector
        ),
    }


def _clear_graph(model) -> None:
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def _fixed_length_ntp_trace(
    model,
    tokenizer,
    config: Dict[str, Any],
    inputs: Dict[str, Any],
    token_count: int,
) -> RolloutTrace:
    """Create a fixed-action synthetic NTP path using the exact top-p support."""
    decoder = build_rollout_decoder(model, tokenizer, config)
    if getattr(decoder, "decoding_mode", None) != NTP_ONLY_DECODER_PATH:
        raise RuntimeError("oracle requires the pure-NTP decoder")
    kwargs = decoder_inputs(inputs)
    input_ids = kwargs["input_ids"]
    generated = input_ids.clone()
    prompt_length = int(input_ids.size(1))
    full_positions = torch.arange(
        prompt_length + int(token_count) + 1, device=input_ids.device
    ).unsqueeze(0)
    visual_features = decoder._visual_features(
        kwargs["pixel_values"], kwargs["image_grid_hws"]
    )
    past_key_values = None
    blocks = []

    with torch.no_grad():
        for block_index in range(int(token_count)):
            prefix_length = int(generated.size(1))
            cache_before = _cache_length(past_key_values)
            prepared = model.language_model.prepare_inputs_for_generation(
                generated,
                past_key_values,
                None,
                inputs_embeds=None,
                use_cache=True,
                position_ids=full_positions[:, cache_before:prefix_length],
            )
            outputs = _unwrap_lm_output(
                _forward_language_model(
                    model,
                    prepared,
                    visual_features=visual_features if block_index == 0 else None,
                )
            )
            logits = outputs.logits[0, -1, :].float()
            distribution = build_filtered_categorical(
                logits,
                history_ids=generated[0].tolist(),
                config=decoder.sampling,
                allowed_token_ids=None,
            )
            local_index = int(torch.argmax(distribution.log_probs).item())
            action = int(distribution.token_ids[local_index].item())
            old_log_prob = float(distribution.log_probs[local_index].item())
            past_key_values = _truncate_legacy_cache(
                outputs.past_key_values, prefix_length
            )
            position_ids = prepared.get("position_ids")
            blocks.append(
                BlockTrace(
                    block_index=block_index,
                    prefix_length=prefix_length,
                    cache_length_before=cache_before,
                    cache_length_after=_cache_length(past_key_values),
                    block_type="ntp",
                    position_ids=(
                        position_ids[0].detach().cpu().tolist()
                        if position_ids is not None
                        else full_positions[0, cache_before:prefix_length].cpu().tolist()
                    ),
                    input_window_ids=prepared["input_ids"][0].detach().cpu().tolist(),
                    action_token_ids=[action],
                    slots=[
                        SlotTrace(
                            slot_index=0,
                            action_token_id=action,
                            support_kind="full",
                            log_prob_old=old_log_prob,
                            support_size=int(distribution.token_ids.numel()),
                            top_k=int(decoder.sampling.top_k),
                            top_p=float(decoder.sampling.top_p),
                            temperature=float(decoder.sampling.temperature),
                        )
                    ],
                    scored_for_grpo=True,
                    source=NTP_ONLY_DECODER_PATH,
                    rejected_proposal_token_ids=None,
                )
            )
            generated = torch.cat(
                [
                    generated,
                    torch.tensor([[action]], device=generated.device, dtype=generated.dtype),
                ],
                dim=1,
            )
            del logits, distribution, outputs

    generated_ids = generated[0, prompt_length:].detach().cpu().tolist()
    trace = RolloutTrace(
        prompt_token_ids=input_ids[0].detach().cpu().tolist(),
        generated_token_ids=generated_ids,
        blocks=blocks,
        sampling=decoder.sampling,
        stopped_on_eos=False,
        truncated=True,
        decoded_text=tokenizer.decode(generated_ids, skip_special_tokens=False),
        decoder_path=NTP_ONLY_DECODER_PATH,
        reward_branch="none",
        committed_final_box_norm_1000=None,
        has_unambiguous_committed_box=False,
        fallback_triggered=False,
        rejected_pbd_proposals=[],
        stop_reason="synthetic_fixed_token_budget",
        max_new_tokens=int(token_count),
        max_reachable_generated_length=int(token_count),
        proposal_events=[],
    )
    validate_ntp_only_trace(trace)
    del visual_features, past_key_values, generated
    gc.collect()
    torch.cuda.empty_cache()
    return trace


def _run_replay(model, tokenizer, config, trace, kwargs, backend: str):
    replayer = build_rollout_replayer(model, tokenizer, config)
    with torch.no_grad():
        old_total, _ = replayer.score(
            trace,
            use_cache=True,
            legacy_nocache_masks=False,
            **kwargs,
        )
    old_total = old_total.detach()
    model.zero_grad(set_to_none=True)
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    gc.collect()
    torch.cuda.empty_cache()
    _reset_peaks(devices)
    checkpoint_report: Dict[str, Any] = {"enabled": False}
    context = (
        functional_kv_layer_checkpointing(model, enabled=True)
        if backend == "checkpoint"
        else nullcontext(checkpoint_report)
    )
    with context as checkpoint_report:
        current, _blocks, token_logps, token_metadata = (
            replayer.score_with_token_logprobs(
                trace,
                use_cache=True,
                legacy_nocache_masks=False,
                **kwargs,
            )
        )
        loss = grpo_clipped_loss(
            current.reshape(1),
            old_total.reshape(1).to(current.device, torch.float32),
            torch.ones(1, device=current.device, dtype=torch.float32),
            clip_epsilon=float(config["objective"]["ppo_clip_epsilon"]),
        )
        scaled_loss = loss / 8.0
        scaled_loss.backward()
        memory = _memory(devices)
        result = {
            "backend": backend,
            "current_logp": float(current.detach().float().cpu()),
            "old_logp": float(old_total.detach().float().cpu()),
            "loss": float(loss.detach().float().cpu()),
            "scaled_loss": float(scaled_loss.detach().float().cpu()),
            "per_token_logps": token_logps.detach().float().cpu().tolist(),
            "per_token_metadata": token_metadata,
            "gradient_contract": _gradient_contract(model),
            "memory": memory,
            "checkpoint": dict(checkpoint_report),
            "lora_grads": _collect_lora_grads(model),
        }
    return result


def _public_result(run: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in run.items() if key != "lora_grads"}


def _exactness_report(baseline, checkpointed) -> Dict[str, Any]:
    left = torch.tensor(baseline["per_token_logps"], dtype=torch.float64)
    right = torch.tensor(checkpointed["per_token_logps"], dtype=torch.float64)
    token_abs = (left - right).abs()
    gradients = compare_grad_dicts(
        baseline["lora_grads"], checkpointed["lora_grads"]
    )
    loss_abs = abs(float(baseline["loss"]) - float(checkpointed["loss"]))
    total_abs = abs(
        float(baseline["current_logp"]) - float(checkpointed["current_logp"])
    )
    contracts_ok = all(
        run["gradient_contract"]["lora_parameter_tensors"] == 504
        and run["gradient_contract"]["lora_gradient_tensors_present"] == 504
        and run["gradient_contract"]["lora_gradient_tensors_finite"] == 504
        and run["gradient_contract"]["projector_trainable_tensors"] == 0
        and run["gradient_contract"]["projector_gradient_tensors_present"] == 0
        for run in (baseline, checkpointed)
    )
    gradient_tight = bool(
        gradients["both_sides_all_finite"]
        and not gradients["missing_names"]
        and gradients["global_cosine_finite"] is not None
        and gradients["global_cosine_finite"] >= 0.99999
        and (
            gradients["max_abs_error_finite"] <= 1e-5
            or gradients["global_rel_l2_error_finite"] <= 1e-4
        )
    )
    passed = bool(
        left.numel() == right.numel()
        and (float(token_abs.max()) if token_abs.numel() else 0.0) <= 1e-6
        and total_abs <= 1e-6
        and loss_abs <= 1e-7
        and contracts_ok
        and gradient_tight
    )
    return {
        "passed": passed,
        "loss_absolute_error": loss_abs,
        "trajectory_logp_absolute_error": total_abs,
        "per_token_logp_max_absolute_error": (
            float(token_abs.max()) if token_abs.numel() else 0.0
        ),
        "per_token_logp_count": int(left.numel()),
        "gradient_comparison": gradients,
        "gradient_contracts_passed": contracts_ok,
        "tolerances": {
            "loss_atol": 1e-7,
            "logp_atol": 1e-6,
            "gradient_max_abs_or_global_rel": [1e-5, 1e-4],
            "gradient_global_cosine_min": 0.99999,
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("oracle requires exactly two visible CUDA devices")
    if args.mode == "memory" and args.backend is None:
        raise RuntimeError("--mode memory requires --backend")
    if args.mode == "preflight" and int(args.token_count) != 72:
        raise RuntimeError("checkpoint preflight requires exactly 72 generated tokens")
    if args.mode == "exactness":
        torch.use_deterministic_algorithms(True)
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    selected_backend = "checkpoint" if args.mode == "preflight" else args.backend
    report: Dict[str, Any] = {
        "format": "ntp_functional_kv_checkpoint_oracle_v1",
        "mode": args.mode,
        "backend": selected_backend,
        "token_count": int(args.token_count),
        "optimizer_constructed": False,
        "optimizer_step_called": False,
        "production_training_modified": False,
        "detach_kv": False,
        "truncated_bptt": False,
        "use_reentrant": False,
        "deterministic_algorithms": args.mode == "exactness",
    }
    model = None
    try:
        config = resolve_experiment_config(Path(args.config))
        validate_sampling_ablation_contract(config)
        report["scientific_contract"] = {
            "group_size": int(config["objective"]["group_size"]),
            "temperature": float(config["rollout"]["temperature"]),
            "top_p": float(config["rollout"]["top_p"]),
            "top_k": int(config["rollout"]["top_k"]),
            "repetition_penalty": float(config["rollout"]["repetition_penalty"]),
            "max_new_tokens": int(config["rollout"]["max_new_tokens"]),
            "detach_kv": bool(config["runtime_contract"]["detach_kv"]),
            "projector_trainable": bool(config["model"]["projector_trainable"]),
            "learning_rate": float(config["training"]["learning_rate"]),
        }
        model, tokenizer, processor, revision, shard = build_policy_two_gpu_live_cache(
            config, first_device=devices[0], second_device=devices[1]
        )
        model.eval()
        report["model_revision"] = revision
        report["shard_backend"] = shard.get("backend")
        pairs = load_verified_pairs(config, "train")
        pair = pairs[int(args.sample_index)]
        inputs = tokenize_rl_pair(processor, pair, devices[0], config=config)
        kwargs = decoder_inputs(inputs)
        trace = _fixed_length_ntp_trace(
            model, tokenizer, config, inputs, int(args.token_count)
        )
        report["trace"] = {
            "sample_index": int(args.sample_index),
            "prompt_token_count": len(trace.prompt_token_ids),
            "generated_token_count": len(trace.generated_token_ids),
            "block_count": len(trace.blocks),
            "synthetic": True,
            "fixed_actions_all_selected_from_exact_top_p_support": True,
        }
        if args.mode == "exactness":
            baseline = _run_replay(
                model, tokenizer, config, trace, kwargs, "baseline"
            )
            report["baseline"] = _public_result(baseline)
            _clear_graph(model)
            baseline_repeat = _run_replay(
                model, tokenizer, config, trace, kwargs, "baseline"
            )
            report["baseline_repeat"] = _public_result(baseline_repeat)
            report["baseline_repeatability"] = _exactness_report(
                baseline, baseline_repeat
            )
            _clear_graph(model)
            checkpointed = _run_replay(
                model, tokenizer, config, trace, kwargs, "checkpoint"
            )
            report["checkpointed"] = _public_result(checkpointed)
            report["comparison"] = _exactness_report(baseline, checkpointed)
            report["status"] = (
                "passed_exactness_oracle"
                if report["comparison"]["passed"]
                else "failed_exactness_oracle"
            )
        else:
            try:
                run = _run_replay(
                    model, tokenizer, config, trace, kwargs, str(selected_backend)
                )
                report["run"] = _public_result(run)
                if args.mode == "preflight":
                    gpu0_peak = int(
                        run["memory"]["cuda:0"]["max_memory_allocated"]
                    )
                    gradient_contract = run["gradient_contract"]
                    report["preflight_gate"] = {
                        "gpu0_peak_allocated": gpu0_peak,
                        "gpu0_peak_limit": 15 * 2**30,
                        "gpu0_peak_below_15_gib": gpu0_peak < 15 * 2**30,
                        "lora_gradients_present": gradient_contract[
                            "lora_gradient_tensors_present"
                        ],
                        "lora_gradients_finite": gradient_contract[
                            "lora_gradient_tensors_finite"
                        ],
                        "projector_gradients_present": gradient_contract[
                            "projector_gradient_tensors_present"
                        ],
                    }
                    passed = bool(
                        report["preflight_gate"]["gpu0_peak_below_15_gib"]
                        and gradient_contract["lora_parameter_tensors"] == 504
                        and gradient_contract["lora_gradient_tensors_present"] == 504
                        and gradient_contract["lora_gradient_tensors_finite"] == 504
                        and gradient_contract["projector_gradient_tensors_present"] == 0
                    )
                    report["status"] = (
                        "passed_checkpoint_preflight"
                        if passed
                        else "failed_checkpoint_preflight"
                    )
                else:
                    report["status"] = "completed"
            except torch.OutOfMemoryError as exc:
                report["status"] = "cuda_oom"
                report["error"] = str(exc)
                report["memory_at_oom"] = _memory(devices)
        write_json(output, report)
    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        if torch.cuda.is_available():
            try:
                report["memory_at_error"] = _memory(devices)
            except Exception:
                pass
        write_json(output, report)
        raise
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
