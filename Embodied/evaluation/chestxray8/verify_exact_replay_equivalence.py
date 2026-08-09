#!/usr/bin/env python3
"""Manual GPU equivalence / CUDA-path diagnostics for Hybrid replay.

Do not run from the agent session. Supports full equivalence comparison and
isolated forward-only diagnostic modes to localize CUDA driver failures.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.grpo import grpo_clipped_loss  # noqa: E402
from rl.bfix_discrepancy import bfix_discrepancy_report  # noqa: E402
from rl.grpo_train_step import (  # noqa: E402
    BLOCK_LOGP_ATOL,
    PRODUCTION_REPLAY,
    assert_grpo_loss_depends_only_on_trajectory_logprob,
    assert_initialization_ratios,
    assert_truncated_bptt_not_thesis_default,
    replay_semantics_report,
)
from rl.grpo import group_relative_advantages  # noqa: E402
from train_chestxray8_sft import build_optimizer  # noqa: E402
from rl.pbd_rl import (  # noqa: E402
    BlockTrace,
    PBDSamplingConfig,
    RolloutTrace,
    SlotTrace,
    _cache_length,
)
from rl.policy_state import is_lora_name, is_projector_name  # noqa: E402
from rl.replay_memory import (  # noqa: E402
    _decoder_layers,
    eval_mode_layer_checkpointing,
    math_sdpa_for_grad_replay,
)  # noqa: E402
from rl.runtime import (  # noqa: E402
    build_policy,
    build_rollout_replayer,
    decoder_inputs,
    load_resolved_config,
    load_verified_pairs,
    tokenize_rl_pair,
    write_json,
)

VALID_MODES = (
    "full",
    "baseline-only",
    "pass-a-only",
    "pass-b-one-block",
    "compare-blocks",
    "grad-forward-only",
    "exact-safe-vs-production",
)

TRAINABILITY_CASES = ("A", "B", "C", "D")


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true|false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CHEST_DIR / "rl" / "chestxray8_grpo_hybrid_native.yaml"),
    )
    parser.add_argument(
        "--trace-jsonl",
        required=True,
        help="JSONL containing at least one Hybrid rollout trace record",
    )
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="full",
        help=(
            "full: stacked vs sequential production-matching truncated-BP; "
            "baseline-only / pass-a-only / pass-b-one-block: isolated diagnostics; "
            "compare-blocks: per-block logp table across cache/mask/SDPA variants; "
            "grad-forward-only: enable_grad trainability matrix; "
            "exact-safe-vs-production: blockwise logp match (no_grad)"
        ),
    )
    parser.add_argument(
        "--trainability-case",
        choices=TRAINABILITY_CASES,
        default=None,
        help=(
            "Required for --mode grad-forward-only. "
            "A=all frozen; B=LoRA only; C=projector only; D=LoRA+projector"
        ),
    )
    parser.add_argument(
        "--replay-use-cache",
        type=_parse_bool,
        default=None,
        help=(
            "Carry KV across blocks for diagnostic/baseline score (true|false). "
            "Does not by itself select legacy MTP masks."
        ),
    )
    parser.add_argument(
        "--legacy-nocache-masks",
        type=_parse_bool,
        default=False,
        help=(
            "If true, pass model use_cache=False so Qwen2 uses "
            "update_causal_mask_with_pad_non_visible_2d (diagnostic only)"
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        type=_parse_bool,
        default=None,
        help="Enable eval-mode layer checkpoint wrapping (true|false)",
    )
    parser.add_argument(
        "--force-math-sdpa",
        type=_parse_bool,
        default=None,
        help="Force math SDPA backend around the diagnostic forward (true|false)",
    )
    parser.add_argument(
        "--sync-after-layer",
        type=_parse_bool,
        default=False,
        help="Synchronize CUDA after every decoder layer forward (true|false)",
    )
    return parser.parse_args()


def _slot_from_dict(raw: Dict[str, Any]) -> SlotTrace:
    return SlotTrace(**raw)


def _block_from_dict(raw: Dict[str, Any]) -> BlockTrace:
    slots = [_slot_from_dict(slot) for slot in raw.get("slots") or []]
    payload = dict(raw)
    payload["slots"] = slots
    return BlockTrace(**payload)


def _trace_from_record(record: Dict[str, Any]) -> RolloutTrace:
    sampling_raw = record["sampling"]
    sampling = PBDSamplingConfig(**sampling_raw)
    blocks = [_block_from_dict(block) for block in record["blocks"]]
    return RolloutTrace(
        prompt_token_ids=list(record["prompt_token_ids"]),
        generated_token_ids=list(record["generated_token_ids"]),
        blocks=blocks,
        sampling=sampling,
        stopped_on_eos=bool(record.get("stopped_on_eos", False)),
        truncated=bool(record.get("truncated", False)),
        decoded_text=record.get("decoded_text"),
        decoder_path=str(record.get("decoder_path", "hybrid")),
        reward_branch=str(record.get("reward_branch", "none")),
        committed_final_box_norm_1000=(
            tuple(record["committed_final_box_norm_1000"])
            if record.get("committed_final_box_norm_1000") is not None
            else None
        ),
        has_unambiguous_committed_box=bool(
            record.get("has_unambiguous_committed_box", False)
        ),
        fallback_triggered=bool(record.get("fallback_triggered", False)),
        rejected_pbd_proposals=list(record.get("rejected_pbd_proposals") or []),
    )


def _selected_grads(model) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            continue
        if ("lora_" in name) or name.startswith("mlp1"):
            grads[name] = parameter.grad.detach().float().cpu()
    return grads


def _zero_grads(model) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad = None


def _tensor_meta(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, (list, tuple)):
        if not value:
            return {"type": type(value).__name__, "len": 0}
        if torch.is_tensor(value[0]):
            return {
                "type": type(value).__name__,
                "len": len(value),
                "element0": _tensor_meta(value[0]),
            }
        return {"type": type(value).__name__, "len": len(value)}
    if isinstance(value, dict):
        return {key: _tensor_meta(item) for key, item in value.items()}
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _nvidia_driver_version() -> Optional[str]:
    try:
        import subprocess

        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip().splitlines()[0].strip()
    except Exception:
        return None


def collect_environment_report(device: torch.device) -> Dict[str, Any]:
    gpu_name = None
    compute_capability = None
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        compute_capability = f"{major}.{minor}"
    transformers_version = None
    peft_version = None
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception as exc:  # pragma: no cover
        transformers_version = f"unavailable: {exc}"
    try:
        import peft

        peft_version = peft.__version__
    except Exception as exc:  # pragma: no cover
        peft_version = f"unavailable: {exc}"
    return {
        "torch_version": torch.__version__,
        "torch_version_cuda": getattr(torch.version, "cuda", None),
        "nvidia_driver_version": _nvidia_driver_version(),
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "transformers_version": transformers_version,
        "peft_version": peft_version,
        "device": str(device),
        "cuda_is_available": bool(torch.cuda.is_available()),
    }


def print_environment_report(env: Dict[str, Any]) -> None:
    print("=== ENVIRONMENT ===")
    print(f"torch.__version__: {env['torch_version']}")
    print(f"torch.version.cuda: {env['torch_version_cuda']}")
    print(f"NVIDIA driver version: {env['nvidia_driver_version']}")
    print(f"GPU name: {env['gpu_name']}")
    print(f"compute capability: {env['compute_capability']}")
    print(f"transformers version: {env['transformers_version']}")
    print(f"PEFT version: {env['peft_version']}")
    print("===================")


@contextmanager
def sync_after_layer_wrapper(
    model,
    *,
    enabled: bool = False,
    tracker: Optional[Dict[str, Any]] = None,
    synchronize: Optional[bool] = None,
) -> Iterator[None]:
    """Optional per-layer CUDA sync / failure tracking (diagnostic only).

    ``enabled`` installs the wrapper. ``synchronize`` defaults to ``enabled``;
    pass ``synchronize=False`` with a tracker to record layer indices without
    synchronizing after every layer.
    """
    if not enabled and tracker is None:
        yield
        return
    do_sync = bool(enabled if synchronize is None else synchronize)
    layers = _decoder_layers(model)
    originals = []
    if tracker is not None:
        tracker.setdefault("num_layers", len(layers))
        tracker.setdefault("last_layer_index", None)
        tracker.setdefault("first_failing_layer_index", None)
        tracker.setdefault("first_failing_layer_error", None)
        tracker.setdefault("layer0_hidden_meta", None)

    def _wrap(layer, layer_index: int):
        original = layer.forward

        def forward_with_sync(*args, **kwargs):
            if tracker is not None:
                tracker["last_layer_index"] = layer_index
                if (
                    tracker.get("layer0_hidden_meta") is None
                    and args
                    and torch.is_tensor(args[0])
                ):
                    tracker["layer0_hidden_meta"] = _tensor_meta(args[0])
            try:
                out = original(*args, **kwargs)
                if do_sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                return out
            except Exception as exc:
                if tracker is not None and tracker.get(
                    "first_failing_layer_index"
                ) is None:
                    tracker["first_failing_layer_index"] = layer_index
                    tracker["first_failing_layer_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                raise

        return forward_with_sync

    try:
        for index, layer in enumerate(layers):
            originals.append((layer, layer.forward))
            layer.forward = _wrap(layer, index)  # type: ignore[method-assign]
        yield
    finally:
        for layer, original in originals:
            layer.forward = original  # type: ignore[method-assign]


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_diag_flags(
    args: argparse.Namespace, config: Dict[str, Any]
) -> Dict[str, bool]:
    train_cfg = config.get("training") or {}
    return {
        "replay_use_cache": (
            bool(args.replay_use_cache)
            if args.replay_use_cache is not None
            else False
        ),
        "legacy_nocache_masks": bool(args.legacy_nocache_masks),
        "gradient_checkpointing": (
            bool(args.gradient_checkpointing)
            if args.gradient_checkpointing is not None
            else bool(train_cfg.get("gradient_checkpointing", False))
        ),
        "force_math_sdpa": (
            bool(args.force_math_sdpa)
            if args.force_math_sdpa is not None
            else False
        ),
        "sync_after_layer": bool(args.sync_after_layer),
    }


def _forward_contexts(
    model,
    flags: Dict[str, bool],
    *,
    layer_tracker: Optional[Dict[str, Any]] = None,
    force_layer_tracking: bool = False,
):
    ckpt = eval_mode_layer_checkpointing(
        model, enabled=bool(flags["gradient_checkpointing"])
    )
    sdpa = math_sdpa_for_grad_replay(enabled=bool(flags["force_math_sdpa"]))
    sync_flag = bool(flags["sync_after_layer"])
    sync = sync_after_layer_wrapper(
        model,
        enabled=sync_flag or force_layer_tracking,
        tracker=layer_tracker,
        synchronize=sync_flag,
    )
    return ckpt, sdpa, sync


def apply_trainability_case(model, case: str) -> Dict[str, Any]:
    """Set requires_grad for diagnostic matrix A/B/C/D without changing values."""
    case = str(case).upper()
    if case not in TRAINABILITY_CASES:
        raise RuntimeError(f"unknown trainability case {case!r}")
    for name, parameter in model.named_parameters():
        lora = is_lora_name(name)
        projector = is_projector_name(name)
        if case == "A":
            parameter.requires_grad_(False)
        elif case == "B":
            parameter.requires_grad_(lora)
        elif case == "C":
            parameter.requires_grad_(projector)
        else:  # D
            parameter.requires_grad_(lora or projector)
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    return {
        "trainability_case": case,
        "trainable_parameter_count": len(trainable),
        "trainable_parameter_numel": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "first_trainable_parameter_name": trainable[0] if trainable else None,
        "lora_trainable_count": sum(1 for name in trainable if is_lora_name(name)),
        "projector_trainable_count": sum(
            1 for name in trainable if is_projector_name(name)
        ),
        "all_parameters_requires_grad_false": len(trainable) == 0,
    }


def _position_id_report(position_ids: Any) -> Optional[Dict[str, Any]]:
    if not torch.is_tensor(position_ids):
        return None
    flat = position_ids.detach().float().reshape(-1)
    return {
        "shape": list(position_ids.shape),
        "dtype": str(position_ids.dtype).replace("torch.", ""),
        "device": str(position_ids.device),
        "min": int(flat.min().item()) if flat.numel() else None,
        "max": int(flat.max().item()) if flat.numel() else None,
    }


@contextmanager
def capture_prepare_inputs(language_model, sink: Dict[str, Any]) -> Iterator[None]:
    """Record attention_mask / position_ids from prepare_inputs_for_generation."""
    original = language_model.prepare_inputs_for_generation

    def wrapped(*args, **kwargs):
        prepared = original(*args, **kwargs)
        sink.clear()
        if isinstance(prepared, dict):
            mask = prepared.get("attention_mask")
            pos = prepared.get("position_ids")
            sink["attention_mask_shape"] = (
                list(mask.shape) if torch.is_tensor(mask) else None
            )
            sink["attention_mask_meta"] = (
                _tensor_meta(mask) if torch.is_tensor(mask) else None
            )
            sink["position_ids"] = _position_id_report(pos)
            input_ids = prepared.get("input_ids")
            inputs_embeds = prepared.get("inputs_embeds")
            if torch.is_tensor(input_ids):
                sink["prepared_input_ids_meta"] = _tensor_meta(input_ids)
            if torch.is_tensor(inputs_embeds):
                sink["prepared_inputs_embeds_meta"] = _tensor_meta(inputs_embeds)
            sink["prepared_use_cache"] = prepared.get("use_cache")
        return prepared

    language_model.prepare_inputs_for_generation = wrapped  # type: ignore[method-assign]
    try:
        yield
    finally:
        language_model.prepare_inputs_for_generation = original  # type: ignore[method-assign]


def run_exact_safe_vs_production(
    *,
    model,
    replayer,
    trace: RolloutTrace,
    decoder_kwargs: Dict[str, Any],
    device: torch.device,
    atol: float,
) -> Dict[str, Any]:
    """Compare no_grad production-cached score vs stop-grad-past safe score."""
    # Stop-grad path is diagnostic-only (forward exact / gradient inexact).
    assert_truncated_bptt_not_thesis_default(
        replay_backend="truncated_bptt_surrogate",
        allow_truncated_bptt_surrogate=True,
    )
    model.eval()
    if not hasattr(replayer, "score_autograd_safe"):
        raise RuntimeError("replayer lacks score_autograd_safe")
    report: Dict[str, Any] = {
        "mode": "exact-safe-vs-production",
        "backward_performed": False,
        "optimizer_step_performed": False,
        "production_replay": dict(PRODUCTION_REPLAY),
        "atol": float(atol),
        "block_logp_atol": float(BLOCK_LOGP_ATOL),
    }
    _cuda_sync(device)
    try:
        with torch.no_grad():
            prod_total, prod_blocks = replayer.score(
                trace,
                use_cache=True,
                legacy_nocache_masks=False,
                **decoder_kwargs,
            )
            safe_total, safe_blocks = replayer.score_autograd_safe(
                trace, **decoder_kwargs
            )
            # Bfix control (must remain rejected until equal to A).
            bfix_total, bfix_blocks = replayer.score(
                trace,
                use_cache=False,
                legacy_nocache_masks=False,
                **decoder_kwargs,
            )
        _cuda_sync(device)
        prod_f = [float(v.detach().float().cpu()) for v in prod_blocks]
        safe_f = [float(v.detach().float().cpu()) for v in safe_blocks]
        bfix_f = [float(v.detach().float().cpu()) for v in bfix_blocks]
        diffs = [abs(a - b) for a, b in zip(prod_f, safe_f)]
        report["production_total"] = float(prod_total.detach().float().cpu())
        report["safe_total"] = float(safe_total.detach().float().cpu())
        report["production_block_log_probs"] = prod_f
        report["safe_block_log_probs"] = safe_f
        report["per_block_abs_diff"] = diffs
        report["max_block_abs_diff"] = max(diffs) if diffs else 0.0
        report["total_abs_diff"] = abs(
            report["production_total"] - report["safe_total"]
        )
        report["bfix_control"] = bfix_discrepancy_report(
            a_block_logps=prod_f, bfix_block_logps=bfix_f
        )
        ok = report["max_block_abs_diff"] <= max(float(atol), BLOCK_LOGP_ATOL)
        report["status"] = "ok" if ok else "mismatch"
        # Forward-only mode: do not claim gradient equivalence here.
        report["verdict"] = (
            "forward_exact_match" if ok else "safe_diverged_from_production"
        )
        report["gradient_equivalence"] = {
            "tested": False,
            "note": (
                "This mode sets backward_performed=false. Run "
                "test_live_vs_stopgrad_gradients.py on CPU for live-vs-stop-grad "
                "gradient comparison (expected: truncated_bptt_surrogate)."
            ),
        }
        # Hard requirement: Bfix must still diverge until a true fix exists.
        report["bfix_still_diverges"] = bool(
            report["bfix_control"]["first_diverging_block"] is not None
            and abs(report["bfix_control"]["a_minus_bfix_total"]) > float(atol)
        )
    except Exception as exc:
        _cuda_sync(device)
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
    return report


def run_grad_forward_only(
    *,
    model,
    replayer,
    trace: RolloutTrace,
    decoder_kwargs: Dict[str, Any],
    device: torch.device,
    trainability_case: str,
    sync_after_layer: bool,
) -> Dict[str, Any]:
    """Production-cached enable_grad score; no backward / optimizer step.

    Isolates whether CUDA failures appear under gradient-bearing forward with
    LoRA and/or projector requires_grad, while keeping carry_cache=true and
    model use_cache=true (production MTP masks).
    """
    model.eval()
    trainability = apply_trainability_case(model, trainability_case)
    lm_dtype = str(model.language_model.dtype).replace("torch.", "")
    if lm_dtype not in {"bfloat16", "float16"}:
        raise RuntimeError(
            f"grad-forward-only expects bf16/fp16 activations, got {lm_dtype}"
        )

    flags = {
        "replay_use_cache": True,
        "legacy_nocache_masks": False,
        "gradient_checkpointing": False,
        "force_math_sdpa": False,
        "sync_after_layer": bool(sync_after_layer),
    }
    layer_tracker: Dict[str, Any] = {}
    prepare_sink: Dict[str, Any] = {}
    block_reports: List[Dict[str, Any]] = []
    first_failing_block_index: Optional[int] = None
    first_failing_block_source: Optional[str] = None

    settings: Dict[str, Any] = {
        "mode": "grad-forward-only",
        "trainability": trainability,
        "production_replay": {
            "carry_cache": True,
            "model_use_cache_for_mtp_masks": True,
            "legacy_nocache_masks": False,
            "gradient_checkpointing": False,
            "force_math_sdpa": False,
        },
        "model_training": bool(model.training),
        "language_model_training": bool(
            getattr(model.language_model, "training", False)
        ),
        "language_model_dtype": lm_dtype,
        "torch_grad_enabled": True,
        "backward_performed": False,
        "optimizer_step_performed": False,
        "sync_after_layer": bool(sync_after_layer),
        "input_metas": {
            key: _tensor_meta(value) for key, value in decoder_kwargs.items()
        },
        "trace_prompt_len": len(trace.prompt_token_ids),
        "trace_generated_len": len(trace.generated_token_ids),
        "trace_scored_blocks": sum(
            1 for block in trace.blocks if block.scored_for_grpo
        ),
        "decoder_path": getattr(trace, "decoder_path", None),
    }
    print("=== GRAD-FORWARD-ONLY SETTINGS ===")
    print(json.dumps(settings, indent=2, default=str))
    print("==================================")

    original_score_one = replayer._score_one_block
    scored_index = 0

    def _score_one_block_instrumented(**kwargs):
        nonlocal scored_index, first_failing_block_index, first_failing_block_source
        block = kwargs["block"]
        past = kwargs.get("past_key_values")
        carry_cache = bool(kwargs.get("carry_cache", True))
        cache_before = _cache_length(past) if carry_cache else 0
        layer_tracker["last_layer_index"] = None
        layer_tracker["first_failing_layer_index"] = None
        layer_tracker["first_failing_layer_error"] = None
        layer_tracker["layer0_hidden_meta"] = None
        prepare_sink.clear()
        block_report: Dict[str, Any] = {
            "scored_block_index": scored_index,
            "block_source": getattr(block, "source", None),
            "block_type": getattr(block, "block_type", None),
            "prefix_length": int(getattr(block, "prefix_length", -1)),
            "cache_length_before": int(cache_before),
            "trace_cache_length_before": int(
                getattr(block, "cache_length_before", -1)
            ),
            "status": "pending",
        }
        _cuda_sync(device)
        block_report["cuda_synchronized_before_block"] = True
        try:
            value, next_cache = original_score_one(**kwargs)
            _cuda_sync(device)
            block_report["cuda_synchronized_after_block"] = True
            block_report["status"] = "ok"
            block_report["block_log_prob"] = float(value.detach().float().cpu())
            block_report["block_log_prob_requires_grad"] = bool(value.requires_grad)
            block_report["attention_mask_shape"] = prepare_sink.get(
                "attention_mask_shape"
            )
            block_report["attention_mask_meta"] = prepare_sink.get(
                "attention_mask_meta"
            )
            block_report["position_ids"] = prepare_sink.get("position_ids")
            block_report["prepared_input_ids_meta"] = prepare_sink.get(
                "prepared_input_ids_meta"
            )
            block_report["prepared_inputs_embeds_meta"] = prepare_sink.get(
                "prepared_inputs_embeds_meta"
            )
            block_report["prepared_use_cache"] = prepare_sink.get(
                "prepared_use_cache"
            )
            block_report["layer0_hidden_meta"] = layer_tracker.get(
                "layer0_hidden_meta"
            )
            block_report["last_layer_index"] = layer_tracker.get("last_layer_index")
            block_report["cache_length_after"] = (
                _cache_length(next_cache) if next_cache is not None else None
            )
            block_reports.append(block_report)
            scored_index += 1
            return value, next_cache
        except Exception as exc:
            _cuda_sync(device)
            block_report["cuda_synchronized_after_block"] = True
            block_report["status"] = "error"
            block_report["error_type"] = type(exc).__name__
            block_report["error_message"] = str(exc)
            block_report["attention_mask_shape"] = prepare_sink.get(
                "attention_mask_shape"
            )
            block_report["attention_mask_meta"] = prepare_sink.get(
                "attention_mask_meta"
            )
            block_report["position_ids"] = prepare_sink.get("position_ids")
            block_report["prepared_input_ids_meta"] = prepare_sink.get(
                "prepared_input_ids_meta"
            )
            block_report["prepared_inputs_embeds_meta"] = prepare_sink.get(
                "prepared_inputs_embeds_meta"
            )
            block_report["layer0_hidden_meta"] = layer_tracker.get(
                "layer0_hidden_meta"
            )
            block_report["last_layer_index"] = layer_tracker.get("last_layer_index")
            block_report["first_failing_layer_index"] = layer_tracker.get(
                "first_failing_layer_index"
            )
            block_report["first_failing_layer_error"] = layer_tracker.get(
                "first_failing_layer_error"
            )
            block_reports.append(block_report)
            if first_failing_block_index is None:
                first_failing_block_index = scored_index
                first_failing_block_source = getattr(block, "source", None)
            raise

    ckpt, sdpa, sync = _forward_contexts(
        model,
        flags,
        layer_tracker=layer_tracker,
        force_layer_tracking=True,
    )
    language_model = model.language_model
    try:
        replayer._score_one_block = _score_one_block_instrumented  # type: ignore[method-assign]
        with torch.enable_grad(), ckpt, sdpa, sync, capture_prepare_inputs(
            language_model, prepare_sink
        ):
            settings["torch_is_grad_enabled_during_score"] = bool(
                torch.is_grad_enabled()
            )
            current, block_logps = replayer.score(
                trace,
                use_cache=True,
                legacy_nocache_masks=False,
                **decoder_kwargs,
            )
        settings["status"] = "ok"
        settings["current_log_prob"] = float(current.detach().float().cpu())
        settings["current_logp_requires_grad"] = bool(current.requires_grad)
        settings["current_log_prob_meta"] = _tensor_meta(current)
        settings["num_block_logps"] = len(block_logps)
        settings["block_log_probs"] = [
            float(value.detach().float().cpu()) for value in block_logps
        ]
        settings["block_logp_requires_grad"] = [
            bool(value.requires_grad) for value in block_logps
        ]
        del current, block_logps
    except Exception as exc:
        settings["status"] = "error"
        settings["error_type"] = type(exc).__name__
        settings["error_message"] = str(exc)
        settings["traceback"] = traceback.format_exc()
        print(settings["traceback"])
    finally:
        replayer._score_one_block = original_score_one  # type: ignore[method-assign]

    settings["blocks"] = block_reports
    settings["first_failing_block_index"] = first_failing_block_index
    settings["first_failing_block_source"] = first_failing_block_source
    settings["first_failing_layer_index"] = next(
        (
            block.get("first_failing_layer_index")
            for block in block_reports
            if block.get("status") == "error"
            and block.get("first_failing_layer_index") is not None
        ),
        layer_tracker.get("first_failing_layer_index"),
    )
    settings["layer_tracker_summary"] = {
        "num_layers": layer_tracker.get("num_layers"),
        "layer0_hidden_meta": layer_tracker.get("layer0_hidden_meta"),
        "sync_after_layer_enabled": bool(sync_after_layer),
    }
    # Prefer the first successful block's prepared input / hidden metas for the
    # case-level summary fields requested by the diagnostic matrix.
    first_ok = next(
        (block for block in block_reports if block.get("status") == "ok"), None
    )
    first_any = block_reports[0] if block_reports else None
    summary_src = first_ok or first_any or {}
    settings["input_dtype_device"] = summary_src.get("prepared_input_ids_meta") or (
        settings["input_metas"].get("input_ids")
    )
    settings["hidden_state_dtype_device"] = summary_src.get("layer0_hidden_meta")
    settings["attention_mask_shape"] = summary_src.get("attention_mask_shape")
    settings["position_id_shape_range"] = summary_src.get("position_ids")
    return settings


def run_baseline_only(
    *,
    model,
    replayer,
    trace: RolloutTrace,
    decoder_kwargs: Dict[str, Any],
    device: torch.device,
    flags: Dict[str, bool],
) -> Dict[str, Any]:
    """Exactly one replayer.score; no backward; no optimizer step."""
    model.eval()
    input_meta = {key: _tensor_meta(value) for key, value in decoder_kwargs.items()}
    settings = {
        "mode": "baseline-only",
        "replay_use_cache": bool(flags["replay_use_cache"]),
        "legacy_nocache_masks": bool(flags["legacy_nocache_masks"]),
        "gradient_checkpointing": bool(flags["gradient_checkpointing"]),
        "force_math_sdpa": bool(flags["force_math_sdpa"]),
        "sync_after_layer": bool(flags["sync_after_layer"]),
        "model_training": bool(model.training),
        "language_model_training": bool(
            getattr(model.language_model, "training", False)
        ),
        "language_model_dtype": str(model.language_model.dtype).replace(
            "torch.", ""
        ),
        "trace_prompt_len": len(trace.prompt_token_ids),
        "trace_generated_len": len(trace.generated_token_ids),
        "trace_scored_blocks": sum(
            1 for block in trace.blocks if block.scored_for_grpo
        ),
        "decoder_path": getattr(trace, "decoder_path", None),
        "input_metas": input_meta,
        "backward_performed": False,
        "optimizer_step_performed": False,
    }
    print("=== BASELINE-ONLY SETTINGS ===")
    print(json.dumps(settings, indent=2))
    print("==============================")

    ckpt, sdpa, sync = _forward_contexts(model, flags)
    _cuda_sync(device)
    settings["cuda_synchronized_before_forward"] = True
    try:
        with torch.no_grad(), ckpt, sdpa, sync:
            current, block_logps = replayer.score(
                trace,
                use_cache=bool(flags["replay_use_cache"]),
                legacy_nocache_masks=bool(flags["legacy_nocache_masks"]),
                **decoder_kwargs,
            )
        _cuda_sync(device)
        settings["cuda_synchronized_after_forward"] = True
        settings["status"] = "ok"
        settings["current_log_prob"] = float(current.detach().float().cpu())
        settings["current_log_prob_meta"] = _tensor_meta(current.detach())
        settings["num_block_logps"] = len(block_logps)
        settings["block_log_probs"] = [
            float(value.detach().float().cpu()) for value in block_logps
        ]
        settings["block_logp_metas"] = [
            _tensor_meta(value.detach()) for value in block_logps[:8]
        ]
        del current, block_logps
    except Exception as exc:
        _cuda_sync(device)
        settings["cuda_synchronized_after_forward"] = True
        settings["status"] = "error"
        settings["error_type"] = type(exc).__name__
        settings["error_message"] = str(exc)
        settings["traceback"] = traceback.format_exc()
        print(settings["traceback"])
        return settings
    return settings


def run_pass_a_only(
    *,
    model,
    replayer,
    trace: RolloutTrace,
    decoder_kwargs: Dict[str, Any],
    device: torch.device,
    flags: Dict[str, bool],
) -> Dict[str, Any]:
    """Pass-A style no-grad full-trajectory score only."""
    report = run_baseline_only(
        model=model,
        replayer=replayer,
        trace=trace,
        decoder_kwargs=decoder_kwargs,
        device=device,
        flags=flags,
    )
    report["mode"] = "pass-a-only"
    return report


def run_pass_b_one_block(
    *,
    model,
    replayer,
    trace: RolloutTrace,
    decoder_kwargs: Dict[str, Any],
    device: torch.device,
    flags: Dict[str, bool],
) -> Dict[str, Any]:
    """Forward only the first scored block via iter_scored_block_logprobs."""
    if bool(flags["replay_use_cache"]):
        raise RuntimeError(
            "pass-b-one-block requires --replay-use-cache false "
            "(independent per-block graphs)"
        )
    model.eval()
    settings: Dict[str, Any] = {
        "mode": "pass-b-one-block",
        "replay_use_cache": False,
        "legacy_nocache_masks": bool(flags["legacy_nocache_masks"]),
        "gradient_checkpointing": bool(flags["gradient_checkpointing"]),
        "force_math_sdpa": bool(flags["force_math_sdpa"]),
        "sync_after_layer": bool(flags["sync_after_layer"]),
        "input_metas": {
            key: _tensor_meta(value) for key, value in decoder_kwargs.items()
        },
        "backward_performed": False,
        "optimizer_step_performed": False,
    }
    print("=== PASS-B-ONE-BLOCK SETTINGS ===")
    print(json.dumps(settings, indent=2))
    print("=================================")

    ckpt, sdpa, sync = _forward_contexts(model, flags)
    _cuda_sync(device)
    settings["cuda_synchronized_before_forward"] = True
    try:
        with torch.no_grad(), ckpt, sdpa, sync:
            iterator = replayer.iter_scored_block_logprobs(
                trace,
                use_cache=False,
                legacy_nocache_masks=bool(flags["legacy_nocache_masks"]),
                **decoder_kwargs,
            )
            first = next(iterator, None)
        _cuda_sync(device)
        settings["cuda_synchronized_after_forward"] = True
        if first is None:
            settings["status"] = "ok"
            settings["num_scored_blocks_seen"] = 0
            settings["first_block_log_prob"] = None
        else:
            settings["status"] = "ok"
            settings["num_scored_blocks_seen"] = 1
            settings["first_block_log_prob"] = float(first.detach().float().cpu())
            settings["first_block_log_prob_meta"] = _tensor_meta(first.detach())
            del first
    except Exception as exc:
        _cuda_sync(device)
        settings["cuda_synchronized_after_forward"] = True
        settings["status"] = "error"
        settings["error_type"] = type(exc).__name__
        settings["error_message"] = str(exc)
        settings["traceback"] = traceback.format_exc()
        print(settings["traceback"])
    return settings


def _score_variant(
    *,
    model,
    replayer,
    trace: RolloutTrace,
    decoder_kwargs: Dict[str, Any],
    device: torch.device,
    carry_cache: bool,
    legacy_nocache_masks: bool,
    force_math_sdpa: bool,
) -> Dict[str, Any]:
    flags = {
        "replay_use_cache": carry_cache,
        "legacy_nocache_masks": legacy_nocache_masks,
        "gradient_checkpointing": False,
        "force_math_sdpa": force_math_sdpa,
        "sync_after_layer": False,
    }
    ckpt, sdpa, sync = _forward_contexts(model, flags)
    _cuda_sync(device)
    with torch.no_grad(), ckpt, sdpa, sync:
        total, block_logps = replayer.score(
            trace,
            use_cache=carry_cache,
            legacy_nocache_masks=legacy_nocache_masks,
            **decoder_kwargs,
        )
    _cuda_sync(device)
    blocks = [float(v.detach().float().cpu()) for v in block_logps]
    return {
        "trajectory_log_prob": float(total.detach().float().cpu()),
        "block_log_probs": blocks,
        "carry_cache": carry_cache,
        "legacy_nocache_masks": legacy_nocache_masks,
        "force_math_sdpa": force_math_sdpa,
        "model_use_cache_for_mtp_masks": (not legacy_nocache_masks),
    }


def run_compare_blocks(
    *,
    model,
    replayer,
    trace: RolloutTrace,
    decoder_kwargs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Per-block logp table across cache/mask/SDPA variants (no backward)."""
    model.eval()
    variants = {
        "A_carry_cache_true_production_masks": _score_variant(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            carry_cache=True,
            legacy_nocache_masks=False,
            force_math_sdpa=False,
        ),
        "B_legacy_model_use_cache_false": _score_variant(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            carry_cache=False,
            legacy_nocache_masks=True,
            force_math_sdpa=False,
        ),
        "Bfix_carry_false_production_masks": _score_variant(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            carry_cache=False,
            legacy_nocache_masks=False,
            force_math_sdpa=False,
        ),
        "C_legacy_plus_math_sdpa": _score_variant(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            carry_cache=False,
            legacy_nocache_masks=True,
            force_math_sdpa=True,
        ),
    }

    a = variants["A_carry_cache_true_production_masks"]["block_log_probs"]
    b = variants["B_legacy_model_use_cache_false"]["block_log_probs"]
    bfix = variants["Bfix_carry_false_production_masks"]["block_log_probs"]
    c = variants["C_legacy_plus_math_sdpa"]["block_log_probs"]
    n = min(len(a), len(b), len(bfix), len(c))
    table = []
    first_ab = None
    first_ac = None
    first_a_bfix = None
    scored_meta = [block for block in trace.blocks if block.scored_for_grpo]
    for i in range(n):
        row = {
            "block_index": i,
            "block_type": scored_meta[i].block_type if i < len(scored_meta) else None,
            "source": getattr(scored_meta[i], "source", None)
            if i < len(scored_meta)
            else None,
            "prefix_length": scored_meta[i].prefix_length
            if i < len(scored_meta)
            else None,
            "cache_length_before": scored_meta[i].cache_length_before
            if i < len(scored_meta)
            else None,
            "A": a[i],
            "B_legacy": b[i],
            "Bfix_production_masks": bfix[i],
            "C_legacy_math_sdpa": c[i],
            "A_minus_B": a[i] - b[i],
            "A_minus_Bfix": a[i] - bfix[i],
            "A_minus_C": a[i] - c[i],
            "B_minus_C": b[i] - c[i],
        }
        table.append(row)
        if first_ab is None and abs(row["A_minus_B"]) > 1e-5:
            first_ab = i
        if first_a_bfix is None and abs(row["A_minus_Bfix"]) > 1e-5:
            first_a_bfix = i
        if first_ac is None and abs(row["A_minus_C"]) > 1e-5:
            first_ac = i

    a_tot = variants["A_carry_cache_true_production_masks"]["trajectory_log_prob"]
    b_tot = variants["B_legacy_model_use_cache_false"]["trajectory_log_prob"]
    bfix_tot = variants["Bfix_carry_false_production_masks"]["trajectory_log_prob"]
    c_tot = variants["C_legacy_plus_math_sdpa"]["trajectory_log_prob"]

    def _ratio(curr: float, old: float) -> float:
        return float(torch.exp(torch.tensor(curr - old)))

    report = {
        "mode": "compare-blocks",
        "variants": variants,
        "per_block_table": table,
        "first_diverging_block_A_vs_B_legacy": first_ab,
        "first_diverging_block_A_vs_Bfix": first_a_bfix,
        "first_diverging_block_A_vs_C": first_ac,
        "trajectory_totals": {
            "A": a_tot,
            "B_legacy": b_tot,
            "Bfix": bfix_tot,
            "C_legacy_math_sdpa": c_tot,
            "A_minus_B": a_tot - b_tot,
            "A_minus_Bfix": a_tot - bfix_tot,
            "B_minus_C": b_tot - c_tot,
        },
        "initialization_ppo_ratios_if_mismatched_configs": {
            "old_A_current_B_legacy": _ratio(b_tot, a_tot),
            "old_B_current_B": _ratio(b_tot, b_tot),
            "old_C_current_C": _ratio(c_tot, c_tot),
            "old_A_current_Bfix": _ratio(bfix_tot, a_tot),
        },
        "root_cause_notes": {
            "cache_discrepancy": (
                "Qwen2 eval SDPA uses update_causal_mask_for_one_gen_window_2d "
                "when model use_cache=True, vs update_causal_mask_with_pad_non_visible_2d "
                "when model use_cache=False. Legacy carry_cache=false previously set "
                "model use_cache=False and therefore changed MTP attention visibility."
            ),
            "sdpa_discrepancy": (
                "Forced math-only SDPA vs the model's default flash+math "
                "(mem_efficient=False) path introduces additional bf16 numerical drift; "
                "it is not required for production-mask equivalence."
            ),
            "chosen_unified_config": (
                "carry_cache=True for scoring; Pass-B carry_cache=False; "
                "legacy_nocache_masks=False (always production MTP masks); "
                "force_math_sdpa=False."
            ),
        },
        "backward_performed": False,
        "optimizer_step_performed": False,
        "status": "ok",
    }
    print("=== COMPARE-BLOCKS SUMMARY ===")
    print(json.dumps(report["trajectory_totals"], indent=2))
    print(
        "first_diverging_block_A_vs_B_legacy=",
        first_ab,
        "first_diverging_block_A_vs_Bfix=",
        first_a_bfix,
    )
    print("==============================")
    return report


def _load_group_records(
    records: List[Dict[str, Any]], seed_index: int
) -> List[Dict[str, Any]]:
    """Load the G=4 records for the same sample as ``records[seed_index]``."""
    seed = records[int(seed_index)]
    sample_index = int(seed.get("sample_index", 0))
    group = [
        rec
        for rec in records
        if int(rec.get("sample_index", -1)) == sample_index
    ]
    group = sorted(group, key=lambda rec: int(rec.get("group_index", 0)))
    if len(group) != 4:
        raise RuntimeError(
            f"expected G=4 traces for sample_index={sample_index}, found {len(group)}"
        )
    return group


def run_full_equivalence(
    *,
    args: argparse.Namespace,
    model,
    replayer,
    traces: List[RolloutTrace],
    group_records: List[Dict[str, Any]],
    decoder_kwargs: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
    revision: str,
    semantics: Dict[str, Any],
    flags: Dict[str, bool],
) -> Dict[str, Any]:
    """Compare stacked G=4 baseline vs sequential (loss_i/G) production replay."""
    assert_grpo_loss_depends_only_on_trajectory_logprob()
    model.eval()
    group_size = 4
    clip_epsilon = float(config["objective"]["ppo_clip_epsilon"])
    train_cfg = config["training"]

    rewards = [
        float(rec.get("reward", {}).get("total_reward", rec.get("advantage", 0.0)))
        for rec in group_records
    ]
    # Prefer stored advantages when present; otherwise recompute from rewards.
    if all("advantage" in rec for rec in group_records):
        advantages = torch.tensor(
            [float(rec["advantage"]) for rec in group_records],
            device=device,
            dtype=torch.float32,
        )
    else:
        advantages = group_relative_advantages(rewards).to(device)

    # Force nonzero advantages for gradient comparison if the group is flat.
    if float(advantages.abs().max()) < 1e-8:
        advantages = torch.tensor(
            [1.0, -1.0, 0.5, -0.5], device=device, dtype=torch.float32
        )

    ckpt_off, sdpa_off, sync_off = _forward_contexts(
        model,
        {
            **flags,
            "force_math_sdpa": False,
            "gradient_checkpointing": False,
            "sync_after_layer": False,
        },
    )

    # Old / current init identity under production cached replay.
    with torch.no_grad(), ckpt_off, sdpa_off, sync_off:
        old_logps = []
        current_init = []
        for trace in traces:
            old_v, _ = replayer.score(
                trace, use_cache=True, legacy_nocache_masks=False, **decoder_kwargs
            )
            cur_v, _ = replayer.score(
                trace, use_cache=True, legacy_nocache_masks=False, **decoder_kwargs
            )
            old_logps.append(float(old_v.detach().float().cpu()))
            current_init.append(float(cur_v.detach().float().cpu()))
            del old_v, cur_v
    init_check = assert_initialization_ratios(current_init, old_logps)
    old_tensor = torch.tensor(old_logps, device=device, dtype=torch.float32)

    # ---- Stacked G=4 baseline (may OOM; report, do not fall back) ----
    _zero_grads(model)
    state0 = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    try:
        with ckpt_off, sdpa_off, sync_off:
            current_stack = torch.stack(
                [
                    replayer.score(
                        trace,
                        use_cache=True,
                        legacy_nocache_masks=False,
                        **decoder_kwargs,
                    )[0]
                    for trace in traces
                ]
            )
        loss_stacked = grpo_clipped_loss(
            current_stack,
            old_tensor,
            advantages,
            clip_epsilon=clip_epsilon,
        )
        loss_stacked.backward()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "OOM during stacked G=4 baseline comparison. Sequential "
                "production path may still fit; re-run with --mode "
                "sequential-only if needed. Refusing Bfix fallback."
            ) from exc
        raise
    grads_stacked = _selected_grads(model)
    loss_stacked_f = float(loss_stacked.detach().float().cpu())
    current_stacked = [float(x) for x in current_stack.detach().float().cpu()]
    del current_stack, loss_stacked
    _zero_grads(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Restore params for sequential path (no optimizer step yet).
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state0:
                parameter.copy_(state0[name].to(parameter.device))

    # ---- Sequential (loss_i / G).backward() ----
    optimizer = build_optimizer(
        model,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        use_8bit_adam=bool(train_cfg["use_8bit_adam"]),
        projector_lr=float(train_cfg["projector_learning_rate"]),
    )
    # Capture pre-step params for update comparison against a second optimizer
    # on the stacked grads path.
    pre_step = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (("lora_" in name) or name.startswith("mlp1"))
    }

    if bool(flags["gradient_checkpointing"]):
        raise RuntimeError(
            "verify --mode full uses sequential_production_cached replay; "
            "gradient_checkpointing must be false (KV-cache checkpoint recompute "
            "is unsafe). Pass --gradient-checkpointing false."
        )

    _zero_grads(model)
    seq_losses: List[float] = []
    seq_current: List[float] = []
    try:
        for index, trace in enumerate(traces):
            current_i, _ = replayer.score(
                trace,
                use_cache=True,
                legacy_nocache_masks=False,
                **decoder_kwargs,
            )
            loss_i = grpo_clipped_loss(
                current_i.reshape(1),
                old_tensor[index].reshape(1),
                advantages[index].reshape(1),
                clip_epsilon=clip_epsilon,
            )
            (loss_i / float(group_size)).backward()
            seq_current.append(float(current_i.detach().float().cpu()))
            seq_losses.append(float(loss_i.detach().float().cpu()))
            del current_i, loss_i
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "OOM during sequential production-cached single-trajectory "
                "GRPO replay on this GPU. Stop here; do not fall back to "
                "cacheless Bfix chain-rule (A != Bfix)."
            ) from exc
        raise

    grads_seq = _selected_grads(model)
    loss_seq_f = float(sum(seq_losses) / group_size)

    # One optimizer step on sequential grads.
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        float(train_cfg["max_grad_norm"]),
    )
    optimizer.step()
    post_seq = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in pre_step
    }

    # Rebuild stacked path update on a fresh clone of initial params.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state0:
                parameter.copy_(state0[name].to(parameter.device))
    optimizer_b = build_optimizer(
        model,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        use_8bit_adam=bool(train_cfg["use_8bit_adam"]),
        projector_lr=float(train_cfg["projector_learning_rate"]),
    )
    _zero_grads(model)
    for name, parameter in model.named_parameters():
        if name in grads_stacked:
            parameter.grad = grads_stacked[name].to(
                device=parameter.device, dtype=parameter.dtype
            )
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad],
        float(train_cfg["max_grad_norm"]),
    )
    optimizer_b.step()
    post_stacked = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in pre_step
    }

    shared = sorted(set(grads_stacked) & set(grads_seq))
    if not shared:
        raise RuntimeError("no overlapping selected grads to compare")
    max_abs = 0.0
    rel = 0.0
    for name in shared:
        diff = (grads_stacked[name] - grads_seq[name]).abs()
        max_abs = max(max_abs, float(diff.max()))
        denom = float(grads_stacked[name].abs().max().clamp(min=1e-12))
        rel = max(rel, float(diff.max() / denom))

    update_max_abs = 0.0
    update_rel = 0.0
    for name in pre_step:
        d_seq = (post_seq[name] - pre_step[name]).abs()
        d_stack = (post_stacked[name] - pre_step[name]).abs()
        diff = (d_seq - d_stack).abs()
        update_max_abs = max(update_max_abs, float(diff.max()))
        denom = float(d_stack.max().clamp(min=1e-12))
        update_rel = max(update_rel, float(diff.max() / denom))

    loss_diff = abs(loss_stacked_f - loss_seq_f)
    logp_diffs = [abs(a - b) for a, b in zip(current_stacked, seq_current)]
    ok = (
        loss_diff <= args.atol
        and max(logp_diffs) <= args.atol
        and (max_abs <= args.atol or rel <= args.rtol)
        and (update_max_abs <= args.atol or update_rel <= args.rtol)
    )
    return {
        "mode": "full",
        "revision": revision,
        "trace_index": args.trace_index,
        "trace_jsonl": str(Path(args.trace_jsonl).resolve()),
        "replay_semantics": semantics,
        "diagnostic_flags": flags,
        "production_replay": dict(PRODUCTION_REPLAY),
        "initialization_ratio_check": init_check,
        "stacked_loss": loss_stacked_f,
        "sequential_loss": loss_seq_f,
        "loss_abs_diff": loss_diff,
        "stacked_current_log_probs": current_stacked,
        "sequential_current_log_probs": seq_current,
        "old_log_probs": old_logps,
        "per_rollout_logprob_max_difference": max(logp_diffs),
        "gradient_max_absolute_difference": max_abs,
        "gradient_relative_difference": rel,
        "optimizer_update_max_absolute_difference": update_max_abs,
        "optimizer_update_relative_difference": update_rel,
        "selected_grad_names": shared,
        "lora_grad_count": sum(1 for name in shared if "lora_" in name),
        "projector_grad_count": sum(1 for name in shared if name.startswith("mlp1")),
        "optimizer_step_performed": True,
        "two_pass_chain_rule_used": False,
        "reference_in_loss": False,
        "verdict": "exact_match" if ok else "mismatch",
        "_ok": ok,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    env = collect_environment_report(device)
    print_environment_report(env)

    config = load_resolved_config(args.config)
    flags = _resolve_diag_flags(args, config)
    records = [
        json.loads(line)
        for line in Path(args.trace_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("trace jsonl is empty")
    record = records[int(args.trace_index)]
    trace_payload = record.get("rollout_trace", record)
    trace = _trace_from_record(trace_payload)

    pairs = load_verified_pairs(config, "train")
    sample_index = int(record.get("sample_index", 0))
    pair = pairs[sample_index]
    model, tokenizer, processor, revision = build_policy(config, device)
    semantics = replay_semantics_report(model, config)
    replayer = build_rollout_replayer(model, tokenizer, config)
    inputs = tokenize_rl_pair(processor, pair, device, config=config)
    decoder_kwargs = decoder_inputs(inputs)

    report: Dict[str, Any]
    exit_error = False
    if args.mode == "baseline-only":
        report = run_baseline_only(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            flags=flags,
        )
        exit_error = report.get("status") == "error"
    elif args.mode == "pass-a-only":
        report = run_pass_a_only(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            flags=flags,
        )
        exit_error = report.get("status") == "error"
    elif args.mode == "pass-b-one-block":
        report = run_pass_b_one_block(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            flags=flags,
        )
        exit_error = report.get("status") == "error"
    elif args.mode == "compare-blocks":
        report = run_compare_blocks(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
        )
        exit_error = report.get("status") == "error"
    elif args.mode == "grad-forward-only":
        if args.trainability_case is None:
            raise RuntimeError(
                "--mode grad-forward-only requires --trainability-case A|B|C|D"
            )
        report = run_grad_forward_only(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            trainability_case=str(args.trainability_case),
            sync_after_layer=bool(flags["sync_after_layer"]),
        )
        exit_error = report.get("status") == "error"
    elif args.mode == "exact-safe-vs-production":
        report = run_exact_safe_vs_production(
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            device=device,
            atol=float(args.atol),
        )
        exit_error = report.get("status") != "ok"
    else:
        group_records = _load_group_records(records, args.trace_index)
        group_traces = [
            _trace_from_record(rec.get("rollout_trace", rec)) for rec in group_records
        ]
        report = run_full_equivalence(
            args=args,
            model=model,
            replayer=replayer,
            traces=group_traces,
            group_records=group_records,
            decoder_kwargs=decoder_kwargs,
            config=config,
            device=device,
            revision=revision,
            semantics=semantics,
            flags=flags,
        )
        exit_error = not bool(report.pop("_ok", False))

    report["environment"] = env
    report["revision"] = revision
    report["trace_index"] = args.trace_index
    report["trace_jsonl"] = str(Path(args.trace_jsonl).resolve())
    report["replay_semantics"] = semantics
    report["diagnostic_flags"] = flags
    write_json(Path(args.output_json), report)
    print(json.dumps(report, indent=2))
    if exit_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
