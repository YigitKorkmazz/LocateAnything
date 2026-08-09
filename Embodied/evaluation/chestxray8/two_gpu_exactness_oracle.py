#!/usr/bin/env python3
"""Fresh-process A1/A2 vs exact two-GPU live-cache shard oracle.

This is a short-fixture backend exactness test only.  It performs no training
optimizer step on the model: AdamW is applied only to CPU clones of the saved
gradients for an update-equivalence comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is present in the experiment env
    np = None  # type: ignore[assignment]

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.inprocess_short_fixture_oracle import (  # noqa: E402
    BASELINE_ABS_ATOL,
    BASELINE_REL_ATOL,
    BASELINE_REL_SLACK,
    normalize_ab_by_a12_baseline,
)
from rl.nan_grad_diagnostics import compare_forward_scalars, compare_grad_dicts  # noqa: E402
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    build_policy,
    build_rollout_replayer,
    load_resolved_config,
    load_verified_pairs,
    write_json,
)
from rl.short_ab_nan_compare import (  # noqa: E402
    _optimizer_update_compare_cpu,
    _zero_grads,
    restore_trainable_init,
    run_one_short_condition,
    snapshot_trainable_init,
)
from rl.short_sequence_fixture import (  # noqa: E402
    DEFAULT_ORACLE_BLOCK_ADVANTAGES,
    build_short_sequence_gradient_oracle_fixture,
)
from rl.two_gpu_shard import (  # noqa: E402
    DecoderShardLayout,
    resolve_locateanything_qwen_decoder,
    shard_locateanything_decoder_two_gpu,
)
from two_gpu_live_cache_feasibility import (  # noqa: E402
    _ReplayKVRecorder,
    _cross_device_boundary_report,
)


OUTPUT_FILENAME = "two_gpu_exactness_oracle.json"
GRADIENT_ARTIFACT_FILENAME = "two_gpu_exactness_oracle_gradients.pt"


def _non_json_safe_fields(value: Any, path: str = "report") -> List[Dict[str, str]]:
    """List every non-JSON-native leaf before sanitization changes it."""
    unsafe: List[Dict[str, str]] = []

    def visit(item: Any, item_path: str) -> None:
        if item is None or isinstance(item, (bool, int, float, str)):
            return
        if isinstance(item, torch.Tensor):
            unsafe.append({"path": item_path, "python_type": "torch.Tensor"})
            return
        if isinstance(item, (torch.dtype, torch.device, Path)):
            unsafe.append({"path": item_path, "python_type": type(item).__name__})
            return
        if np is not None and isinstance(item, np.generic):
            unsafe.append({"path": item_path, "python_type": type(item).__name__})
            return
        if is_dataclass(item) and not isinstance(item, type):
            unsafe.append({"path": item_path, "python_type": type(item).__name__})
            visit(asdict(item), item_path)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                key_text = str(key)
                if not isinstance(key, str):
                    unsafe.append({"path": f"{item_path}.<key:{key_text}>", "python_type": type(key).__name__})
                visit(child, f"{item_path}.{key_text}")
            return
        if isinstance(item, (list, tuple, set)):
            if isinstance(item, (tuple, set)):
                unsafe.append({"path": item_path, "python_type": type(item).__name__})
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")
            return
        unsafe.append({"path": item_path, "python_type": type(item).__name__})

    visit(value, path)
    return unsafe


def _json_sanitize(value: Any) -> Any:
    """Convert report diagnostics to JSON without ever embedding gradient values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_sanitize(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
            "numel": int(value.numel()),
        }
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_sanitize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_sanitize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_sanitize(child) for child in value]
    return {"python_type": type(value).__name__, "repr": repr(value)}


def _write_report_safely(path: Path, report: Dict[str, Any]) -> None:
    existing = list(
        ((report.get("report_serialization") or {}).get("non_json_safe_fields") or [])
    )
    current = _non_json_safe_fields(report)
    seen = set()
    report["report_serialization"] = {
        "non_json_safe_fields": [
            item for item in (existing + current)
            if not (tuple(item.items()) in seen or seen.add(tuple(item.items())))
        ]
    }
    try:
        write_json(path, _json_sanitize(report))
    except Exception as exc:  # pragma: no cover - emergency write path
        emergency = {
            "status": report.get("status", "unknown"),
            "report_serialization": report.get("report_serialization"),
            "serialization_error_type": type(exc).__name__,
            "serialization_error": str(exc),
            "preserved_top_level_keys": sorted(str(key) for key in report),
        }
        path.write_text(json.dumps(emergency, indent=2, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--image-max-side", type=int, default=96)
    parser.add_argument("--target-max-prompt-tokens", type=int, default=192)
    parser.add_argument("--adamw-lr", type=float, default=1e-5)
    return parser.parse_args()


def _sync_and_memory(devices: List[torch.device]) -> Dict[str, Dict[str, float]]:
    report: Dict[str, Dict[str, float]] = {}
    for device in devices:
        index = int(device.index or 0)
        torch.cuda.synchronize(index)
        report[str(device)] = {
            "allocated_mb": float(torch.cuda.memory_allocated(index) / (1024**2)),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated(index) / (1024**2)),
            "reserved_mb": float(torch.cuda.memory_reserved(index) / (1024**2)),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved(index) / (1024**2)),
        }
    return report


def _reset_peaks(devices: List[torch.device]) -> None:
    for device in devices:
        torch.cuda.reset_peak_memory_stats(int(device.index or 0))


def _lora_grad_bank(model, expected_names: List[str]) -> Dict[str, torch.Tensor]:
    named = dict(model.named_parameters())
    return {
        name: named[name].grad.detach().float().cpu().clone()
        for name in expected_names
        if name in named and named[name].grad is not None
    }


def _gradient_masks(expected_names: List[str], bank: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    missing = [name for name in expected_names if name not in bank]
    zero = [name for name in expected_names if name in bank and torch.count_nonzero(bank[name]).item() == 0]
    nonfinite = [
        name for name in expected_names
        if name in bank and not bool(torch.isfinite(bank[name]).all().item())
    ]
    return {
        "parameter_names": list(expected_names),
        "num_expected": len(expected_names),
        "num_present": len(bank),
        "missing_mask": missing,
        "zero_mask": zero,
        "nonfinite_mask": nonfinite,
        "all_present": not missing,
        "all_finite": not nonfinite,
        "nontrivial": len(zero) < len(expected_names),
    }


def _update_no_worse(ae: Dict[str, Any], a12: Dict[str, Any]) -> Dict[str, Any]:
    if ae.get("status") != "ok" or a12.get("status") != "ok":
        return {"status": "unavailable", "no_worse_than_baseline": False}
    ae_abs = float(ae["max_abs_update_diff"])
    a12_abs = float(a12["max_abs_update_diff"])
    ae_rel = float(ae["max_rel_update_diff"])
    a12_rel = float(a12["max_rel_update_diff"])
    if bool(a12.get("within_tol")):
        ok = bool(ae.get("within_tol"))
        criterion = "A1_A2_exact_requires_A_E_within_tol"
    else:
        ok = bool(
            ae_abs <= max(a12_abs * BASELINE_REL_SLACK, 1e-6) + BASELINE_ABS_ATOL
            and ae_rel <= a12_rel * BASELINE_REL_SLACK + BASELINE_REL_ATOL
        )
        criterion = "A_E_update_error_no_worse_than_A1_A2_slack"
    return {
        "status": "ok",
        "criterion": criterion,
        "baseline_slack": BASELINE_REL_SLACK,
        "ae_over_a12_abs": ae_abs / max(a12_abs, BASELINE_ABS_ATOL),
        "ae_over_a12_rel": ae_rel / max(a12_rel, BASELINE_REL_ATOL),
        "no_worse_than_baseline": ok,
    }


def _run_condition(
    *,
    label: str,
    model,
    replayer,
    trace,
    decoder_kwargs: Dict[str, Any],
    advantages: List[float],
    devices: List[torch.device],
    expected_names: List[str],
    callbacks: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    _zero_grads(model)
    _reset_peaks(devices)
    run = run_one_short_condition(
        label=label,
        model=model,
        replayer=replayer,
        trace=trace,
        decoder_kwargs=decoder_kwargs,
        old_logp=torch.zeros((), device=devices[0], dtype=torch.float32),
        advantage=torch.ones((), device=devices[0], dtype=torch.float32),
        clip_epsilon=0.2,
        device=devices[0],
        use_selective_offload=False,
        use_fp32=False,
        threshold_bytes=0,
        pin_memory=False,
        protect_attn_bias=True,
        verify_unpack_values=False,
        sync_backward=True,
        detect_anomaly=False,
        pristine_no_hooks=True,
        oracle_block_advantages=advantages,
        replay_score_callbacks=callbacks,
        install_diagnostic_probes=False,
    )
    run["per_gpu_memory"] = _sync_and_memory(devices)
    run["lora_gradients"] = _lora_grad_bank(model, expected_names)
    run["lora_gradient_masks"] = _gradient_masks(expected_names, run["lora_gradients"])
    return run


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    gradients_path = output_dir / GRADIENT_ARTIFACT_FILENAME
    if output_path.exists() or gradients_path.exists():
        raise FileExistsError("refusing to overwrite existing oracle artifacts")
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    report: Dict[str, Any] = {
        "oracle": "A1_A2_pristine_vs_E_two_gpu_live_cache_exactness",
        "output_filename": OUTPUT_FILENAME,
        "gradient_artifact_filename": GRADIENT_ARTIFACT_FILENAME,
        "same_process": True,
        "same_model_revision": True,
        "case_b_lora_and_projector_comparison": True,
        "objective": "L_total = L_GRPO only; fixed short-fixture per-block advantages; old=current.detach()",
        "carry_cache": True,
        "detached_kv": False,
        "bfix": False,
        "truncated_bptt": False,
        "attn_implementation": "sdpa",
        "optimizer_step_on_model": False,
    }
    artifact = None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("requires CUDA_VISIBLE_DEVICES=2,3 (visible cuda:0,cuda:1)")
        config = load_resolved_config(args.config)
        model, tokenizer, processor, revision = build_policy(config, devices[0])
        model.eval()
        replayer = build_rollout_replayer(model, tokenizer, config)
        pairs = load_verified_pairs(config, "train")
        pair = pairs[int(args.sample_index)]
        fixture = build_short_sequence_gradient_oracle_fixture(
            pair=pair,
            processor=processor,
            device=devices[0],
            config=config,
            target_max_prompt_tokens=int(args.target_max_prompt_tokens),
            image_max_side_candidates=(int(args.image_max_side),),
        )
        trace = fixture["rollout_trace"]
        decoder_kwargs = fixture["decoder_kwargs"]
        advantages = [float(x) for x in fixture["meta"]["oracle_block_advantages"]]
        artifact = snapshot_trainable_init(model, include_projector=True)
        expected_names = list(artifact["tensor_names"])
        lora_names = [name for name in expected_names if "lora_" in name]
        projector_names = [name for name in expected_names if "mlp1." in name]
        if len(lora_names) != 504 or len(projector_names) != 6 or len(expected_names) != 510:
            raise RuntimeError(
                "expected 504 LoRA + 6 projector Case-B tensors, got "
                f"{len(lora_names)} + {len(projector_names)}"
            )
        report.update(
            {
                "model_revision": revision,
                "config_path": config["_config_path"],
                "fixture": fixture["meta"],
                "fixed_advantages": advantages,
                "old_logp_policy": "per-block old_logp=current.detach() inside oracle loss",
                "trainable_parameter_names": expected_names,
                "lora_parameter_count": len(lora_names),
                "projector_parameter_count": len(projector_names),
                "trainable_parameter_count": len(expected_names),
            }
        )

        runs: Dict[str, Dict[str, Any]] = {}
        restores: Dict[str, Any] = {}
        for label in ("A1_pristine", "A2_pristine"):
            restores[label] = restore_trainable_init(
                model, artifact, device=devices[0], clear_cpu_refs=False
            )
            model.eval()
            runs[label] = _run_condition(
                label=label, model=model, replayer=replayer, trace=trace,
                decoder_kwargs=decoder_kwargs, advantages=advantages, devices=devices,
                expected_names=expected_names,
            )
            _zero_grads(model)
            gc.collect()

        restores["E_two_gpu_live_cache"] = restore_trainable_init(
            model, artifact, device=devices[0], clear_cpu_refs=False
        )
        shard = shard_locateanything_decoder_two_gpu(
            model, first_device=devices[0], second_device=devices[1]
        )
        model.eval()
        kv_recorder = _ReplayKVRecorder(DecoderShardLayout(devices[0], devices[1], 18))
        resolved_decoder = resolve_locateanything_qwen_decoder(model).decoder
        resolved_decoder._chestxray8_two_gpu_boundary_events = []
        runs["E_two_gpu_live_cache"] = _run_condition(
            label="E_two_gpu_live_cache", model=model, replayer=replayer, trace=trace,
            decoder_kwargs=decoder_kwargs, advantages=advantages, devices=devices,
            expected_names=expected_names,
            callbacks={"on_scored_block_cache": kv_recorder.record},
        )
        runs["E_two_gpu_live_cache"]["live_kv"] = kv_recorder.report()
        runs["E_two_gpu_live_cache"]["cross_device_boundary"] = _cross_device_boundary_report(model)

        # run_one_short_condition retains its historical ``selected_grads``
        # tensor payload.  Capture the exact offending paths for audit, then
        # remove it: all A1/A2/E full gradient values are written only to the
        # dedicated .pt artifact below, never to JSON.
        report["report_serialization"] = {
            "non_json_safe_fields": _non_json_safe_fields(runs, "report.runs")
        }
        banks = {label: run.pop("lora_gradients") for label, run in runs.items()}
        for run in runs.values():
            selected = run.pop("selected_grads", None)
            if isinstance(selected, dict):
                run["selected_grads_json_omitted"] = {
                    "reason": "full tensors are stored in two_gpu_exactness_oracle_gradients.pt",
                    "tensor_count": len(selected),
                }
        compare_a12 = {
            "forward": compare_forward_scalars(runs["A1_pristine"], runs["A2_pristine"]),
            "gradients": compare_grad_dicts(banks["A1_pristine"], banks["A2_pristine"]),
        }
        compare_ae = {
            "forward": compare_forward_scalars(runs["A1_pristine"], runs["E_two_gpu_live_cache"]),
            "gradients": compare_grad_dicts(banks["A1_pristine"], banks["E_two_gpu_live_cache"]),
        }
        normalized = normalize_ab_by_a12_baseline(
            compare_ae["gradients"], compare_a12["gradients"]
        )
        update_a12 = _optimizer_update_compare_cpu(
            model, banks["A1_pristine"], banks["A2_pristine"], lr=float(args.adamw_lr)
        )
        update_ae = _optimizer_update_compare_cpu(
            model, banks["A1_pristine"], banks["E_two_gpu_live_cache"], lr=float(args.adamw_lr)
        )
        update_normalized = _update_no_worse(update_ae, update_a12)
        e_run = runs["E_two_gpu_live_cache"]
        forward = compare_ae["forward"]
        forward_exact = bool(
            forward.get("status") == "ok"
            and float(forward.get("total_logp_abs_diff") or 0.0) <= 1e-5
            and float(forward.get("max_block_abs_diff") or 0.0) <= 1e-5
            and float(forward.get("loss_abs_diff") or 0.0) <= 1e-5
        )
        masks_ok = all(
            run["lora_gradient_masks"]["all_present"]
            and run["lora_gradient_masks"]["all_finite"]
            and run["lora_gradient_masks"]["nontrivial"]
            for run in runs.values()
        )
        live_ok = bool((e_run.get("live_kv") or {}).get("all_replay_caches_live_and_colocated"))
        boundary_ok = bool((e_run.get("cross_device_boundary") or {}).get("all_transfers_cross_cuda_0_to_cuda_1_with_autograd")) and bool((e_run.get("cross_device_boundary") or {}).get("gradient_reaches_early_and_late_lora"))
        acceptance = {
            "forward_exact": forward_exact,
            "all_504_lora_and_6_projector_gradients_present_finite_nontrivial": masks_ok,
            "ae_no_worse_than_a12_gradient_baseline": bool(normalized.get("ab_no_worse_than_a12_baseline")),
            "adamw_update_no_worse_than_baseline": bool(update_normalized.get("no_worse_than_baseline")),
            "live_kv_pass": live_ok,
            "cross_device_autograd_pass": boundary_ok,
        }
        acceptance["passed"] = all(acceptance.values())
        torch.save({"A1": banks["A1_pristine"], "A2": banks["A2_pristine"], "E": banks["E_two_gpu_live_cache"]}, gradients_path)
        report.update(
            {
                "restores": restores,
                "shard": shard,
                "runs": runs,
                "compare_A1_A2": compare_a12,
                "compare_A_E": compare_ae,
                "baseline_normalized_A_E": normalized,
                "adamw_update_A1_A2": update_a12,
                "adamw_update_A_E": update_ae,
                "adamw_update_normalized_A_E": update_normalized,
                "gradient_artifact": str(gradients_path),
                "acceptance": acceptance,
                "status": "passed" if acceptance["passed"] else "failed_diagnostics",
            }
        )
    except Exception as exc:
        report.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        _write_report_safely(output_path, report)
        raise
    finally:
        if artifact is not None and isinstance(artifact.get("tensors"), dict):
            artifact["tensors"].clear()
    _write_report_safely(output_path, report)
    print(json.dumps({"status": report["status"], "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
