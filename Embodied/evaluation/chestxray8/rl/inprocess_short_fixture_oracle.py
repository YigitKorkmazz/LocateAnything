"""In-process Level-1 short-fixture A1/A2 + A/B gradient oracle.

Same model instance, same CUDA process, same fixture tensors.
A1/A2 establish pristine repeatability; A/B tests selective offload (G0 SDPA).
"""

from __future__ import annotations

import gc
import json
import math
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from rl.nan_grad_diagnostics import compare_forward_scalars, compare_grad_dicts
from rl.short_ab_nan_compare import (
    NONTRIVIAL_GRAD_NORM_EPS,
    ORACLE_LOSS_ABS_EPS,
    _optimizer_update_compare_cpu,
    _selected_grads,
    _zero_grads,
    restore_trainable_init,
    run_one_short_condition,
    snapshot_trainable_init,
)
# Allow A/B to exceed A1/A2 by a small relative slack when baseline is noisy.
BASELINE_REL_SLACK = 1.05
BASELINE_REL_ATOL = 1e-6
BASELINE_ABS_ATOL = 1e-6


def _release_after_run(model: nn.Module, device: torch.device) -> None:
    """Clear grads and drop graph/cache temporaries before restore."""
    _zero_grads(model)
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _compact_run(run: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(run)
    grads = compact.pop("selected_grads", None)
    if isinstance(grads, dict):
        compact["selected_grads_summary"] = {
            "num_tensors": len(grads),
            "names_head": sorted(grads.keys())[:16],
        }
    return compact


def _loss_ok(run: Dict[str, Any]) -> bool:
    if run.get("error_type") == "degenerate_zero_loss":
        return False
    loss = ((run.get("pre_backward_scalars") or {}).get("loss") or {}).get("value")
    try:
        return abs(float(loss)) > ORACLE_LOSS_ABS_EPS
    except (TypeError, ValueError):
        return False


def _nontrivial(grad_cmp: Dict[str, Any]) -> bool:
    return bool(
        not grad_cmp.get("all_gradients_zero")
        and float(grad_cmp.get("global_grad_norm_a") or 0.0) > NONTRIVIAL_GRAD_NORM_EPS
        and float(grad_cmp.get("global_grad_norm_b") or 0.0) > NONTRIVIAL_GRAD_NORM_EPS
        and int(grad_cmp.get("num_a_nonzero_params") or 0) >= 1
        and int(grad_cmp.get("num_b_nonzero_params") or 0) >= 1
    )


def normalize_ab_by_a12_baseline(
    ab_grads: Dict[str, Any], a12_grads: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare A/B error against pristine A1/A2 repeatability baseline."""
    ab_rel = ab_grads.get("global_rel_l2_error_finite")
    a12_rel = a12_grads.get("global_rel_l2_error_finite")
    ab_abs = float(ab_grads.get("max_abs_error_finite") or 0.0)
    a12_abs = float(a12_grads.get("max_abs_error_finite") or 0.0)
    ab_rel_f = float(ab_rel) if ab_rel is not None and math.isfinite(float(ab_rel)) else None
    a12_rel_f = (
        float(a12_rel) if a12_rel is not None and math.isfinite(float(a12_rel)) else None
    )
    if ab_rel_f is None or a12_rel_f is None:
        ratio = None
    else:
        ratio = ab_rel_f / max(a12_rel_f, BASELINE_REL_ATOL)

    a12_exact = bool(a12_grads.get("within_tol"))
    if a12_exact:
        no_worse = bool(ab_grads.get("within_tol"))
        criterion = "a12_exact_requires_ab_within_tol"
    else:
        # Baseline is noisy: A/B must not exceed A1/A2 by more than slack.
        rel_ok = (
            ab_rel_f is not None
            and a12_rel_f is not None
            and ab_rel_f <= a12_rel_f * BASELINE_REL_SLACK + BASELINE_REL_ATOL
        )
        abs_ok = ab_abs <= max(a12_abs * BASELINE_REL_SLACK, 1e-4) + BASELINE_ABS_ATOL
        no_worse = bool(rel_ok and abs_ok)
        criterion = "ab_global_rel_l2_and_max_abs_le_a12_times_slack"

    return {
        "a12_within_tol": a12_exact,
        "ab_within_tol": bool(ab_grads.get("within_tol")),
        "a12_global_rel_l2": a12_rel_f,
        "ab_global_rel_l2": ab_rel_f,
        "ab_over_a12_global_rel_l2": ratio,
        "a12_max_abs_error": a12_abs,
        "ab_max_abs_error": ab_abs,
        "ab_over_a12_max_abs": (
            ab_abs / max(a12_abs, BASELINE_ABS_ATOL) if a12_abs > 0 or ab_abs > 0 else 0.0
        ),
        "baseline_rel_slack": BASELINE_REL_SLACK,
        "criterion": criterion,
        "ab_no_worse_than_a12_baseline": no_worse,
    }


def interpret_inprocess_oracle(
    *,
    a12_match: bool,
    ab_match: bool,
    ab_no_worse: bool,
) -> str:
    if not a12_match:
        return (
            "A1/A2 mismatch => underlying CUDA backward nondeterminism; "
            "exact bitwise equality is unattainable under current kernels; "
            "numerical baseline tolerance established from A1/A2. "
            + (
                "A/B is no worse than that baseline."
                if ab_no_worse
                else "A/B exceeds the pristine repeatability baseline."
            )
        )
    if ab_match:
        return "A1/A2 match and A/B match => Level-1 exactness passes."
    return (
        "A1/A2 match and A/B mismatch => selective offload causes the mismatch."
    )


def run_inprocess_short_fixture_gradient_oracle(
    *,
    model: nn.Module,
    replayer,
    decoder_kwargs: Dict[str, Any],
    trace,
    device: torch.device,
    oracle_block_advantages: Sequence[float],
    threshold_bytes: int,
    pin_memory: bool,
    protect_attn_bias: bool,
    verify_unpack_values: bool = False,
    sync_backward: bool = True,
    detect_anomaly: bool = False,
    clip_epsilon: float = 0.2,
    lr: float = 1e-5,
    compare_optimizer_update: bool = True,
    allow_storage_dedup: bool = False,
    full_unpack_verify: bool = False,
    include_projector: bool = False,
    identity_guard_level_b: str = "G0",
) -> Dict[str, Any]:
    """Single-process A1/A2 pristine control + A/B selective-offload oracle."""
    model.eval()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    init_artifact = snapshot_trainable_init(
        model, include_projector=bool(include_projector)
    )
    advs = [float(x) for x in oracle_block_advantages]
    dummy_old = torch.tensor(0.0, device=device, dtype=torch.float32)
    dummy_adv = torch.tensor(1.0, device=device, dtype=torch.float32)

    runs: Dict[str, Any] = {}
    grad_bank: Dict[str, Dict[str, torch.Tensor]] = {}

    schedule = (
        ("A1_pristine", True, False, "G0"),
        ("A2_pristine", True, False, "G0"),
        ("B_selective_offload_bf16", False, True, str(identity_guard_level_b or "G0")),
    )

    for label, pristine, offload, guard in schedule:
        print(f"=== IN-PROCESS ORACLE {label} ===")
        restore_report = restore_trainable_init(
            model, init_artifact, device=device, clear_cpu_refs=False
        )
        model.eval()
        _zero_grads(model)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        result = run_one_short_condition(
            label=label,
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            old_logp=dummy_old,
            advantage=dummy_adv,
            clip_epsilon=float(clip_epsilon),
            device=device,
            use_selective_offload=bool(offload),
            use_fp32=False,
            threshold_bytes=int(threshold_bytes),
            pin_memory=bool(pin_memory),
            protect_attn_bias=bool(protect_attn_bias),
            verify_unpack_values=bool(verify_unpack_values) and bool(offload),
            sync_backward=bool(sync_backward),
            detect_anomaly=bool(detect_anomaly),
            pristine_no_hooks=bool(pristine),
            cuda_sync_localize=False,
            run_lora_sanity=False,
            oracle_block_advantages=advs,
            identity_guard_level=str(guard),
            allow_storage_dedup=bool(allow_storage_dedup),
            full_unpack_verify=bool(full_unpack_verify) and bool(offload),
            probe_down_proj=False,
            oracle_repro_seed=None,
            # Keep A1/A2/B instrumentation matched for gradient equivalence.
            install_diagnostic_probes=False,
        )
        result["trainable_init_restore"] = {
            "num_restored": restore_report.get("num_restored"),
            "bytes_restored": restore_report.get("bytes_restored"),
            "storage_pointer_consistent": restore_report.get(
                "storage_pointer_consistent"
            ),
        }
        grads = result.get("selected_grads") or _selected_grads(model)
        # Ensure CPU copies even if run_one_short_condition omitted them.
        if not grads and result.get("backward_status") == "ok":
            grads = _selected_grads(model)
            result["selected_grads"] = grads
        grad_bank[label] = {
            k: v.detach().float().cpu().clone() for k, v in grads.items()
        }
        runs[label] = _compact_run(result)
        print(
            json.dumps(
                {
                    "event": "inprocess_condition_done",
                    "label": label,
                    "status": result.get("status"),
                    "forward": result.get("forward_status"),
                    "backward": result.get("backward_status"),
                    "num_grad_tensors": len(grad_bank[label]),
                    "loss": (
                        (result.get("pre_backward_scalars") or {})
                        .get("loss", {})
                        .get("value")
                    ),
                },
                indent=2,
                default=str,
            )
        )
        _release_after_run(model, device)

    # Alias A := A1 for A/B comparison (same pristine path).
    grad_bank["A_no_offload_bf16"] = grad_bank["A1_pristine"]
    runs["A_no_offload_bf16"] = dict(runs["A1_pristine"])
    runs["A_no_offload_bf16"]["label"] = "A_no_offload_bf16"
    runs["A_no_offload_bf16"]["alias_of"] = "A1_pristine"

    compare_a12 = {
        "forward": compare_forward_scalars(
            runs["A1_pristine"], runs["A2_pristine"]
        ),
        "grads": compare_grad_dicts(
            grad_bank["A1_pristine"], grad_bank["A2_pristine"]
        ),
    }
    compare_ab = {
        "forward": compare_forward_scalars(
            runs["A_no_offload_bf16"], runs["B_selective_offload_bf16"]
        ),
        "grads": compare_grad_dicts(
            grad_bank["A_no_offload_bf16"], grad_bank["B_selective_offload_bf16"]
        ),
    }
    baseline = normalize_ab_by_a12_baseline(
        compare_ab["grads"], compare_a12["grads"]
    )

    a1_ok = bool(
        (runs["A1_pristine"].get("trainable_grad_report") or {}).get(
            "all_trainable_grads_finite"
        )
    )
    a2_ok = bool(
        (runs["A2_pristine"].get("trainable_grad_report") or {}).get(
            "all_trainable_grads_finite"
        )
    )
    b_ok = bool(
        (runs["B_selective_offload_bf16"].get("trainable_grad_report") or {}).get(
            "all_trainable_grads_finite"
        )
    )
    a12_match = bool((compare_a12.get("grads") or {}).get("within_tol"))
    ab_match = bool((compare_ab.get("grads") or {}).get("within_tol"))
    ab_no_worse = bool(baseline.get("ab_no_worse_than_a12_baseline"))
    nontrivial = _nontrivial(compare_ab["grads"])
    loss_ok = (
        _loss_ok(runs["A1_pristine"])
        and _loss_ok(runs["A2_pristine"])
        and _loss_ok(runs["B_selective_offload_bf16"])
    )
    fwd = compare_ab.get("forward") or {}
    fwd_ab = bool(
        fwd.get("status") == "ok"
        and float(fwd.get("loss_abs_diff") or 0.0) <= 1e-5
        and float(fwd.get("total_logp_abs_diff") or 0.0) <= 1e-5
        and float(fwd.get("max_block_abs_diff") or 0.0) <= 1e-5
    )

    interpretation = interpret_inprocess_oracle(
        a12_match=a12_match, ab_match=ab_match, ab_no_worse=ab_no_worse
    )

    optimizer_cmp: Dict[str, Any] = {"status": "skipped"}
    if compare_optimizer_update and a1_ok and b_ok and ab_no_worse and nontrivial:
        optimizer_cmp = _optimizer_update_compare_cpu(
            model,
            grad_bank["A_no_offload_bf16"],
            grad_bank["B_selective_offload_bf16"],
            lr=float(lr),
        )
        optimizer_cmp["gate"] = "ab_no_worse_than_a12_baseline"
    elif compare_optimizer_update:
        optimizer_cmp = {
            "status": "skipped_ab_worse_than_baseline_or_nonfinite",
            "ab_no_worse_than_a12_baseline": ab_no_worse,
            "A1_finite": a1_ok,
            "B_finite": b_ok,
            "note": (
                "refusing optimizer.step until A/B is no worse than A1/A2 baseline "
                "and grads are finite/nontrivial"
            ),
        }

    exactness = bool(
        a1_ok
        and a2_ok
        and b_ok
        and loss_ok
        and nontrivial
        and fwd_ab
        and ab_no_worse
        and (
            (not compare_optimizer_update)
            or optimizer_cmp.get("within_tol") is True
            or optimizer_cmp.get("status", "").startswith("skipped")
        )
    )

    # Drop CPU payload after oracle completes.
    if isinstance(init_artifact.get("tensors"), dict):
        init_artifact["tensors"].clear()
    del init_artifact
    grad_bank.clear()
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    b_stats = (
        (runs["B_selective_offload_bf16"].get("selective_offload_summary") or {}).get(
            "stats"
        )
        or {}
    )
    FULL_A_UNAVAILABLE_REASON = (
        "pristine no-offload FP32-LoRA activation graph exceeds practical "
        "single-RTX-3090 capacity; CUDA driver reports invalid argument near "
        "decoder layer 30."
    )

    return {
        "mode": "short_sequence_ab_gradient_oracle_inprocess",
        "process_isolation": False,
        "same_model_instance": True,
        "same_cuda_process": True,
        "identity_guard_level_b": str(identity_guard_level_b or "G0"),
        "oracle_block_advantages": list(advs),
        "runs": runs,
        "compare_A1_A2": compare_a12,
        "compare_A_B": compare_ab,
        "baseline_normalized_A_B": baseline,
        "interpretation": {
            "summary": interpretation,
            "a12_match": a12_match,
            "ab_match": ab_match,
            "ab_no_worse_than_a12_baseline": ab_no_worse,
        },
        "optimizer_update_A_vs_B": optimizer_cmp,
        "acceptance": {
            "level": "1_short_fixture_gradient_oracle_inprocess",
            "short_fixture_gradient_exactness": exactness,
            "a12_pristine_repeatability_within_tol": a12_match,
            "ab_within_tol": ab_match,
            "ab_no_worse_than_a12_baseline": ab_no_worse,
            "forward_match_A_B": fwd_ab,
            "grad_match_A1_A2": a12_match,
            "grad_match_A_B": ab_match,
            "optimizer_update_match": optimizer_cmp.get("within_tol"),
            "optimizer_update_required": bool(compare_optimizer_update) and ab_no_worse,
            "nontrivial_gradient_required": True,
            "nontrivial_gradients": nontrivial,
            "nondegenerate_loss": loss_ok,
            "trainable_gradients_finite_A1": a1_ok,
            "trainable_gradients_finite_A2": a2_ok,
            "trainable_gradients_finite_B": b_ok,
            "global_grad_norm_a12_a": (compare_a12.get("grads") or {}).get(
                "global_grad_norm_a"
            ),
            "global_grad_norm_a12_b": (compare_a12.get("grads") or {}).get(
                "global_grad_norm_b"
            ),
            "global_rel_l2_a12": (compare_a12.get("grads") or {}).get(
                "global_rel_l2_error_finite"
            ),
            "global_rel_l2_ab": (compare_ab.get("grads") or {}).get(
                "global_rel_l2_error_finite"
            ),
            "ab_over_a12_global_rel_l2": baseline.get("ab_over_a12_global_rel_l2"),
            "num_grad_tensors_compared": (compare_ab.get("grads") or {}).get(
                "num_compared"
            ),
            "offloaded_tensors_B": b_stats.get("offloaded_tensors"),
            "sdpa_identity_pack_calls_B": b_stats.get("sdpa_identity_pack_calls"),
            "detached_kv": False,
            "bfix": False,
            "truncated_bptt": False,
            "full_production_no_offload_reference_feasible": False,
            "full_production_no_offload_reference_unavailable_reason": (
                FULL_A_UNAVAILABLE_REASON
            ),
            "passes": exactness,
            "interpretation": interpretation,
        },
    }
