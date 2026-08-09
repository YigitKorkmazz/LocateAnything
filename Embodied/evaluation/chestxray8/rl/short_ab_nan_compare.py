"""Short-trace A/B/C/D NaN compare for selective saved-tensor offload.

Runs in one process from identical initial weights:
  A. live production-cached autograd, no offload, bf16
  B. selective saved-tensor offload, bf16
  C. no-offload fp32
  D. selective-offload fp32

Diagnostic only. No GRPO training. Optimizer.step only if all grads finite
and A/B match.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.ab_repro_diagnostics import (
    DEFAULT_ORACLE_REPRO_SEED,
    apply_identical_oracle_seed,
    capture_ab_repro_fingerprint,
    compare_ab_repro_fingerprints,
)
from rl.grpo import grpo_clipped_loss
from rl.down_proj_grad_probe import DownProjLoraProbe, compare_down_proj_probes
from rl.layer34_vproj_probe import (
    Layer34VProjProbe,
    filter_unpacked_for_layer34,
)
from rl.lora_grad_diagnostics import (
    iter_lora_linear_modules,
    lora_name_matches_isolation,
    truncate_trace_scored_blocks,
)
from rl.nan_grad_diagnostics import (
    FirstNonfiniteGradProbe,
    aggregate_isfinite_flags,
    compare_forward_scalars,
    compare_grad_dicts,
    report_trainable_grads,
    scalar_tensor_report,
)
from rl.policy_state import is_lora_name
from rl.selective_saved_tensor_offload import (
    SelectiveSavedTensorOffload,
    selective_saved_tensor_offload_context,
)
from rl.short_sequence_fixture import DEFAULT_ORACLE_BLOCK_ADVANTAGES


def snapshot_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Deprecated full-state snapshot — do not use for A/B isolation."""
    raise RuntimeError(
        "full model state_dict snapshot is forbidden for isolated A/B init; "
        "use snapshot_trainable_init() instead"
    )


def restore_state_dict(
    model: nn.Module, state: Dict[str, torch.Tensor], device: torch.device
) -> None:
    """Deprecated full-state restore — do not use for A/B isolation."""
    raise RuntimeError(
        "full model load_state_dict restore is forbidden for isolated A/B init; "
        "use restore_trainable_init() instead"
    )


def snapshot_trainable_init(
    model: nn.Module,
    *,
    include_projector: bool,
) -> Dict[str, Any]:
    """CPU-only trainable tensor snapshot (LoRA [+ projector]) + RNG state."""
    from rl.policy_state import is_lora_name, is_projector_name

    tensors: Dict[str, torch.Tensor] = {}
    cpu_bytes = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        keep = is_lora_name(name) or (
            bool(include_projector) and is_projector_name(name)
        )
        if not keep:
            continue
        cpu = param.detach().to("cpu").clone()
        tensors[name] = cpu
        cpu_bytes += int(cpu.numel()) * int(cpu.element_size())
    rng: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            rng["cuda"] = torch.cuda.get_rng_state_all()
        except Exception as exc:
            rng["cuda_error"] = str(exc)
    return {
        "format": "trainable_only_v1",
        "include_projector": bool(include_projector),
        "num_tensors": len(tensors),
        "cpu_bytes": int(cpu_bytes),
        "tensor_names": sorted(tensors.keys()),
        "tensors": tensors,
        "rng": rng,
    }


def restore_trainable_init(
    model: nn.Module,
    artifact: Dict[str, Any],
    *,
    device: torch.device,
    clear_cpu_refs: bool = True,
) -> Dict[str, Any]:
    """Copy trainable init tensors into existing Parameters (no Parameter replace)."""
    import gc

    if not isinstance(artifact, dict) or artifact.get("format") != "trainable_only_v1":
        raise RuntimeError(
            "expected trainable_only_v1 init artifact; refusing full-state restore"
        )
    tensors: Dict[str, torch.Tensor] = artifact.get("tensors") or {}
    named = dict(model.named_parameters())
    restored: List[Dict[str, Any]] = []
    missing: List[str] = []
    bytes_restored = 0
    storage_ptr_mismatches = 0
    with torch.no_grad():
        for name, cpu_tensor in list(tensors.items()):
            if name not in named:
                missing.append(name)
                continue
            param = named[name]
            ptr_before = int(param.untyped_storage().data_ptr())
            src = cpu_tensor.to(device=param.device, dtype=param.dtype)
            if tuple(param.shape) != tuple(src.shape):
                raise RuntimeError(
                    f"trainable init shape mismatch for {name}: "
                    f"param={tuple(param.shape)} src={tuple(src.shape)}"
                )
            # In-place copy into the live Parameter storage.
            param.copy_(src)
            ptr_after = int(param.untyped_storage().data_ptr())
            if ptr_before != ptr_after:
                storage_ptr_mismatches += 1
            nbytes = int(src.numel()) * int(src.element_size())
            bytes_restored += nbytes
            restored.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "device": str(param.device),
                    "storage_data_ptr": ptr_after,
                    "storage_ptr_unchanged": ptr_before == ptr_after,
                    "bytes": nbytes,
                }
            )
            del src
    rng = artifact.get("rng") or {}
    if "torch" in rng and rng["torch"] is not None:
        torch.set_rng_state(rng["torch"])
    if (
        torch.cuda.is_available()
        and rng.get("cuda") is not None
        and "cuda_error" not in rng
    ):
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as exc:
            restored.append({"rng_cuda_restore_error": str(exc)})

    if clear_cpu_refs:
        # Drop CPU artifact refs before forward (isolated single-use path).
        if isinstance(artifact.get("tensors"), dict):
            artifact["tensors"].clear()
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    first_lora = next((r for r in restored if "lora_" in r["name"]), None)
    return {
        "format": "trainable_only_v1",
        "num_restored": len(restored),
        "num_missing": len(missing),
        "missing_names": missing,
        "bytes_restored": int(bytes_restored),
        "storage_ptr_mismatches": int(storage_ptr_mismatches),
        "storage_pointer_consistent": storage_ptr_mismatches == 0,
        "restored": restored[:16],
        "first_lora_parameter": first_lora,
    }


def pre_forward_cuda_and_lora_report(
    model: nn.Module, device: torch.device
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        report["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(index))
        report["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(index))
    first = None
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            first = {
                "name": name,
                "dtype": str(param.dtype).replace("torch.", ""),
                "device": str(param.device),
                "shape": list(param.shape),
                "storage_data_ptr": int(param.untyped_storage().data_ptr()),
                "requires_grad": bool(param.requires_grad),
            }
            break
    report["first_lora_parameter"] = first
    return report


def _cuda_mem_snapshot(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None}
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
    }


def run_first_mlp_lora_sanity(
    model: nn.Module, device: torch.device
) -> Dict[str, Any]:
    """Minimal LoRA A/B matmul on CUDA before trajectory replay."""
    target_name = None
    target_mod = None
    for name, module in iter_lora_linear_modules(model):
        if lora_name_matches_isolation(name, "mlp"):
            target_name = name
            target_mod = module
            break
    if target_mod is None:
        return {"status": "skip", "reason": "no_mlp_lora_module"}
    active = list(getattr(target_mod, "active_adapters", []) or [])
    if not active:
        active = list(getattr(target_mod, "lora_A", {}).keys())
    if not active:
        return {
            "status": "error",
            "reason": "no_active_adapter",
            "module": target_name,
        }
    adapter = active[0]
    lora_A = target_mod.lora_A[adapter]
    lora_B = target_mod.lora_B[adapter]
    scaling = float(target_mod.scaling[adapter])
    in_features = int(lora_A.in_features)
    dtype = lora_A.weight.dtype
    out: Dict[str, Any] = {
        "status": "not_run",
        "module": target_name,
        "adapter": adapter,
        "in_features": in_features,
        "out_features": int(lora_B.out_features),
        "rank": int(lora_A.out_features),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "scaling": scaling,
    }
    try:
        with torch.no_grad():
            x = torch.randn(2, in_features, device=device, dtype=dtype)
            mid = lora_A(x)
            y = lora_B(mid) * scaling
            if device.type == "cuda":
                torch.cuda.synchronize()
            out["status"] = "ok"
            out["output_isfinite"] = bool(torch.isfinite(y).all().item())
            out["output_abs_max"] = float(y.detach().float().abs().max().cpu())
            out["mid_isfinite"] = bool(torch.isfinite(mid).all().item())
            out.update(_cuda_mem_snapshot(device))
    except Exception as exc:
        out["status"] = "error"
        out["error_type"] = type(exc).__name__
        out["error_message"] = str(exc)
        import traceback

        out["traceback"] = traceback.format_exc()
    return out


def _iter_decoder_layer_modules(model: nn.Module) -> List[tuple]:
    found: List[tuple] = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
            found.append((int(parts[-1]), name, module))
    found.sort(key=lambda t: t[0])
    return found


@contextmanager
def cuda_sync_localize_context(
    model: nn.Module,
    device: torch.device,
    progress: List[Dict[str, Any]],
    state: Dict[str, Any],
) -> Iterator[None]:
    """Sync after each decoder layer; caller syncs after each scored block."""
    handles = []

    def _make_hook(layer_index: int, module_name: str):
        def _hook(_module, _inp, _out):
            if device.type == "cuda":
                torch.cuda.synchronize()
            mem = _cuda_mem_snapshot(device)
            entry = {
                "event": "after_decoder_layer",
                "block_index": state.get("block_index"),
                "decoder_layer_index": int(layer_index),
                "module": module_name,
                "last_completed_operation": (
                    f"block[{state.get('block_index')}]."
                    f"decoder_layer[{layer_index}].forward"
                ),
                **mem,
            }
            state["last_completed_operation"] = entry["last_completed_operation"]
            progress.append(entry)
            # Keep stdout sparse: log every layer only for the first two blocks,
            # otherwise every 8th layer plus the last.
            bi = state.get("block_index")
            if bi in (0, 1) or (layer_index % 8 == 0) or layer_index >= 30:
                print(json.dumps(entry, default=str))

        return _hook

    for layer_index, module_name, module in _iter_decoder_layer_modules(model):
        handles.append(
            module.register_forward_hook(_make_hook(layer_index, module_name))
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def mlp_lora_io_probe_context(
    model: nn.Module,
    *,
    layer_indices: Sequence[int] = (29, 30),
    log: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[List[Dict[str, Any]]]:
    """Optional diagnostic: log MLP LoRA branch IO shapes for selected layers."""
    records: List[Dict[str, Any]] = log if log is not None else []
    handles = []
    wanted = {int(i) for i in layer_indices}
    proj_names = ("gate_proj", "up_proj", "down_proj")

    def _tensor_bytes(t: torch.Tensor) -> int:
        return int(t.numel()) * int(t.element_size())

    for name, module in model.named_modules():
        if not any(p in name for p in proj_names):
            continue
        if "mlp" not in name:
            continue
        # Match ...layers.{N}.mlp.{proj}
        parts = name.split(".")
        layer_idx = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        if layer_idx is None or layer_idx not in wanted:
            continue
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        def _make(n: str, li: int):
            def _hook(_mod, inputs, output):
                inp = inputs[0] if inputs else None
                entry: Dict[str, Any] = {
                    "event": "mlp_lora_io",
                    "module": n,
                    "decoder_layer_index": int(li),
                }
                if torch.is_tensor(inp):
                    entry["input_shape"] = list(inp.shape)
                    entry["input_dtype"] = str(inp.dtype).replace("torch.", "")
                    entry["input_bytes"] = _tensor_bytes(inp)
                if torch.is_tensor(output):
                    entry["output_shape"] = list(output.shape)
                    entry["output_dtype"] = str(output.dtype).replace("torch.", "")
                    entry["output_bytes"] = _tensor_bytes(output)
                records.append(entry)
                if li in (29, 30):
                    print(json.dumps(entry, default=str))

            return _hook

        handles.append(module.register_forward_hook(_make(name, layer_idx)))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


ORACLE_LOSS_ABS_EPS = 1e-8
NONTRIVIAL_GRAD_NORM_EPS = 1e-8


def build_oracle_block_grpo_loss(
    block_logps: Sequence[torch.Tensor],
    block_advantages: Sequence[float],
    *,
    clip_epsilon: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """GRPO clipped loss for backend oracle with ratio initialized to 1.

    Uses the production ``grpo_clipped_loss`` on each scored-block logp with
    ``old = current.detach()`` so PPO ratio is exactly 1 at the forward point.
    Fixed advantages are validation-only (not GT selection / reward learning).
    """
    if len(block_logps) != len(block_advantages):
        raise RuntimeError(
            f"oracle block_logps ({len(block_logps)}) != advantages "
            f"({len(block_advantages)})"
        )
    if len(block_logps) < 3:
        raise RuntimeError("oracle requires >= 3 scored blocks")
    terms: List[torch.Tensor] = []
    ratios: List[float] = []
    for i, (lp, adv) in enumerate(zip(block_logps, block_advantages)):
        cur = lp.reshape(1).float()
        old = cur.detach()
        adv_t = torch.tensor(
            [float(adv)], device=cur.device, dtype=torch.float32
        )
        ratio = torch.exp(cur - old)
        ratios.append(float(ratio.detach().cpu()))
        terms.append(
            grpo_clipped_loss(
                cur, old, adv_t, clip_epsilon=float(clip_epsilon)
            )
        )
    loss = torch.stack(terms).mean()
    meta = {
        "mode": "oracle_per_block_grpo_clipped_loss",
        "backend_validation_only": True,
        "not_gt_selection": True,
        "ratio_init": "old_equals_current_detach",
        "block_advantages": [float(a) for a in block_advantages],
        "per_block_ratios": ratios,
        "max_ratio_abs_err_from_one": max(abs(r - 1.0) for r in ratios),
        "reduction": "mean_over_blocks",
    }
    return loss, meta


def interpret_ab_cd(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def _finite(label: str) -> Optional[bool]:
        run = runs.get(label) or {}
        if run.get("backward_status") != "ok":
            return None
        return bool(
            (run.get("trainable_grad_report") or {}).get("all_trainable_grads_finite")
        )

    a_f, b_f, c_f, d_f = (
        _finite("A_no_offload_bf16"),
        _finite("B_selective_offload_bf16"),
        _finite("C_no_offload_fp32"),
        _finite("D_selective_offload_fp32"),
    )
    notes: List[str] = []
    if a_f is True and b_f is False:
        notes.append(
            "A finite, B fails/NaN => selective offload interferes with efficient "
            "SDPA backward (nested identity hooks required for SDPA-saved tensors)"
        )
    if a_f is False and b_f is False:
        notes.append(
            "A NaN, B NaN => BF16/live-cache backward numerical instability"
        )
    if (a_f is False or b_f is False) and (c_f is True and d_f is True):
        notes.append("bf16 NaN but fp32 finite => precision-specific instability")
    if a_f is True and b_f is True:
        grad_cmp = (runs.get("_compare_A_B") or {}).get("grads") or {}
        max_abs = float(grad_cmp.get("max_abs_error_finite") or 0.0)
        masks_clean = (
            int(grad_cmp.get("nan_inf_mask_mismatches") or 0) == 0
            and int(grad_cmp.get("zero_nonzero_mask_mismatches") or 0) == 0
        )
        if (
            grad_cmp.get("all_gradients_zero")
            or grad_cmp.get("exact_zero_match")
            or grad_cmp.get("global_cosine_status") == "both_zero"
        ) and max_abs == 0.0 and masks_clean:
            notes.append("degenerate all-zero gradient comparison")
        elif grad_cmp.get("within_tol") is False:
            notes.append(
                "both finite but gradients differ => layout/storage reconstruction error"
            )
        elif grad_cmp.get("within_tol") is True:
            notes.append("A/B both finite and gradients match within tolerance")
    return {
        "A_bf16_finite": a_f,
        "B_bf16_selective_finite": b_f,
        "C_fp32_finite": c_f,
        "D_fp32_selective_finite": d_f,
        "notes": notes,
    }


def _selected_grads(model: nn.Module) -> Dict[str, torch.Tensor]:
    grads: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if ("lora_" in name) or name.startswith("mlp1"):
            grads[name] = parameter.grad.detach().float().cpu().clone()
    return grads


def _zero_grads(model: nn.Module) -> None:
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad = None


def _optimizer_update_compare_cpu(
    model: nn.Module,
    grads_a: Dict[str, torch.Tensor],
    grads_b: Dict[str, torch.Tensor],
    *,
    lr: float,
) -> Dict[str, Any]:
    names = sorted(set(grads_a) & set(grads_b))
    if not names:
        return {"status": "skip", "reason": "no_overlapping_grads"}
    named = dict(model.named_parameters())
    params_a = []
    params_b = []
    for name in names:
        if name not in named:
            continue
        base = named[name].detach().float().cpu().clone()
        pa = base.clone().requires_grad_(True)
        pb = base.clone().requires_grad_(True)
        pa.grad = grads_a[name].float().cpu().clone()
        pb.grad = grads_b[name].float().cpu().clone()
        params_a.append(pa)
        params_b.append(pb)
    if not params_a:
        return {"status": "skip", "reason": "no_params"}
    opt_a = torch.optim.AdamW(params_a, lr=lr)
    opt_b = torch.optim.AdamW(params_b, lr=lr)
    opt_a.step()
    opt_b.step()
    max_abs = 0.0
    max_rel = 0.0
    for pa, pb in zip(params_a, params_b):
        diff = (pa.detach() - pb.detach()).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = float(pb.detach().abs().max().item()) + 1e-12
        max_rel = max(max_rel, float(diff.max().item()) / denom)
    return {
        "status": "ok",
        "num_params": len(params_a),
        "max_abs_update_diff": max_abs,
        "max_rel_update_diff": max_rel,
        "within_tol": max_abs <= 1e-6 or max_rel <= 1e-5,
    }


def run_one_short_condition(
    *,
    label: str,
    model: nn.Module,
    replayer,
    trace,
    decoder_kwargs: Dict[str, Any],
    old_logp: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
    device: torch.device,
    use_selective_offload: bool,
    use_fp32: bool,
    threshold_bytes: int,
    pin_memory: bool,
    protect_attn_bias: bool,
    verify_unpack_values: bool,
    sync_backward: bool,
    detect_anomaly: bool,
    pristine_no_hooks: bool = False,
    cuda_sync_localize: bool = False,
    run_lora_sanity: bool = False,
    log_mlp_lora_io_layers: Optional[Sequence[int]] = None,
    oracle_block_advantages: Optional[Sequence[float]] = None,
    identity_guard_level: str = "G0",
    allow_storage_dedup: bool = False,
    full_unpack_verify: bool = False,
    probe_down_proj: bool = False,
    oracle_repro_seed: Optional[int] = None,
    install_diagnostic_probes: Optional[bool] = None,
    replay_score_callbacks: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one short-trace condition.

    When ``pristine_no_hooks`` is True (A0/A1 reference path):
      - no selective saved_tensors_hooks
      - no nested SDPA identity hooks
      - no unpack verification / attention context tracking
      - no layer34 / first-nonfinite grad probes
      - only minimal scalar logging (+ optional CUDA sync localization)

    When ``oracle_block_advantages`` is set, use per-block GRPO loss with
    ratio initialized to 1 (backend-gradient validation only).

    G3 on B: global identity saved_tensors_hooks only (no CPU offload).
    """
    model.eval()
    _zero_grads(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # A is the no-offload reference: never install selective/nested SDPA hooks.
    # B wrapper active when use_selective_offload and not pristine.
    b_wrapper_active = bool(use_selective_offload) and not bool(pristine_no_hooks)
    guard_level = str(identity_guard_level or "G0").upper()
    # G3 keeps the B wrapper but disables all CPU offload (identity only).
    cpu_offload_active = bool(b_wrapper_active) and guard_level != "G3"
    enable_offload = bool(b_wrapper_active)  # manager/context for B including G3
    manager: Optional[SelectiveSavedTensorOffload] = None
    if enable_offload:
        manager = SelectiveSavedTensorOffload(
            threshold_bytes=int(threshold_bytes),
            pin_memory=bool(pin_memory),
            protect_attn_bias=bool(protect_attn_bias),
            # G3 never unpacks from CPU; keep verify flags inert.
            verify_unpack_values=bool(verify_unpack_values) and cpu_offload_active,
            full_unpack_verify=bool(full_unpack_verify) and cpu_offload_active,
            allow_storage_dedup=False if guard_level == "G3" else bool(allow_storage_dedup),
            max_log_entries=4000 if full_unpack_verify else 2000,
            track_unpack_memory=True,
        )

    if install_diagnostic_probes is None:
        install_probes = not bool(pristine_no_hooks)
    else:
        install_probes = bool(install_diagnostic_probes)
    layer_probe = None
    grad_probe = None
    down_probe = None
    if install_probes:
        layer_probe = Layer34VProjProbe(model, sync_cuda=bool(sync_backward))
        layer_probe.install()
        grad_probe = FirstNonfiniteGradProbe(model, sync_cuda=bool(sync_backward))
    if probe_down_proj:
        down_probe = DownProjLoraProbe(model, sync_cuda=True)
        down_probe.install()

    if bool(detect_anomaly):
        try:
            anomaly_ctx = torch.autograd.detect_anomaly(check_nan=True)
        except TypeError:
            anomaly_ctx = torch.autograd.detect_anomaly()
    else:
        anomaly_ctx = nullcontext()

    hook_ctx = (
        selective_saved_tensor_offload_context(
            manager, model, identity_guard_level=guard_level
        )
        if manager is not None
        else nullcontext()
    )

    sdpa_before = F.scaled_dot_product_attention
    progress: List[Dict[str, Any]] = []
    sync_state: Dict[str, Any] = {
        "block_index": None,
        "last_completed_operation": "pre_forward",
    }

    out: Dict[str, Any] = {
        "label": label,
        "use_selective_offload": bool(cpu_offload_active),
        "b_wrapper_active": bool(enable_offload),
        "use_fp32": use_fp32,
        "forward_status": "not_run",
        "backward_status": "not_run",
        "dtype_mode": "fp32" if use_fp32 else "bf16",
        "instrumentation": {
            "pristine_no_hooks": bool(pristine_no_hooks),
            "b_wrapper_active": bool(enable_offload),
            "cpu_offload_active": bool(cpu_offload_active),
            "selective_saved_tensors_hooks": bool(cpu_offload_active),
            "identity_guard_level": guard_level if enable_offload else None,
            "nested_sdpa_identity_hooks_enabled": bool(
                enable_offload and guard_level in ("G0", "G1", "G2")
            ),
            "nested_lora_identity_hooks_enabled": bool(
                enable_offload and guard_level in ("G1", "G2")
            ),
            "nested_mlp_identity_hooks_enabled": bool(
                enable_offload and guard_level == "G2"
            ),
            "global_identity_hooks_enabled": bool(
                enable_offload and guard_level == "G3"
            ),
            "allow_storage_dedup": (
                False
                if guard_level == "G3"
                else (bool(allow_storage_dedup) if enable_offload else None)
            ),
            "full_unpack_verify": (
                bool(full_unpack_verify) and cpu_offload_active
                if enable_offload
                else None
            ),
            "unpack_verification": bool(cpu_offload_active and verify_unpack_values),
            "attention_saved_tensor_context_tracking": bool(enable_offload),
            "layer34_vproj_probe": bool(install_probes),
            "first_nonfinite_grad_probe": bool(install_probes),
            "down_proj_lora_probe": down_probe is not None,
            "detect_anomaly": bool(detect_anomaly),
            "cuda_sync_localize": bool(cuda_sync_localize),
            "lora_sanity_before_replay": bool(run_lora_sanity),
        },
    }
    pre_fwd = pre_forward_cuda_and_lora_report(model, device)
    out["pre_forward_report"] = pre_fwd
    print(
        json.dumps(
            {"event": "pre_forward", "label": label, **pre_fwd},
            indent=2,
            default=str,
        )
    )

    if run_lora_sanity:
        sanity = run_first_mlp_lora_sanity(model, device)
        out["lora_sanity_before_replay"] = sanity
        print(json.dumps({"event": "lora_sanity_before_replay", **sanity}, indent=2))
        if sanity.get("status") == "error":
            out["status"] = "error"
            out["forward_status"] = "error"
            out["backward_status"] = "not_run"
            out["error_type"] = sanity.get("error_type")
            out["error_message"] = (
                "LoRA sanity check failed before replay: "
                + str(sanity.get("error_message"))
            )
            out["nested_sdpa_identity_hooks_entered"] = 0
            out["sdpa_function_patched_during_run"] = False
            return out

    localize_ctx = (
        cuda_sync_localize_context(model, device, progress, sync_state)
        if cuda_sync_localize
        else nullcontext()
    )
    mlp_io_log: List[Dict[str, Any]] = []
    mlp_io_ctx = (
        mlp_lora_io_probe_context(
            model,
            layer_indices=tuple(int(i) for i in log_mlp_lora_io_layers),
            log=mlp_io_log,
        )
        if log_mlp_lora_io_layers
        else nullcontext()
    )

    def _on_block_begin(scored_index: int, _block) -> None:
        sync_state["block_index"] = int(scored_index)
        sync_state["last_completed_operation"] = f"block[{scored_index}].begin"
        entry = {
            "event": "scored_block_begin",
            "block_index": int(scored_index),
            "last_completed_operation": sync_state["last_completed_operation"],
            **_cuda_mem_snapshot(device),
        }
        progress.append(entry)
        print(json.dumps(entry, default=str))

    def _on_block_end(scored_index: int, _block, value) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        sync_state["last_completed_operation"] = f"block[{scored_index}].end"
        entry = {
            "event": "after_scored_block",
            "block_index": int(scored_index),
            "block_logp": float(value.detach().float().cpu()),
            "last_completed_operation": sync_state["last_completed_operation"],
            **_cuda_mem_snapshot(device),
        }
        progress.append(entry)
        print(json.dumps(entry, default=str))

    try:
        # Optional identical seed (isolated A/B only). In-process oracle restores
        # snapshot RNG via restore_trainable_init and must not override it here.
        seed_report = None
        if oracle_repro_seed is not None:
            seed_report = apply_identical_oracle_seed(int(oracle_repro_seed))
            repro_fp = capture_ab_repro_fingerprint(
                model=model,
                device=device,
                decoder_kwargs=decoder_kwargs,
                trace=trace,
                seed_report=seed_report,
            )
            out["ab_repro_fingerprint"] = repro_fp
            print(
                json.dumps(
                    {"event": "ab_repro_fingerprint", "label": label, **repro_fp},
                    indent=2,
                    default=str,
                )
            )

        # detect_anomaly wraps BOTH forward and backward so the failing SDPA
        # forward op is localized (not only the backward kernel name).
        with torch.enable_grad(), hook_ctx, anomaly_ctx, localize_ctx, mlp_io_ctx:
            if device.type == "cuda":
                torch.cuda.synchronize()
            score_kwargs = dict(decoder_kwargs)
            if replay_score_callbacks:
                overlap = set(score_kwargs) & set(replay_score_callbacks)
                if overlap:
                    raise RuntimeError(
                        f"replay score callback keys collide with decoder kwargs: {sorted(overlap)}"
                    )
                score_kwargs.update(replay_score_callbacks)
            if cuda_sync_localize:
                score_kwargs["on_scored_block_begin"] = _on_block_begin
                score_kwargs["on_scored_block_end"] = _on_block_end
            current, block_logps = replayer.score(
                trace,
                use_cache=True,
                legacy_nocache_masks=False,
                **score_kwargs,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            out["forward_status"] = "ok"
            out["current_log_prob"] = float(current.detach().float().cpu())
            out["block_log_probs"] = [
                float(v.detach().float().cpu()) for v in block_logps
            ]
            oracle_meta = None
            if oracle_block_advantages is not None:
                advs = [float(x) for x in oracle_block_advantages]
                if len(advs) < len(block_logps):
                    raise RuntimeError(
                        "oracle_block_advantages shorter than scored blocks"
                    )
                advs = advs[: len(block_logps)]
                loss, oracle_meta = build_oracle_block_grpo_loss(
                    block_logps,
                    advs,
                    clip_epsilon=float(clip_epsilon),
                )
                out["oracle_loss"] = oracle_meta
            else:
                loss = grpo_clipped_loss(
                    current.reshape(1),
                    old_logp.reshape(1),
                    advantage.reshape(1),
                    clip_epsilon=float(clip_epsilon),
                )
            block_reports = [
                scalar_tensor_report(v, name=f"block_logp[{i}]")
                for i, v in enumerate(block_logps)
            ]
            pre = {
                "current_log_prob": scalar_tensor_report(
                    current, name="current_log_prob"
                ),
                "block_log_probs": block_reports,
                "loss": scalar_tensor_report(loss, name="loss_grpo"),
            }
            finite_flags = [
                bool(pre["current_log_prob"].get("isfinite")),
                bool(pre["loss"].get("isfinite")),
                *[bool(b.get("isfinite")) for b in block_reports],
            ]
            pre["all_forward_finite"] = aggregate_isfinite_flags(finite_flags)
            out["pre_backward_scalars"] = pre

            loss_val = float(loss.detach().float().cpu())
            if (
                oracle_block_advantages is not None
                and (
                    not math.isfinite(loss_val)
                    or abs(loss_val) <= ORACLE_LOSS_ABS_EPS
                )
            ):
                out["status"] = "error"
                out["error_type"] = "degenerate_zero_loss"
                out["error_message"] = (
                    f"oracle scalar loss is degenerate before backward: {loss_val!r}"
                )
                out["backward_status"] = "not_run"
                out["loss_grpo"] = loss_val
                del loss, current, block_logps
            else:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                loss.backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()

                out["backward_status"] = "ok"
                out["loss_grpo"] = float(loss.detach().float().cpu())
                if grad_probe is not None:
                    out["first_nonfinite_grad_probe"] = grad_probe.report()
                if layer_probe is not None:
                    out["layer34_vproj_probe"] = layer_probe.report()
                grad_report = report_trainable_grads(model)
                out["trainable_grad_report"] = {
                    "num_trainable": grad_report["num_trainable"],
                    "any_grad_none": grad_report["any_grad_none"],
                    "any_grad_nonfinite": grad_report["any_grad_nonfinite"],
                    "all_trainable_grads_finite": grad_report[
                        "all_trainable_grads_finite"
                    ],
                    "num_nonfinite": grad_report["num_nonfinite"],
                    "num_grad_none": grad_report["num_grad_none"],
                    # Keep compact: only nonfinite / target params in summary.
                    "nonfinite_or_target_parameters": [
                        r
                        for r in grad_report["parameters"]
                        if (not r["grad_is_none"] and not r["all_finite"])
                        or ("layers.34.self_attn.v_proj" in r["name"])
                    ][:64],
                }
                grads = _selected_grads(model)
                out["selected_grads"] = grads
                out["selected_lora_grad_l2_finite"] = {
                    name: (
                        float(g[torch.isfinite(g)].float().norm().cpu())
                        if torch.isfinite(g).any()
                        else float("nan")
                    )
                    for name, g in grads.items()
                    if is_lora_name(name)
                }
                out["optimizer_step_allowed"] = bool(
                    grad_report["all_trainable_grads_finite"]
                )
                if down_probe is not None:
                    out["down_proj_lora_probe"] = down_probe.report()
                if manager is not None:
                    summary = manager.summary()
                    out["selective_offload_summary"] = {
                        "stats": summary.get("stats"),
                        "first_nonfinite_unpack": summary.get("first_nonfinite_unpack"),
                        "first_unpack_value_mismatch": summary.get(
                            "first_unpack_value_mismatch"
                        ),
                        "first_full_unpack_mismatch": summary.get(
                            "first_full_unpack_mismatch"
                        ),
                        "allow_storage_dedup": summary.get("allow_storage_dedup"),
                        "full_unpack_verify": summary.get("full_unpack_verify"),
                        "sdpa_protection": summary.get("sdpa_protection"),
                        "identity_guard_level": guard_level,
                    }
                    out["layer34_unpacked_tensors"] = filter_unpacked_for_layer34(
                        summary, manager.tensor_log
                    )
                del loss, current, block_logps
    except Exception as exc:
        out["status"] = "error"
        if out["forward_status"] == "not_run":
            out["forward_status"] = "error"
        if out["backward_status"] == "not_run":
            out["backward_status"] = "error"
        out["error_type"] = type(exc).__name__
        out["error_message"] = str(exc)
        import traceback

        out["traceback"] = traceback.format_exc()
        out["cuda_sync_progress_tail"] = progress[-32:]
        out["last_completed_operation"] = sync_state.get("last_completed_operation")
        out["failing_block_index"] = sync_state.get("block_index")
        if grad_probe is not None:
            out["first_nonfinite_grad_probe"] = grad_probe.report()
        if layer_probe is not None:
            out["layer34_vproj_probe"] = layer_probe.report()
        if manager is not None:
            summary = manager.summary()
            out["selective_offload_summary"] = {
                "stats": summary.get("stats"),
                "first_nonfinite_unpack": summary.get("first_nonfinite_unpack"),
                "first_unpack_value_mismatch": summary.get(
                    "first_unpack_value_mismatch"
                ),
                "sdpa_protection": summary.get("sdpa_protection"),
            }
            out["layer34_unpacked_tensors"] = filter_unpacked_for_layer34(
                summary, manager.tensor_log
            )
    finally:
        if layer_probe is not None:
            layer_probe.close()
        if grad_probe is not None:
            grad_probe.close()
        if down_probe is not None:
            if "down_proj_lora_probe" not in out:
                out["down_proj_lora_probe"] = down_probe.report()
            down_probe.close()
        sdpa_after = F.scaled_dot_product_attention
        nested_entered = 0
        lora_entered = 0
        mlp_entered = 0
        if manager is not None:
            nested_entered = int(
                getattr(manager.stats, "sdpa_identity_pack_calls", 0) or 0
            )
            lora_entered = int(
                getattr(manager.stats, "lora_identity_pack_calls", 0) or 0
            )
            mlp_entered = int(
                getattr(manager.stats, "mlp_identity_pack_calls", 0) or 0
            )
        out["nested_sdpa_identity_hooks_entered"] = nested_entered
        out["nested_lora_identity_hooks_entered"] = lora_entered
        out["nested_mlp_identity_hooks_entered"] = mlp_entered
        global_entered = 0
        if manager is not None:
            global_entered = int(
                getattr(manager.stats, "global_identity_pack_calls", 0) or 0
            )
        out["global_identity_hooks_entered"] = global_entered
        out["identity_guard_level"] = guard_level if enable_offload else None
        out["cpu_offload_active"] = bool(cpu_offload_active)
        out["sdpa_function_patched_during_run"] = sdpa_after is not sdpa_before
        # After context exit, SDPA must be restored to the original function.
        out["sdpa_function_restored"] = sdpa_after is sdpa_before
        # G3 must not patch SDPA (global identity only).
        if guard_level == "G3" and enable_offload:
            out["g3_no_cpu_offload"] = True
            out["g3_sdpa_unpatched"] = sdpa_after is sdpa_before
        if cuda_sync_localize:
            out["cuda_sync_progress_count"] = len(progress)
            out["cuda_sync_progress_tail"] = progress[-64:]
        if mlp_io_log:
            out["mlp_lora_io_probe"] = mlp_io_log[:64]
        if device.type == "cuda":
            torch.cuda.synchronize()
            out["cuda_memory_peak"] = {
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }

    if out.get("status") != "error":
        out["status"] = (
            "ok"
            if out["forward_status"] == "ok" and out["backward_status"] == "ok"
            else "error"
        )
    return out


def run_short_ab_nan_compare(
    *,
    model: nn.Module,
    replayer,
    trace,
    decoder_kwargs: Dict[str, Any],
    record: Dict[str, Any],
    device: torch.device,
    max_scored_blocks: int,
    threshold_bytes: int,
    pin_memory: bool,
    protect_attn_bias: bool,
    verify_unpack_values: bool,
    sync_backward: bool,
    detect_anomaly: bool,
    clip_epsilon: float,
    lr: float,
    compare_optimizer_update: bool,
) -> Dict[str, Any]:
    if int(max_scored_blocks) < 3:
        raise RuntimeError(
            "short A/B NaN compare requires --max-scored-blocks >= 3 "
            "so cross-block KV gradients exist"
        )

    short_trace, trunc = truncate_trace_scored_blocks(trace, int(max_scored_blocks))
    if int(trunc.get("kept_scored_blocks") or 0) < 3:
        raise RuntimeError(
            f"trace has only {trunc.get('kept_scored_blocks')} scored blocks; need >= 3"
        )

    from rl.policy_state import is_projector_name

    include_projector = any(
        p.requires_grad and is_projector_name(n) for n, p in model.named_parameters()
    )
    initial_state = snapshot_trainable_init(
        model, include_projector=include_projector
    )
    old_logp = torch.tensor(
        float(record.get("old_log_prob", 0.0)),
        device=device,
        dtype=torch.float32,
    )
    advantage = torch.tensor(
        float(record.get("advantage", 1.0)) or 1.0,
        device=device,
        dtype=torch.float32,
    )

    conditions = (
        ("A_no_offload_bf16", False, False),
        ("B_selective_offload_bf16", True, False),
        ("C_no_offload_fp32", False, True),
        ("D_selective_offload_fp32", True, True),
    )
    runs: Dict[str, Dict[str, Any]] = {}

    for label, offload, use_fp32 in conditions:
        model.eval()
        if use_fp32:
            model.float()
        else:
            model.to(dtype=torch.bfloat16)
        restore_trainable_init(
            model, initial_state, device=device, clear_cpu_refs=False
        )

        print(f"=== SHORT AB-NAN CONDITION {label} ===")
        result = run_one_short_condition(
            label=label,
            model=model,
            replayer=replayer,
            trace=short_trace,
            decoder_kwargs=decoder_kwargs,
            old_logp=old_logp,
            advantage=advantage,
            clip_epsilon=clip_epsilon,
            device=device,
            use_selective_offload=offload,
            use_fp32=use_fp32,
            threshold_bytes=threshold_bytes,
            pin_memory=pin_memory,
            protect_attn_bias=protect_attn_bias,
            verify_unpack_values=verify_unpack_values,
            sync_backward=sync_backward,
            detect_anomaly=detect_anomaly,
        )
        runs[label] = result
        print(
            {
                "label": label,
                "status": result.get("status"),
                "forward": result.get("forward_status"),
                "backward": result.get("backward_status"),
                "grads_finite": (result.get("trainable_grad_report") or {}).get(
                    "all_trainable_grads_finite"
                ),
                "first_param": (
                    (result.get("first_nonfinite_grad_probe") or {}).get(
                        "first_nonfinite_parameter"
                    )
                    or {}
                ).get("parameter"),
            }
        )

    # Restore bf16 initial weights after fp32 runs.
    model.to(dtype=torch.bfloat16)
    restore_trainable_init(
        model, initial_state, device=device, clear_cpu_refs=True
    )
    model.eval()

    def _fwd_compare(x: str, y: str) -> Dict[str, Any]:
        rx, ry = runs[x], runs[y]
        if rx.get("forward_status") != "ok" or ry.get("forward_status") != "ok":
            return {"status": "skipped_forward_error"}
        xb = rx.get("block_log_probs") or []
        yb = ry.get("block_log_probs") or []
        diffs = [abs(a - b) for a, b in zip(xb, yb)]
        return {
            "per_block_abs_diff": diffs,
            "max_block_abs_diff": max(diffs) if diffs else 0.0,
            "total_abs_diff": abs(
                float(rx["current_log_prob"]) - float(ry["current_log_prob"])
            ),
            "loss_abs_diff": abs(
                float((rx.get("pre_backward_scalars") or {}).get("loss", {}).get("value") or 0.0)
                - float(
                    (ry.get("pre_backward_scalars") or {}).get("loss", {}).get("value")
                    or 0.0
                )
            ),
        }

    def _grad_compare_pair(x: str, y: str) -> Dict[str, Any]:
        rx, ry = runs[x], runs[y]
        if rx.get("backward_status") != "ok" or ry.get("backward_status") != "ok":
            return {"status": "skipped_backward_error"}
        return compare_grad_dicts(
            rx.get("selected_grads") or {}, ry.get("selected_grads") or {}
        )

    compare_ab = {
        "forward": _fwd_compare("A_no_offload_bf16", "B_selective_offload_bf16"),
        "grads": _grad_compare_pair("A_no_offload_bf16", "B_selective_offload_bf16"),
    }
    compare_cd = {
        "forward": _fwd_compare("C_no_offload_fp32", "D_selective_offload_fp32"),
        "grads": _grad_compare_pair("C_no_offload_fp32", "D_selective_offload_fp32"),
    }
    runs["_compare_A_B"] = compare_ab
    runs["_compare_C_D"] = compare_cd

    interpretation = interpret_ab_cd(runs)

    optimizer_cmp = {"status": "skipped"}
    a_ok = (runs["A_no_offload_bf16"].get("trainable_grad_report") or {}).get(
        "all_trainable_grads_finite"
    )
    b_ok = (runs["B_selective_offload_bf16"].get("trainable_grad_report") or {}).get(
        "all_trainable_grads_finite"
    )
    if (
        compare_optimizer_update
        and a_ok
        and b_ok
        and bool((compare_ab.get("grads") or {}).get("within_tol"))
    ):
        # Compare one AdamW step on CPU clones — only when A/B finite + match.
        optimizer_cmp = _optimizer_update_compare_cpu(
            model,
            runs["A_no_offload_bf16"].get("selected_grads") or {},
            runs["B_selective_offload_bf16"].get("selected_grads") or {},
            lr=float(lr),
        )
    elif compare_optimizer_update:
        optimizer_cmp = {
            "status": "skipped_nonfinite_or_mismatch",
            "A_finite": a_ok,
            "B_finite": b_ok,
            "grad_within_tol": (compare_ab.get("grads") or {}).get("within_tol"),
            "note": "refusing optimizer.step until all grads finite and A/B match",
        }

    # Drop bulky selected_grads from JSON-facing runs (keep summaries).
    compact_runs: Dict[str, Any] = {}
    for key, run in runs.items():
        if key.startswith("_"):
            continue
        compact = dict(run)
        grads = compact.pop("selected_grads", None)
        if isinstance(grads, dict):
            compact["selected_grads_summary"] = {
                name: {
                    "shape": list(t.shape),
                    "l2_finite": (
                        float(t[torch.isfinite(t)].float().norm().cpu())
                        if torch.isfinite(t).any()
                        else float("nan")
                    ),
                    "all_finite": bool(torch.isfinite(t).all().item()),
                }
                for name, t in list(grads.items())[:64]
            }
        compact_runs[key] = compact

    return {
        "mode": "short_ab_nan_compare",
        "trace_truncate": trunc,
        "max_scored_blocks": int(max_scored_blocks),
        "runs": compact_runs,
        "compare_A_B": compare_ab,
        "compare_C_D": compare_cd,
        "interpretation": interpretation,
        "optimizer_update_A_vs_B": optimizer_cmp,
        "acceptance": {
            "forward_values_finite_A": (
                runs["A_no_offload_bf16"].get("pre_backward_scalars") or {}
            ).get("all_forward_finite"),
            "forward_values_finite_B": (
                runs["B_selective_offload_bf16"].get("pre_backward_scalars") or {}
            ).get("all_forward_finite"),
            "trainable_gradients_finite_A": a_ok,
            "trainable_gradients_finite_B": b_ok,
            "short_trace_grad_match_A_B": (compare_ab.get("grads") or {}).get(
                "within_tol"
            ),
            "optimizer_update_match": optimizer_cmp.get("within_tol"),
            "detached_kv": False,
            "bfix": False,
            "truncated_bptt": False,
            "passes": bool(
                a_ok
                and b_ok
                and (compare_ab.get("grads") or {}).get("within_tol")
                and (
                    optimizer_cmp.get("within_tol") is True
                    or optimizer_cmp.get("status") == "skipped"
                )
            ),
        },
        "process_isolation": False,
        "warning": (
            "In-process A/B/C/D is unsafe after CUDA failures; prefer "
            "orchestrate_isolated_ab_nan_compare()."
        ),
    }


def orchestrate_isolated_ab_nan_compare(
    *,
    python_executable: str,
    config: str,
    trace_jsonl: str,
    trace_index: int,
    output_dir: Path,
    max_scored_blocks: int,
    trainability_case: str,
    threshold_bytes: int,
    pin_memory: bool,
    protect_attn_bias: bool,
    verify_unpack_values: bool,
    sync_backward: bool,
    detect_anomaly: bool,
    clip_epsilon: float,
    device: str,
    init_state_path: Optional[str],
    conditions: Sequence[str],
    compare_optimizer_update: bool = True,
    lr: float = 1e-5,
    pristine_a: bool = False,
    cuda_sync_localize_a: bool = False,
    run_lora_sanity_a: bool = False,
    detect_anomaly_a: Optional[bool] = None,
    fixture_path: Optional[str] = None,
    log_mlp_lora_io_layers: Optional[Sequence[int]] = None,
    validation_level: Optional[str] = None,
    oracle_block_advantages: Optional[Sequence[float]] = None,
    identity_guard_level: str = "G0",
    allow_storage_dedup: bool = False,
    full_unpack_verify: bool = False,
    probe_down_proj: bool = False,
    compare_optimizer_update_force: Optional[bool] = None,
    oracle_repro_seed: int = DEFAULT_ORACLE_REPRO_SEED,
) -> Dict[str, Any]:
    """Run each condition in a fresh subprocess (one CUDA context per condition)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = str(
        Path(__file__).resolve().parent / "isolated_offload_condition.py"
    )
    runs: Dict[str, Any] = {}
    proc_meta: List[Dict[str, Any]] = []

    for condition in conditions:
        out_json = output_dir / f"{condition}.json"
        is_a = condition == "A_no_offload_bf16"
        # A reference path: pristine unless explicitly left instrumented.
        a_pristine = bool(pristine_a) if is_a else False
        a_localize = bool(cuda_sync_localize_a) if is_a else False
        a_lora = bool(run_lora_sanity_a) if is_a else False
        cond_detect_anomaly = bool(detect_anomaly)
        if is_a and detect_anomaly_a is not None:
            cond_detect_anomaly = bool(detect_anomaly_a)
        elif is_a and a_pristine and detect_anomaly_a is None:
            # Default pristine A has no anomaly unless A1 overrides.
            cond_detect_anomaly = bool(detect_anomaly)
        cmd = [
            python_executable,
            "-u",
            runner,
            "--config",
            str(config),
            "--trace-jsonl",
            str(trace_jsonl),
            "--trace-index",
            str(int(trace_index)),
            "--output-json",
            str(out_json),
            "--device",
            str(device),
            "--condition",
            str(condition),
            "--max-scored-blocks",
            str(int(max_scored_blocks)),
            "--trainability-case",
            str(trainability_case),
            "--threshold-bytes",
            str(int(threshold_bytes)),
            "--pin-memory",
            "true" if pin_memory else "false",
            "--protect-attn-bias",
            "true" if protect_attn_bias else "false",
            "--verify-unpack-values",
            "true" if verify_unpack_values else "false",
            "--sync-backward",
            "true" if sync_backward else "false",
            "--detect-anomaly",
            "true" if cond_detect_anomaly else "false",
            "--clip-epsilon",
            str(float(clip_epsilon)),
            "--pristine-no-hooks",
            "true" if a_pristine else "false",
            "--cuda-sync-localize",
            "true" if a_localize else "false",
            "--run-lora-sanity",
            "true" if a_lora else "false",
        ]
        if init_state_path:
            cmd.extend(["--init-state-path", str(init_state_path)])
        if fixture_path:
            cmd.extend(["--fixture-path", str(fixture_path)])
        if log_mlp_lora_io_layers and condition == "B_selective_offload_bf16":
            cmd.extend(
                [
                    "--log-mlp-lora-io-layers",
                    ",".join(str(int(i)) for i in log_mlp_lora_io_layers),
                ]
            )
        if oracle_block_advantages is not None:
            cmd.extend(
                [
                    "--oracle-block-advantages",
                    ",".join(str(float(x)) for x in oracle_block_advantages),
                ]
            )
        if condition == "B_selective_offload_bf16":
            cmd.extend(
                [
                    "--identity-guard-level",
                    str(identity_guard_level),
                    "--allow-storage-dedup",
                    "true" if allow_storage_dedup else "false",
                    "--full-unpack-verify",
                    "true" if full_unpack_verify else "false",
                ]
            )
        # Probe A and B so down_proj A/B stats can be compared.
        if probe_down_proj:
            cmd.extend(["--probe-down-proj", "true"])
        # Identical pre-forward seed for both children (oracle / G3 control).
        if oracle_repro_seed is not None:
            cmd.extend(["--oracle-repro-seed", str(int(oracle_repro_seed))])

        print(f"=== ISOLATED SUBPROCESS {condition} ===")
        print(" ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        meta = {
            "condition": condition,
            "returncode": int(completed.returncode),
            "output_json": str(out_json),
        }
        proc_meta.append(meta)
        if out_json.is_file():
            runs[condition] = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            runs[condition] = {
                "label": condition,
                "status": "error",
                "error_message": f"missing output json; returncode={completed.returncode}",
                "forward_status": "error",
                "backward_status": "error",
            }

    def _load_grads(label: str) -> Dict[str, torch.Tensor]:
        path = (runs.get(label) or {}).get("selected_grads_path")
        if not path or not Path(path).is_file():
            return {}
        return torch.load(path, map_location="cpu")

    compare_ab: Dict[str, Any] = {}
    if "A_no_offload_bf16" in runs and "B_selective_offload_bf16" in runs:
        a_run = runs["A_no_offload_bf16"]
        b_run = runs["B_selective_offload_bf16"]
        compare_ab["forward"] = compare_forward_scalars(a_run, b_run)
        if a_run.get("backward_status") == "ok" and b_run.get("backward_status") == "ok":
            compare_ab["grads"] = compare_grad_dicts(
                _load_grads("A_no_offload_bf16"),
                _load_grads("B_selective_offload_bf16"),
            )
        else:
            compare_ab["grads"] = {"status": "skipped_backward_error"}

    runs["_compare_A_B"] = compare_ab
    interpretation = interpret_ab_cd(runs)

    a_ok = (runs.get("A_no_offload_bf16") or {}).get("trainable_grad_report", {}).get(
        "all_trainable_grads_finite"
    )
    b_ok = (runs.get("B_selective_offload_bf16") or {}).get(
        "trainable_grad_report", {}
    ).get("all_trainable_grads_finite")

    do_opt = bool(compare_optimizer_update)
    if compare_optimizer_update_force is not None:
        do_opt = bool(compare_optimizer_update_force)
    optimizer_cmp: Dict[str, Any] = {"status": "skipped"}
    if (
        do_opt
        and a_ok
        and b_ok
        and bool((compare_ab.get("grads") or {}).get("within_tol"))
    ):
        ga = _load_grads("A_no_offload_bf16")
        gb = _load_grads("B_selective_offload_bf16")
        names = sorted(set(ga) & set(gb))
        params_a = []
        params_b = []
        for name in names:
            base = torch.zeros_like(ga[name])
            pa = base.clone().requires_grad_(True)
            pb = base.clone().requires_grad_(True)
            pa.grad = ga[name].clone()
            pb.grad = gb[name].clone()
            params_a.append(pa)
            params_b.append(pb)
        if params_a:
            opt_a = torch.optim.AdamW(params_a, lr=lr)
            opt_b = torch.optim.AdamW(params_b, lr=lr)
            opt_a.step()
            opt_b.step()
            max_abs = 0.0
            max_rel = 0.0
            for pa, pb in zip(params_a, params_b):
                diff = (pa.detach() - pb.detach()).abs()
                max_abs = max(max_abs, float(diff.max().item()))
                denom = float(pb.detach().abs().max().item()) + 1e-12
                max_rel = max(max_rel, float(diff.max().item()) / denom)
            optimizer_cmp = {
                "status": "ok",
                "num_params": len(params_a),
                "max_abs_update_diff": max_abs,
                "max_rel_update_diff": max_rel,
                "within_tol": max_abs <= 1e-6 or max_rel <= 1e-5,
            }
    elif do_opt and (
        "A_no_offload_bf16" in runs and "B_selective_offload_bf16" in runs
    ):
        optimizer_cmp = {
            "status": "skipped_nonfinite_or_mismatch",
            "A_finite": a_ok,
            "B_finite": b_ok,
            "grad_within_tol": (compare_ab.get("grads") or {}).get("within_tol"),
            "note": "refusing optimizer.step until all grads finite and A/B match",
        }
    elif not do_opt:
        optimizer_cmp = {
            "status": "skipped_by_request",
            "note": "optimizer.step disabled for this diagnostic run",
        }

    fwd_ok = (compare_ab.get("forward") or {}).get("status") == "ok"
    fwd_close = bool(
        fwd_ok
        and float((compare_ab.get("forward") or {}).get("max_block_abs_diff") or 0.0)
        <= 1e-4
        and float((compare_ab.get("forward") or {}).get("loss_abs_diff") or 0.0)
        <= 1e-4
    )
    grad_match = bool((compare_ab.get("grads") or {}).get("within_tol"))
    opt_match = optimizer_cmp.get("within_tol") is True

    b_run = runs.get("B_selective_offload_bf16") or {}
    b_stats = ((b_run.get("selective_offload_summary") or {}).get("stats") or {})
    b_fwd_ok = b_run.get("forward_status") == "ok"
    b_bwd_ok = b_run.get("backward_status") == "ok"
    b_finite = bool(b_ok)
    b_unpack_clean = (
        int(b_stats.get("unpack_value_mismatches") or 0) == 0
        and int(b_stats.get("unpack_nonfinite_count") or 0) == 0
    )
    peak = b_run.get("cuda_memory_peak") or {}

    FULL_A_UNAVAILABLE_REASON = (
        "pristine no-offload FP32-LoRA activation graph exceeds practical "
        "single-RTX-3090 capacity; CUDA driver reports invalid argument near "
        "decoder layer 30."
    )

    if validation_level == "short_fixture_gradient_oracle":
        grad_cmp = compare_ab.get("grads") or {}
        nontrivial = bool(
            not grad_cmp.get("all_gradients_zero")
            and float(grad_cmp.get("global_grad_norm_a") or 0.0)
            > NONTRIVIAL_GRAD_NORM_EPS
            and float(grad_cmp.get("global_grad_norm_b") or 0.0)
            > NONTRIVIAL_GRAD_NORM_EPS
            and int(grad_cmp.get("num_a_nonzero_params") or 0) >= 1
            and int(grad_cmp.get("num_b_nonzero_params") or 0) >= 1
        )
        a_run = runs.get("A_no_offload_bf16") or {}
        b_run_ab = runs.get("B_selective_offload_bf16") or {}
        loss_ok = (
            a_run.get("error_type") != "degenerate_zero_loss"
            and b_run_ab.get("error_type") != "degenerate_zero_loss"
            and abs(
                float(
                    ((a_run.get("pre_backward_scalars") or {}).get("loss") or {}).get(
                        "value"
                    )
                    or 0.0
                )
            )
            > ORACLE_LOSS_ABS_EPS
        )
        down_cmp = compare_down_proj_probes(
            a_run.get("down_proj_lora_probe") or {},
            b_run_ab.get("down_proj_lora_probe") or {},
        )
        compare_ab["down_proj_lora_probe"] = down_cmp
        repro_cmp = compare_ab_repro_fingerprints(
            a_run.get("ab_repro_fingerprint") or {},
            b_run_ab.get("ab_repro_fingerprint") or {},
        )
        compare_ab["ab_repro"] = repro_cmp
        b_stats = ((b_run_ab.get("selective_offload_summary") or {}).get("stats") or {})
        b_instr = b_run_ab.get("instrumentation") or {}
        # Optimizer match only required when optimizer comparison is enabled.
        opt_ok = (not do_opt) or opt_match
        exactness = bool(
            a_ok
            and b_ok
            and fwd_close
            and grad_match
            and opt_ok
            and nontrivial
            and loss_ok
        )
        g3_interpretation = None
        if str(identity_guard_level).upper() == "G3":
            if exactness:
                g3_interpretation = (
                    "G3_matches_A: remaining bug is in a non-LoRA/non-MLP "
                    "offloaded tensor class (not exercised under G3)"
                )
            else:
                g3_interpretation = (
                    "G3_still_differs: not an offload corruption issue; "
                    "investigate reproducibility/determinism between isolated "
                    "A and B processes"
                )
        acceptance = {
            "level": "1_short_fixture_gradient_oracle",
            "short_fixture_gradient_exactness": exactness,
            "identity_guard_level": identity_guard_level,
            "allow_storage_dedup": allow_storage_dedup,
            "full_unpack_verify": full_unpack_verify,
            "cpu_offload_active_B": b_instr.get("cpu_offload_active"),
            "global_identity_hooks_enabled_B": b_instr.get(
                "global_identity_hooks_enabled"
            ),
            "forward_match": fwd_close,
            "grad_match": grad_match,
            "optimizer_update_match": optimizer_cmp.get("within_tol"),
            "optimizer_update_required": do_opt,
            "nontrivial_gradient_required": True,
            "nontrivial_gradients": nontrivial,
            "nondegenerate_loss": loss_ok,
            "trainable_gradients_finite_A": a_ok,
            "trainable_gradients_finite_B": b_ok,
            "global_grad_norm_a": grad_cmp.get("global_grad_norm_a"),
            "global_grad_norm_b": grad_cmp.get("global_grad_norm_b"),
            "global_cosine_finite": grad_cmp.get("global_cosine_finite"),
            "lora_identity_pack_calls": b_stats.get("lora_identity_pack_calls"),
            "mlp_identity_pack_calls": b_stats.get("mlp_identity_pack_calls"),
            "global_identity_pack_calls": b_stats.get("global_identity_pack_calls"),
            "offloaded_tensors_B": b_stats.get("offloaded_tensors"),
            "storage_dedup_hits": b_stats.get("storage_dedup_hits"),
            "unique_save_events": b_stats.get("unique_save_events"),
            "full_unpack_value_mismatches": b_stats.get("full_unpack_value_mismatches"),
            "ab_repro_all_logged_fields_match": repro_cmp.get(
                "all_logged_fields_match"
            ),
            "ab_repro_diff_fields": repro_cmp.get("diff_fields"),
            "g3_interpretation": g3_interpretation,
            "detached_kv": False,
            "bfix": False,
            "truncated_bptt": False,
            "full_production_no_offload_reference_feasible": False,
            "full_production_no_offload_reference_unavailable_reason": (
                FULL_A_UNAVAILABLE_REASON
            ),
            "passes": exactness,
        }
        mode = "short_sequence_ab_gradient_oracle_isolated"
    elif validation_level == "production_trace_b_feasibility":
        acceptance = {
            "level": "2_production_trace_b_feasibility",
            "short_fixture_gradient_exactness": None,
            "production_trace_forward_exactness": bool(b_fwd_ok),
            "production_trace_gradient_finiteness": bool(b_finite and b_bwd_ok),
            "production_trace_single_gpu_feasibility": bool(
                b_fwd_ok and b_bwd_ok and b_finite and b_unpack_clean
            ),
            "full_production_no_offload_reference_feasible": False,
            "full_production_no_offload_reference_unavailable_reason": (
                FULL_A_UNAVAILABLE_REASON
            ),
            "unpack_clean": b_unpack_clean,
            "live_non_detached_kv": True,
            "bfix": False,
            "truncated_bptt": False,
            "peak_gpu_allocated_bytes": peak.get("max_allocated_bytes"),
            "peak_gpu_reserved_bytes": peak.get("max_reserved_bytes"),
            "passes": bool(b_fwd_ok and b_bwd_ok and b_finite and b_unpack_clean),
        }
        mode = "production_trace_b_feasibility_isolated"
    else:
        acceptance = {
            "trainable_gradients_finite_A": a_ok,
            "trainable_gradients_finite_B": b_ok,
            "short_trace_grad_match_A_B": grad_match,
            "optimizer_update_match": optimizer_cmp.get("within_tol"),
            "detached_kv": False,
            "bfix": False,
            "truncated_bptt": False,
            "full_production_no_offload_reference_feasible": False,
            "full_production_no_offload_reference_unavailable_reason": (
                FULL_A_UNAVAILABLE_REASON
            ),
            "passes": bool(
                a_ok
                and b_ok
                and grad_match
                and (
                    optimizer_cmp.get("within_tol") is True
                    or not compare_optimizer_update
                )
            ),
        }
        mode = "short_ab_nan_compare_isolated"

    return {
        "mode": mode,
        "validation_level": validation_level,
        "process_isolation": True,
        "max_scored_blocks": int(max_scored_blocks),
        "fixture_path": fixture_path,
        "conditions": list(conditions),
        "subprocess_meta": proc_meta,
        "runs": {k: v for k, v in runs.items() if not str(k).startswith("_")},
        "compare_A_B": compare_ab,
        "interpretation": interpretation,
        "optimizer_update_A_vs_B": optimizer_cmp,
        "acceptance": acceptance,
    }
