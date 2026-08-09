"""NaN / Inf gradient diagnostics for selective-offload investigation.

Diagnostic-only helpers. Never used to alter GRPO objective. Hard-fails before
optimizer.step when any trainable gradient is non-finite.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _as_float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return out
    return out


def aggregate_isfinite_flags(*flags: Any) -> bool:
    """AND-reduce finite flags.

    Accepts:
      - a single bool
      - an iterable of bools
      - a mix of bools and iterables of bools
    """
    collected: List[bool] = []
    for flag in flags:
        if isinstance(flag, bool):
            collected.append(flag)
            continue
        if flag is None:
            collected.append(False)
            continue
        # Iterable of per-scalar bools (e.g. block reports).
        try:
            iterator = iter(flag)
        except TypeError as exc:
            raise TypeError(
                f"isfinite flag must be bool or iterable of bools, got {type(flag)!r}"
            ) from exc
        for item in iterator:
            if isinstance(item, bool):
                collected.append(item)
            elif isinstance(item, dict) and "isfinite" in item:
                collected.append(bool(item.get("isfinite")))
            else:
                collected.append(bool(item))
    if not collected:
        return True
    return bool(all(collected))


def scalar_tensor_report(tensor: torch.Tensor, *, name: str = "tensor") -> Dict[str, Any]:
    """Report value/dtype/isfinite for a scalar or small tensor without large temps."""
    if not torch.is_tensor(tensor):
        return {"name": name, "type": type(tensor).__name__}
    report: Dict[str, Any] = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        report.update({"value": None, "isfinite": True, "has_nan": False, "has_inf": False})
        return report
    if tensor.numel() == 1:
        value = float(tensor.detach().float().cpu().item())
        report["value"] = value
        report["isfinite"] = bool(math.isfinite(value))
        report["has_nan"] = bool(math.isnan(value))
        report["has_inf"] = bool(math.isinf(value))
        return report
    # Bounded sample for larger tensors (no full-tensor isfinite allocation).
    from rl.selective_saved_tensor_offload import sample_tensor_values

    samples, lin_idxs, finite_flags = sample_tensor_values(tensor, max_samples=32)
    report["sample_lin_indices"] = list(lin_idxs)
    report["sample_values"] = list(samples)
    report["sample_isfinite"] = list(finite_flags)
    report["isfinite"] = all(finite_flags) if finite_flags else True
    report["has_nan"] = any(math.isnan(v) for v in samples)
    report["has_inf"] = any(math.isinf(v) for v in samples)
    return report


def analyze_grad_tensor(grad: Optional[torch.Tensor]) -> Dict[str, Any]:
    if grad is None:
        return {
            "grad_is_none": True,
            "numel": 0,
            "finite_count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "max_abs_finite": None,
            "l2_finite": None,
            "all_finite": False,
        }
    # Work on a detached float view for counting; keep peak memory modest by
    # chunking along the flat dimension via narrow on a view when possible.
    flat = grad.detach()
    numel = int(flat.numel())
    if numel == 0:
        return {
            "grad_is_none": False,
            "numel": 0,
            "finite_count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "max_abs_finite": 0.0,
            "l2_finite": 0.0,
            "all_finite": True,
            "dtype": str(grad.dtype).replace("torch.", ""),
        }

    # Prefer a single-pass on float32 for LoRA-sized tensors (typically small).
    # For very large grads, fall back to sampled + chunked counts.
    if numel <= 4_000_000:
        g = flat.float().reshape(-1)
        nan_mask = torch.isnan(g)
        inf_mask = torch.isinf(g)
        finite_mask = torch.isfinite(g)
        nan_count = int(nan_mask.sum().item())
        inf_count = int(inf_mask.sum().item())
        finite_count = int(finite_mask.sum().item())
        if finite_count:
            finite_vals = g[finite_mask]
            max_abs = float(finite_vals.abs().max().item())
            l2 = float(finite_vals.norm().item())
        else:
            max_abs = None
            l2 = None
        return {
            "grad_is_none": False,
            "numel": numel,
            "finite_count": finite_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "max_abs_finite": max_abs,
            "l2_finite": l2,
            "all_finite": nan_count == 0 and inf_count == 0,
            "dtype": str(grad.dtype).replace("torch.", ""),
            "shape": list(grad.shape),
        }

    # Large-grad path: chunked counting without holding full masks.
    g = flat.reshape(-1)
    chunk = 1 << 20
    nan_count = 0
    inf_count = 0
    finite_count = 0
    max_abs = None
    l2_sq = 0.0
    for start in range(0, numel, chunk):
        piece = g[start : start + chunk].float()
        nan_count += int(torch.isnan(piece).sum().item())
        inf_count += int(torch.isinf(piece).sum().item())
        finite = torch.isfinite(piece)
        finite_count += int(finite.sum().item())
        if finite.any():
            vals = piece[finite]
            local_max = float(vals.abs().max().item())
            max_abs = local_max if max_abs is None else max(max_abs, local_max)
            l2_sq += float((vals * vals).sum().item())
    return {
        "grad_is_none": False,
        "numel": numel,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "max_abs_finite": max_abs,
        "l2_finite": math.sqrt(l2_sq) if finite_count else None,
        "all_finite": nan_count == 0 and inf_count == 0,
        "dtype": str(grad.dtype).replace("torch.", ""),
        "shape": list(grad.shape),
        "chunked": True,
    }


def report_trainable_grads(model: nn.Module) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    any_nonfinite = False
    any_none = False
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        stats = analyze_grad_tensor(parameter.grad)
        row = {"name": name, **stats}
        rows.append(row)
        if stats["grad_is_none"]:
            any_none = True
        elif not stats["all_finite"]:
            any_nonfinite = True
    return {
        "parameters": rows,
        "num_trainable": len(rows),
        "any_grad_none": any_none,
        "any_grad_nonfinite": any_nonfinite,
        "all_trainable_grads_finite": (not any_nonfinite) and (not any_none),
        "num_nonfinite": sum(1 for r in rows if not r["grad_is_none"] and not r["all_finite"]),
        "num_grad_none": sum(1 for r in rows if r["grad_is_none"]),
    }


def assert_trainable_grads_finite(model: nn.Module) -> Dict[str, Any]:
    """Hard-fail before optimizer.step if any trainable gradient is non-finite."""
    report = report_trainable_grads(model)
    if not report["all_trainable_grads_finite"]:
        offenders = [
            {
                "name": r["name"],
                "nan_count": r["nan_count"],
                "inf_count": r["inf_count"],
                "grad_is_none": r["grad_is_none"],
            }
            for r in report["parameters"]
            if r["grad_is_none"] or not r["all_finite"]
        ][:32]
        raise RuntimeError(
            "non-finite or missing trainable gradients; refusing optimizer.step. "
            f"offenders={offenders}"
        )
    return report


class FirstNonfiniteGradProbe:
    """Register hooks to localize the first non-finite grad during backward."""

    def __init__(self, model: nn.Module, *, sync_cuda: bool = True) -> None:
        self.sync_cuda = bool(sync_cuda)
        self.first_parameter: Optional[Dict[str, Any]] = None
        self.first_module: Optional[Dict[str, Any]] = None
        self.handles: List[Any] = []
        self._register(model)

    def _maybe_sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _register(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            def _param_hook(grad, *, _name=name):
                if self.first_parameter is not None:
                    return grad
                self._maybe_sync()
                stats = analyze_grad_tensor(grad)
                if grad is not None and not stats["all_finite"]:
                    self.first_parameter = {
                        "parameter": _name,
                        "grad_stats": stats,
                    }
                return grad

            self.handles.append(parameter.register_hook(_param_hook))

        for name, module in model.named_modules():
            if name == "":
                continue

            def _module_hook(mod, grad_input, grad_output, *, _name=name):
                if self.first_module is not None:
                    return
                self._maybe_sync()
                for idx, g in enumerate(grad_output or ()):
                    if g is None or not torch.is_tensor(g):
                        continue
                    stats = analyze_grad_tensor(g)
                    if not stats["all_finite"]:
                        self.first_module = {
                            "module": _name,
                            "grad_output_index": idx,
                            "grad_stats": stats,
                            "module_class": type(mod).__name__,
                        }
                        break

            try:
                self.handles.append(module.register_full_backward_hook(_module_hook))
            except Exception:
                continue

    def close(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()

    def report(self) -> Dict[str, Any]:
        return {
            "first_nonfinite_parameter": self.first_parameter,
            "first_nonfinite_module": self.first_module,
            "sync_cuda": self.sync_cuda,
        }


def _cosine_similarity_clamped(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[float, float]:
    """Return (clamped_cosine, raw_cosine). Reported cosine is always in [-1, 1]."""
    raw = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    if raw != raw:  # NaN
        return raw, raw
    clamped = max(-1.0, min(1.0, raw))
    return clamped, raw


def compare_grad_dicts(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Elementwise LoRA/projector grad compare with zero-vector cosine semantics."""
    names = sorted(set(a) | set(b))
    missing = [n for n in names if n not in a or n not in b]
    max_abs = 0.0
    max_rel_l2 = 0.0
    min_cosine: Optional[float] = None
    min_cosine_raw: Optional[float] = None
    nan_mask_mismatches = 0
    zero_nonzero_mask_mismatches = 0
    num_both_zero_tensors = 0
    num_both_nonzero_tensors = 0
    num_one_sided_zero_tensors = 0
    num_a_nonzero_params = 0
    num_b_nonzero_params = 0
    num_a_nonzero_elems = 0
    num_b_nonzero_elems = 0
    compared = 0
    first_bad = None
    flat_a: List[torch.Tensor] = []
    flat_b: List[torch.Tensor] = []
    for name in names:
        if name not in a or name not in b:
            continue
        ga = a[name].float().reshape(-1)
        gb = b[name].float().reshape(-1)
        a_nan = torch.isnan(ga)
        b_nan = torch.isnan(gb)
        a_inf = torch.isinf(ga)
        b_inf = torch.isinf(gb)
        if bool((a_nan != b_nan).any().item()) or bool((a_inf != b_inf).any().item()):
            nan_mask_mismatches += 1
            if first_bad is None:
                first_bad = {"name": name, "reason": "nan_inf_mask_mismatch"}
        a_zero_elem = ga == 0
        b_zero_elem = gb == 0
        if bool((a_zero_elem != b_zero_elem).any().item()):
            zero_nonzero_mask_mismatches += 1
            if first_bad is None:
                first_bad = {"name": name, "reason": "zero_nonzero_mask_mismatch"}

        a_nz = int((~a_zero_elem & torch.isfinite(ga)).sum().item())
        b_nz = int((~b_zero_elem & torch.isfinite(gb)).sum().item())
        num_a_nonzero_elems += a_nz
        num_b_nonzero_elems += b_nz
        if a_nz > 0:
            num_a_nonzero_params += 1
        if b_nz > 0:
            num_b_nonzero_params += 1

        a_all_zero = bool(torch.isfinite(ga).all().item()) and a_nz == 0
        b_all_zero = bool(torch.isfinite(gb).all().item()) and b_nz == 0
        finite = torch.isfinite(ga) & torch.isfinite(gb)
        if finite.any():
            flat_a.append(ga[finite])
            flat_b.append(gb[finite])
        if a_all_zero and b_all_zero:
            num_both_zero_tensors += 1
            # Identically zero: exact match; do not evaluate cosine.
        elif a_all_zero != b_all_zero:
            num_one_sided_zero_tensors += 1
            if first_bad is None:
                first_bad = {"name": name, "reason": "one_sided_zero_tensor"}
        else:
            num_both_nonzero_tensors += 1
            if finite.any():
                da = ga[finite]
                db = gb[finite]
                diff = (da - db).abs()
                local_max = float(diff.max().item())
                max_abs = max(max_abs, local_max)
                denom = float(db.norm().item()) + 1e-12
                rel = float((da - db).norm().item()) / denom
                max_rel_l2 = max(max_rel_l2, rel)
                cosine, cosine_raw = _cosine_similarity_clamped(da, db)
                min_cosine = (
                    cosine if min_cosine is None else min(min_cosine, cosine)
                )
                min_cosine_raw = (
                    cosine_raw
                    if min_cosine_raw is None
                    else min(min_cosine_raw, cosine_raw)
                )
                if first_bad is None and (local_max > 1e-3 and rel > 1e-3):
                    first_bad = {
                        "name": name,
                        "reason": "value_mismatch",
                        "max_abs": local_max,
                        "rel_l2": rel,
                    }
        compared += 1

    both_finite = all(
        bool(torch.isfinite(a[n]).all().item()) and bool(torch.isfinite(b[n]).all().item())
        for n in names
        if n in a and n in b
    )
    global_rel_l2 = None
    global_cosine = None
    global_cosine_raw = None
    global_cosine_status = None
    global_norm_a = 0.0
    global_norm_b = 0.0
    if flat_a:
        ca = torch.cat(flat_a)
        cb = torch.cat(flat_b)
        global_norm_a = float(ca.norm().item())
        global_norm_b = float(cb.norm().item())
        if global_norm_a == 0.0 and global_norm_b == 0.0:
            global_cosine = None
            global_cosine_raw = None
            global_cosine_status = "both_zero"
            global_rel_l2 = 0.0
        elif global_norm_a == 0.0 or global_norm_b == 0.0:
            global_cosine = None
            global_cosine_raw = None
            global_cosine_status = "one_sided_zero"
            global_rel_l2 = float("inf")
            if first_bad is None:
                first_bad = {"name": "__global__", "reason": "one_sided_zero_global"}
        else:
            global_rel_l2 = float((ca - cb).norm().item()) / (global_norm_b + 1e-12)
            global_cosine, global_cosine_raw = _cosine_similarity_clamped(ca, cb)
            global_cosine_status = "both_nonzero"

    cosine_ok = (
        num_both_nonzero_tensors == 0
        or (min_cosine is not None and min_cosine >= 0.999)
    )
    exact_zero_match = (
        compared > 0
        and num_both_zero_tensors == compared
        and num_one_sided_zero_tensors == 0
        and max_abs == 0.0
    )
    values_match = (
        compared > 0
        and not missing
        and both_finite
        and nan_mask_mismatches == 0
        and zero_nonzero_mask_mismatches == 0
        and num_one_sided_zero_tensors == 0
        and (max_abs <= 1e-4 or max_rel_l2 <= 1e-3 or exact_zero_match)
        and cosine_ok
        and global_cosine_status in (None, "both_zero", "both_nonzero")
        and (
            global_cosine_status != "both_nonzero"
            or (
                global_cosine is not None
                and global_cosine >= 0.999
                and (global_rel_l2 is not None and global_rel_l2 <= 1e-3)
            )
        )
    )
    return {
        "num_compared": compared,
        "missing_names": missing,
        "max_abs_error_finite": max_abs,
        "max_rel_l2_error_finite": max_rel_l2,
        "min_cosine_finite": min_cosine,
        "min_cosine_raw": min_cosine_raw,
        "min_cosine_status": (
            "both_zero"
            if num_both_nonzero_tensors == 0 and num_one_sided_zero_tensors == 0
            else ("both_nonzero" if num_both_nonzero_tensors > 0 else "one_sided_zero")
        ),
        "global_rel_l2_error_finite": global_rel_l2,
        "global_cosine_finite": global_cosine,
        "global_cosine_raw": global_cosine_raw,
        "global_cosine_status": global_cosine_status,
        "global_grad_norm_a": global_norm_a,
        "global_grad_norm_b": global_norm_b,
        "num_both_zero_tensors": num_both_zero_tensors,
        "num_both_nonzero_tensors": num_both_nonzero_tensors,
        "num_one_sided_zero_tensors": num_one_sided_zero_tensors,
        "num_a_nonzero_params": num_a_nonzero_params,
        "num_b_nonzero_params": num_b_nonzero_params,
        "num_a_nonzero_elems": num_a_nonzero_elems,
        "num_b_nonzero_elems": num_b_nonzero_elems,
        "nan_inf_mask_mismatches": nan_mask_mismatches,
        "zero_nonzero_mask_mismatches": zero_nonzero_mask_mismatches,
        "both_sides_all_finite": both_finite,
        "all_gradients_zero": bool(
            global_norm_a == 0.0 and global_norm_b == 0.0 and compared > 0
        ),
        "exact_zero_match": exact_zero_match,
        "first_bad": first_bad,
        "within_tol": bool(values_match),
    }


def compare_forward_scalars(
    a_run: Dict[str, Any], b_run: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare per-block logps, total logp, and scalar GRPO loss."""
    if a_run.get("forward_status") != "ok" or b_run.get("forward_status") != "ok":
        return {"status": "skipped_forward_error"}
    ab = list(a_run.get("block_log_probs") or [])
    bb = list(b_run.get("block_log_probs") or [])
    n = min(len(ab), len(bb))
    diffs = [abs(float(ab[i]) - float(bb[i])) for i in range(n)]
    a_loss = (a_run.get("pre_backward_scalars") or {}).get("loss", {}) or {}
    b_loss = (b_run.get("pre_backward_scalars") or {}).get("loss", {}) or {}
    total_logp_abs = abs(
        float(a_run.get("current_log_prob", 0.0))
        - float(b_run.get("current_log_prob", 0.0))
    )
    loss_abs = abs(
        float(a_loss.get("value") or 0.0) - float(b_loss.get("value") or 0.0)
    )
    max_block = max(diffs) if diffs else 0.0
    within_tol = bool(
        max_block <= 1e-5 and total_logp_abs <= 1e-5 and loss_abs <= 1e-5
    )
    return {
        "status": "ok",
        "num_blocks_compared": n,
        "per_block_abs_diff": diffs,
        "max_block_abs_diff": max_block,
        "total_logp_abs_diff": total_logp_abs,
        "loss_abs_diff": loss_abs,
        "a_block_log_probs": ab,
        "b_block_log_probs": bb,
        "a_total_logp": float(a_run.get("current_log_prob", 0.0)),
        "b_total_logp": float(b_run.get("current_log_prob", 0.0)),
        "a_loss": a_loss.get("value"),
        "b_loss": b_loss.get("value"),
        "within_tol": within_tol,
    }
