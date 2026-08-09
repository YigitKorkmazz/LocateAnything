#!/usr/bin/env python3
"""CPU forensic oracle for two-GPU G=4 live-cache smoke checkpoints/metrics.

It never constructs a model or touches CUDA.  It compares serialized training
state and the completed-attempt JSONL records produced by the smoke runner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from rl.inprocess_short_fixture_oracle import normalize_ab_by_a12_baseline
from rl.nan_grad_diagnostics import compare_grad_dicts
from two_gpu_exactness_oracle import _write_report_safely


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-json", required=True)
    p.add_argument("--checkpoint-load-oracle", default=None)
    p.add_argument("--resaved-checkpoint", default=None)
    p.add_argument("--uninterrupted-a-checkpoint", default=None)
    p.add_argument("--uninterrupted-a-metrics", default=None)
    p.add_argument("--uninterrupted-b-checkpoint", default=None)
    p.add_argument("--uninterrupted-b-metrics", default=None)
    p.add_argument("--resumed-checkpoint", default=None)
    p.add_argument("--resumed-metrics", default=None)
    p.add_argument("--resumed-summary", default=None)
    p.add_argument("--step5-checkpoint", default=None)
    return p.parse_args()


def _load(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _hash(value: Any) -> str:
    h = hashlib.sha256()
    modes: set[str] = set()

    def tensor_bytes(tensor: torch.Tensor) -> bytes:
        cpu = tensor.detach().to("cpu").contiguous()
        try:
            # uint8 is universally NumPy-compatible and preserves BF16 bits.
            modes.add("raw_storage_uint8")
            return cpu.view(torch.uint8).numpy().tobytes()
        except (RuntimeError, TypeError):
            # Diagnostic fallback only; exact acceptance always uses torch.equal.
            modes.add("float32_diagnostic_fallback")
            return cpu.float().contiguous().numpy().tobytes()

    def visit(x: Any) -> None:
        if isinstance(x, torch.Tensor):
            h.update(b"tensor"); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
            h.update(tensor_bytes(x)); return
        if np is not None and isinstance(x, np.ndarray):
            h.update(b"ndarray"); h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
            h.update(x.tobytes()); return
        if np is not None and isinstance(x, np.generic):
            h.update(b"numpy_scalar"); visit(x.item()); return
        if isinstance(x, dict):
            h.update(b"dict")
            for key in sorted(x, key=str): h.update(str(key).encode()); visit(x[key])
            return
        if isinstance(x, (list, tuple)):
            h.update(b"seq")
            for item in x: visit(item)
            return
        h.update(repr(x).encode("utf-8"))
    visit(value); return h.hexdigest()


def _summary(value: Any) -> Dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {"python_type": "torch.Tensor", "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if np is not None and isinstance(value, np.ndarray):
        return {"python_type": "numpy.ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    return {"python_type": type(value).__name__, "repr": repr(value)[:240]}


def _exact_comparison(a: Any, b: Any, path: str = "root") -> Dict[str, Any]:
    """Exact, ndarray-safe recursive comparison with audit-friendly paths."""
    mismatches: list[Dict[str, Any]] = []
    compared = 0

    def mismatch(where: str, left: Any, right: Any, reason: str) -> None:
        mismatches.append({"path": where, "reason": reason, "left": _summary(left), "right": _summary(right)})

    def visit(left: Any, right: Any, where: str) -> bool:
        nonlocal compared
        compared += 1
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                mismatch(where, left, right, "tensor_type_mismatch"); return False
            same_meta = left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
            same_value = same_meta and bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
            if not same_value: mismatch(where, left, right, "tensor_metadata_or_value_mismatch")
            return same_value
        if np is not None and (isinstance(left, np.ndarray) or isinstance(right, np.ndarray)):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                mismatch(where, left, right, "ndarray_type_mismatch"); return False
            if left.shape != right.shape or left.dtype != right.dtype:
                mismatch(where, left, right, "ndarray_metadata_mismatch"); return False
            try: equal = bool(np.array_equal(left, right, equal_nan=True))
            except TypeError: equal = bool(np.array_equal(left, right))
            if not equal: mismatch(where, left, right, "ndarray_value_mismatch")
            return equal
        if np is not None and (isinstance(left, np.generic) or isinstance(right, np.generic)):
            if not isinstance(left, np.generic) or not isinstance(right, np.generic):
                mismatch(where, left, right, "numpy_scalar_type_mismatch"); return False
            return visit(left.item(), right.item(), where + ".item")
        if isinstance(left, (torch.dtype, torch.device, Path)) or isinstance(right, (torch.dtype, torch.device, Path)):
            equal = type(left) is type(right) and str(left) == str(right)
            if not equal: mismatch(where, left, right, "dtype_device_path_mismatch")
            return equal
        if isinstance(left, dict) or isinstance(right, dict):
            if not isinstance(left, dict) or not isinstance(right, dict):
                mismatch(where, left, right, "dict_type_mismatch"); return False
            if set(left) != set(right):
                mismatch(where + ".keys", sorted(map(str, left)), sorted(map(str, right)), "dict_key_set_mismatch")
                return False
            outcomes = [visit(left[key], right[key], f"{where}.{key}") for key in sorted(left, key=str)]
            return all(outcomes)
        if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
            if type(left) is not type(right) or not isinstance(left, (tuple, list)):
                mismatch(where, left, right, "sequence_type_mismatch"); return False
            if len(left) != len(right):
                mismatch(where + ".length", len(left), len(right), "sequence_length_mismatch"); return False
            outcomes = [visit(x, y, f"{where}[{index}]") for index, (x, y) in enumerate(zip(left, right))]
            return all(outcomes)
        if isinstance(left, set) or isinstance(right, set):
            if not isinstance(left, set) or not isinstance(right, set):
                mismatch(where, left, right, "set_type_mismatch"); return False
            equal = sorted(map(repr, left)) == sorted(map(repr, right))
            if not equal: mismatch(where, left, right, "set_member_mismatch")
            return equal
        if isinstance(left, float) or isinstance(right, float):
            equal = isinstance(left, float) and isinstance(right, float) and ((math.isnan(left) and math.isnan(right)) or left == right)
            if not equal: mismatch(where, left, right, "float_exact_or_nan_mismatch")
            return bool(equal)
        equal = type(left) is type(right) and bool(left == right)
        if not equal: mismatch(where, left, right, "scalar_type_or_value_mismatch")
        return bool(equal)

    equal = visit(a, b, path)
    return {"equal": bool(equal), "compared_field_count": compared, "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None, "mismatches": mismatches[:64]}


def _exact(a: Any, b: Any) -> bool:
    return bool(_exact_comparison(a, b)["equal"])


def _optimizer_bank(value: Any, prefix: str = "optimizer") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor): out[prefix] = value.detach().float().cpu()
    elif isinstance(value, dict):
        for k, v in value.items(): out.update(_optimizer_bank(v, f"{prefix}.{k}"))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value): out.update(_optimizer_bank(v, f"{prefix}.{i}"))
    return out


def _optimizer_parameter_names(payload: Mapping[str, Any]) -> Dict[int, str]:
    """Reconstruct AdamW state-id names using the same Case-B grouping order."""
    names = list(payload["trainable_state"])
    groups = [[n for n in names if "lora_" in n], [n for n in names if "lora_" not in n]]
    mapping: Dict[int, str] = {}
    for state_group, group_names in zip(payload["optimizer_state"]["param_groups"], groups):
        mapping.update({int(pid): name for pid, name in zip(state_group["params"], group_names)})
    return mapping


def _state_metrics(a: torch.Tensor, b: torch.Tensor, reference: torch.Tensor, baseline_abs: torch.Tensor | None = None) -> Dict[str, Any]:
    x, y, ref = a.detach().float().cpu().reshape(-1), b.detach().float().cpu().reshape(-1), reference.detach().float().cpu().reshape(-1)
    diff = (x - y).abs(); ref_max = float(ref.abs().max().item()) if ref.numel() else 0.0
    max_abs, max_index = (float(diff.max().item()), int(diff.argmax().item())) if diff.numel() else (0.0, 0)
    metrics = {"shape": list(a.shape), "dtype": str(a.dtype), "a_norm": float(x.norm().item()), "b_norm": float(y.norm().item()),
        "reference_norm": float(ref.norm().item()), "a_abs_max": float(x.abs().max().item()) if x.numel() else 0.0,
        "b_abs_max": float(y.abs().max().item()) if y.numel() else 0.0, "reference_abs_max": ref_max,
        "rel_l2": float((x-y).norm().item() / (y.norm().item() + 1e-12)), "max_abs": max_abs,
        "normalized_max_error": float(max_abs / max(ref_max, 1e-12)), "p99_abs": float(torch.quantile(diff, .99).item()) if diff.numel() else 0.0,
        "p999_abs": float(torch.quantile(diff, .999).item()) if diff.numel() else 0.0, "max_abs_index": max_index,
        "a_value_at_max": float(x[max_index].item()) if x.numel() else None, "b_value_at_max": float(y[max_index].item()) if y.numel() else None,
        "reference_value_at_max": float(ref[max_index].item()) if ref.numel() else None,
        "reference_near_zero": ref_max <= 1e-12}
    if baseline_abs is not None:
        base = baseline_abs.detach().float().cpu().reshape(-1).abs()
        metrics["elements_exceeding_matched_ab_baseline"] = int((diff > base).sum().item())
        metrics["fraction_exceeding_matched_ab_baseline"] = float((diff > base).float().mean().item()) if diff.numel() else 0.0
    return metrics


def _optimizer_scale_aware(a: Mapping[str, Any], b: Mapping[str, Any], resumed: Mapping[str, Any]) -> Dict[str, Any]:
    names = _optimizer_parameter_names(a); details, worst, failures = {}, None, []
    for pid, a_state in a["optimizer_state"]["state"].items():
        for key, av in a_state.items():
            if not isinstance(av, torch.Tensor): continue
            bv, rv = b["optimizer_state"]["state"][pid][key], resumed["optimizer_state"]["state"][pid][key]
            field = f"optimizer.state.{pid}.{key}"; label = {"parameter_name": names.get(int(pid), f"<unknown:{pid}>"), "adamw_key": key}
            if key == "step":
                exact = bool(torch.equal(av.cpu(), bv.cpu()) and torch.equal(av.cpu(), rv.cpu()))
                details[field] = {**label, "exact": exact, "a": float(av.item()), "b": float(bv.item()), "resumed": float(rv.item())}
                if not exact: failures.append(field)
                continue
            ab_diff = av.float() - bv.float(); ar = _state_metrics(av, rv, bv, ab_diff)
            ab = _state_metrics(av, bv, bv)
            ar.update(label); ar["ab_baseline"] = ab
            ar.update({"A_norm": ar["a_norm"], "B_norm": ar["reference_norm"], "resumed_norm": ar["b_norm"],
                       "A_value_at_max": ar["a_value_at_max"], "B_value_at_max": ar["reference_value_at_max"],
                       "resumed_value_at_max": ar["b_value_at_max"]})
            raw_exceeds = ar["max_abs"] > ab["max_abs"] * 1.05 + 1e-12
            scale_exceeds = ar["normalized_max_error"] > ab["normalized_max_error"] * 1.05 + 1e-12
            ar["raw_and_scale_exceed_matched_baseline"] = bool(raw_exceeds and scale_exceeds)
            details[field] = ar
            if ar["raw_and_scale_exceed_matched_baseline"]: failures.append(field)
            if worst is None or ar["max_abs"] > worst["max_abs"]: worst = {"field": field, **ar}
    return {"per_state": details, "worst_raw_max_abs_state": worst, "failing_scale_aware_states": failures,
            "step_fields_exact": not any(field.endswith(".step") for field in failures),
            "no_state_exceeds_matched_ab_raw_and_scale_baseline": not failures}


def _per_tensor(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    result, first = {}, None
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            item = {"status": "missing"}; result[name] = item
            if first is None: first = {"name": name, **item}
            continue
        x, y = a[name].float().reshape(-1), b[name].float().reshape(-1)
        diff = x - y; max_abs = float(diff.abs().max().item()) if x.numel() else 0.0
        rel = float(diff.norm().item() / (y.norm().item() + 1e-12))
        if x.norm().item() == 0.0 or y.norm().item() == 0.0: cosine = None
        else: cosine = float(torch.nn.functional.cosine_similarity(x, y, dim=0).item())
        item = {"rel_l2": rel, "max_abs": max_abs, "cosine": cosine, "exact": bool(torch.equal(x, y))}
        result[name] = item
        if first is None and not item["exact"]: first = {"name": name, **item}
    return {"per_tensor": result, "first_divergent": first}


def _checkpoint_hashes(payload: Mapping[str, Any]) -> Dict[str, str]:
    counters = {k: payload.get(k) for k in ("global_step", "optimizer_step_count", "attempted_group_count", "skipped_zero_advantage_group_count", "sample_cursor", "seed", "config_path")}
    return {"tensor_checksum_mode": "raw_storage_uint8; float32_diagnostic_fallback if uint8 view is unsupported",
            "trainable_state_checksum": _hash(payload["trainable_state"]), "optimizer_state_checksum": _hash(payload["optimizer_state"]),
            "rng_state_checksum": _hash(payload["rng_state"]), "counters_seed_schedule_checksum": _hash(counters)}


def _records(path: str | Path) -> Dict[int, Dict[str, Any]]:
    out = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("step_status") == "optimizer_step_completed" and int(row.get("global_step", -1)) in range(6, 11):
            out[int(row["global_step"])] = row
    return out


def _all_update_records(path: str | Path) -> Dict[int, Dict[str, Any]]:
    return {int(row["global_step"]): row for row in (json.loads(line) for line in Path(path).read_text().splitlines())
            if row.get("step_status") == "optimizer_step_completed"}


def _per_update_state_observability(a_metrics: str, b_metrics: str, resumed_metrics: str) -> Dict[str, Any]:
    a, b, r = _all_update_records(a_metrics), _all_update_records(b_metrics), _all_update_records(resumed_metrics)
    out, first_ab, first_ar = {}, None, None
    for step in sorted(set(a) & set(b)):
        ab = a[step].get("parameter_update", {}).get("lora_after_checksum") == b[step].get("parameter_update", {}).get("lora_after_checksum")
        ar = None if step not in r else a[step].get("parameter_update", {}).get("lora_after_checksum") == r[step].get("parameter_update", {}).get("lora_after_checksum")
        out[str(step)] = {"lora_checksum_A_vs_B_exact": bool(ab), "lora_checksum_A_vs_resumed_exact": ar,
                          "optimizer_state_snapshot": "unavailable_per_update: checkpoints exist only at steps 5 and 10"}
        if first_ab is None and not ab: first_ab = step
        if first_ar is None and ar is False: first_ar = step
    return {"per_update": out, "first_lora_checksum_divergence_A_vs_B": first_ab,
            "first_lora_checksum_divergence_A_vs_resumed": first_ar,
            "adamw_first_divergence": "unavailable between checkpoint steps; exact step-5/10 states are compared separately"}


def _identity_projection(row: Mapping[str, Any]) -> Dict[str, Any]:
    trajectories = []
    for t in row.get("trajectories", []):
        trajectories.append({k: t.get(k) for k in ("rollout_seed", "reward", "committed_branch", "committed_final_bbox_norm_1000", "generated_token_ids_checksum", "old_logp", "current_logp", "loss_grpo_unscaled", "loss_grpo_scaled")})
    return {"attempted_group_count": row.get("attempted_group_count"), "sample_index": row.get("sample_index"),
            "sample_cursor_before_group": row.get("sample_cursor_before_group"), "reward_group": row.get("reward_group"), "trajectories": trajectories}


def _control_projection(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = {key: row.get(key) for key in ("global_step", "optimizer_step_count_before_group", "optimizer_step_count_after_group", "attempted_group_count", "sample_index", "sample_cursor_before_group", "sample_cursor_after_step")}
    result["deterministic_attempt_seed"] = row.get("attempt_seed", row.get("step_seed"))
    result["rollout_seeds"] = [item.get("rollout_seed") for item in row.get("trajectories", [])]
    return result


def _execution_identity(a_metrics: str, b_metrics: str) -> Dict[str, Any]:
    a, b = _records(a_metrics), _records(b_metrics); per_step, first = {}, None
    for step in range(6, 11):
        left, right = a.get(step), b.get(step)
        if left is None or right is None:
            item = {"status": "missing_update_record", "a_present": left is not None, "b_present": right is not None}
        else:
            control_a, control_b = _control_projection(left), _control_projection(right)
            policy_a, policy_b = _identity_projection(left), _identity_projection(right)
            control = _exact_comparison(control_a, control_b, f"update_{step}.control")
            policy = _exact_comparison(policy_a, policy_b, f"update_{step}.policy")
            item = {"status": "ok" if control["equal"] and policy["equal"] else "different", "control_state": control,
                    "policy_dependent_trajectory": policy, "a_checksum": _hash(policy_a), "b_checksum": _hash(policy_b), "a": policy_a, "b": policy_b}
        per_step[str(step)] = item
        if first is None and item["status"] != "ok": first = {"global_step": step, **item}
    control_exact = all(item.get("control_state", {}).get("equal", False) for item in per_step.values())
    update6_exact = per_step["6"].get("policy_dependent_trajectory", {}).get("equal", False)
    return {"updates_6_to_10": per_step, "control_state_exact": control_exact,
            "update_6_policy_identity_exact": update6_exact, "first_policy_divergence": first}


def _state_compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    lora_a, lora_b = a["lora_state"], b["lora_state"]
    opt_a, opt_b = _optimizer_bank(a["optimizer_state"]), _optimizer_bank(b["optimizer_state"])
    lora = compare_grad_dicts(lora_a, lora_b); optimizer = compare_grad_dicts(opt_a, opt_b)
    return {"lora": {"global": lora, **_per_tensor(lora_a, lora_b)},
            "optimizer": {"global": optimizer, **_per_tensor(opt_a, opt_b)},
            "counters_equal": {k: a.get(k) == b.get(k) for k in ("optimizer_step_count", "attempted_group_count", "skipped_zero_advantage_group_count", "sample_cursor", "seed")},
            "rng_exact": _exact(a["rng_state"], b["rng_state"]),
            "optimizer_param_groups_exact": _exact(a["optimizer_state"].get("param_groups"), b["optimizer_state"].get("param_groups"))}


def _atomic_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp"); os.close(fd)
    try: torch.save(payload, tmp); os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp): os.unlink(tmp)
        raise


def main() -> None:
    args = parse_args(); output = Path(args.output_json).resolve(); report: Dict[str, Any] = {"status": "running"}
    try:
        if args.checkpoint_load_oracle:
            source = _load(args.checkpoint_load_oracle); resaved = Path(args.resaved_checkpoint or output.with_suffix(".resaved.pt"))
            snapshot = copy.deepcopy(source); _atomic_save(resaved, snapshot); roundtrip = _load(resaved)
            fields = ("trainable_state", "lora_state", "optimizer_state", "rng_state", "global_step", "optimizer_step_count", "attempted_group_count", "skipped_zero_advantage_group_count", "sample_cursor", "seed")
            comparisons = {field: _exact_comparison(source.get(field), roundtrip.get(field), field) for field in fields}
            exact = {field: item["equal"] for field, item in comparisons.items()}
            report.update({"mode": "checkpoint_load_oracle", "source_checkpoint": str(args.checkpoint_load_oracle), "resaved_checkpoint": str(resaved),
                           "source_hashes": _checkpoint_hashes(source), "resaved_hashes": _checkpoint_hashes(roundtrip), "exact_fields": exact,
                           "field_comparisons": comparisons,
                           "rng_exact_fields": {key: _exact_comparison(source["rng_state"].get(key), roundtrip["rng_state"].get(key), f"rng_state.{key}") for key in ("torch_cpu", "torch_cuda", "python", "numpy")},
                           "optimizer_param_groups_exact": _exact_comparison(source["optimizer_state"].get("param_groups"), roundtrip["optimizer_state"].get("param_groups"), "optimizer_state.param_groups"),
                           "all_504_lora_per_tensor": _per_tensor(source["lora_state"], roundtrip["lora_state"]),
                           "adamw_exp_avg_exp_avg_sq_step_per_state_key": _per_tensor(_optimizer_bank(source["optimizer_state"]), _optimizer_bank(roundtrip["optimizer_state"])),
                           "status": "passed" if all(exact.values()) else "failed"})
        else:
            required = (args.uninterrupted_a_checkpoint, args.uninterrupted_a_metrics, args.uninterrupted_b_checkpoint, args.uninterrupted_b_metrics, args.resumed_checkpoint, args.resumed_metrics, args.step5_checkpoint, args.resumed_summary)
            if not all(required): raise ValueError("full comparison requires A/B/resumed checkpoints+metrics and step-5 checkpoint")
            a, b, resumed, step5 = _load(args.uninterrupted_a_checkpoint), _load(args.uninterrupted_b_checkpoint), _load(args.resumed_checkpoint), _load(args.step5_checkpoint)
            baseline = _state_compare(a, b); resumed_cmp = _state_compare(a, resumed)
            normalized = {"lora": normalize_ab_by_a12_baseline(resumed_cmp["lora"]["global"], baseline["lora"]["global"])}
            optimizer_scale = _optimizer_scale_aware(a, b, resumed)
            execution = _execution_identity(args.uninterrupted_a_metrics, args.resumed_metrics)
            per_update_state = _per_update_state_observability(args.uninterrupted_a_metrics, args.uninterrupted_b_metrics, args.resumed_metrics)
            resumed_summary = json.loads(Path(args.resumed_summary).read_text())
            step5_load = {"checkpoint_hashes": _checkpoint_hashes(step5), "lora_tensor_count": len(step5["lora_state"]),
                          "optimizer_state_tensor_count": len(_optimizer_bank(step5["optimizer_state"])),
                          "counters": {k: step5.get(k) for k in ("optimizer_step_count", "attempted_group_count", "skipped_zero_advantage_group_count", "sample_cursor")},
                          "rng_present": {k: k in step5["rng_state"] for k in ("torch_cpu", "torch_cuda", "python", "numpy")},
                          "actual_post_load_model_optimizer_snapshot": resumed_summary.get("state_immediately_after_loading_step5")}
            optimizer_finite = bool(resumed_cmp["optimizer"]["global"]["both_sides_all_finite"])
            pass_baseline = bool(normalized["lora"]["ab_no_worse_than_a12_baseline"] and optimizer_scale["no_state_exceeds_matched_ab_raw_and_scale_baseline"] and optimizer_scale["step_fields_exact"] and optimizer_finite)
            report.update({"mode": "ten_update_resume_equivalence", "checkpoint_hashes": {"uninterrupted_a": _checkpoint_hashes(a), "uninterrupted_b": _checkpoint_hashes(b), "resumed": _checkpoint_hashes(resumed), "step5": _checkpoint_hashes(step5)},
                           "execution_identity_updates_6_to_10": execution, "state_immediately_after_loading_step5": step5_load,
                           "per_update_state_divergence_observability": per_update_state,
                           "ten_update_pristine_repeatability_baseline_A_vs_B": baseline, "resumed_vs_uninterrupted_A": resumed_cmp,
                           "baseline_normalized_resumed_vs_uninterrupted": normalized, "optimizer_scale_aware_comparison": optimizer_scale,
                           "acceptance": {"control_state_seed_schedule_exact": execution["control_state_exact"], "update_6_inputs_and_execution_identity_exact": execution["update_6_policy_identity_exact"],
                           "final_lora_no_worse_than_10_update_baseline": bool(normalized["lora"]["ab_no_worse_than_a12_baseline"]),
                           "final_optimizer_no_worse_scale_aware": optimizer_scale["no_state_exceeds_matched_ab_raw_and_scale_baseline"],
                           "adamw_step_fields_exact": optimizer_scale["step_fields_exact"], "no_missing_nonfinite_optimizer_state": optimizer_finite},
                           "status": "passed" if execution["control_state_exact"] and execution["update_6_policy_identity_exact"] and pass_baseline else "resume_equivalence_failed"})
    except Exception as exc:
        report.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
    _write_report_safely(output, report); print(json.dumps({"status": report["status"], "output": str(output)}))
    if report["status"] not in ("passed",): raise SystemExit(1)


if __name__ == "__main__": main()
