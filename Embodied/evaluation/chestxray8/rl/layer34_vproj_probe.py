"""Instrument Hybrid layer-34 ``v_proj`` (LoRA) for NaN localization."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rl.nan_grad_diagnostics import analyze_grad_tensor
from rl.selective_saved_tensor_offload import sample_tensor_values


LAYER34_VPROJ_SUBSTR = "layers.34.self_attn.v_proj"
LORA_B_WEIGHT_SUBSTR = f"{LAYER34_VPROJ_SUBSTR}.lora_B.default.weight"


def _finite_sample_stats(tensor: Optional[torch.Tensor], *, name: str) -> Dict[str, Any]:
    if tensor is None or not torch.is_tensor(tensor):
        return {"name": name, "present": False}
    vals, idxs, finite = sample_tensor_values(tensor, max_samples=16)
    return {
        "name": name,
        "present": True,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "sample_lin_indices": list(idxs),
        "sample_values": list(vals),
        "sample_isfinite": list(finite),
        "all_samples_finite": bool(all(finite)) if finite else True,
        "has_nan_sample": any(v != v for v in vals),  # NaN != NaN
        "has_inf_sample": any(v == float("inf") or v == float("-inf") for v in vals),
    }


def find_layer34_vproj_modules(
    model: nn.Module,
) -> Dict[str, Tuple[str, nn.Module]]:
    """Return named modules under ``layers.34.self_attn.v_proj``."""
    found: Dict[str, Tuple[str, nn.Module]] = {}
    for name, module in model.named_modules():
        if LAYER34_VPROJ_SUBSTR not in name:
            continue
        if name.endswith(LAYER34_VPROJ_SUBSTR) or name.endswith(
            f"{LAYER34_VPROJ_SUBSTR}.base_layer"
        ):
            key = "v_proj" if name.endswith(LAYER34_VPROJ_SUBSTR) else "base_layer"
            found[key] = (name, module)
        if ".lora_A.default" in name and name.endswith("lora_A.default"):
            found["lora_A"] = (name, module)
        if ".lora_B.default" in name and name.endswith("lora_B.default"):
            found["lora_B"] = (name, module)
    return found


class Layer34VProjProbe:
    """Forward/backward finite-stat probe for layer-34 ``v_proj`` LoRA path."""

    def __init__(self, model: nn.Module, *, sync_cuda: bool = True) -> None:
        self.model = model
        self.sync_cuda = bool(sync_cuda)
        self.handles: List[Any] = []
        self.forward_events: List[Dict[str, Any]] = []
        self.backward_events: List[Dict[str, Any]] = []
        self.lora_b_grad_events: List[Dict[str, Any]] = []
        self.modules = find_layer34_vproj_modules(model)
        self.first_nonfinite_event: Optional[Dict[str, Any]] = None

    def _sync(self) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _note_nonfinite(self, event: Dict[str, Any]) -> None:
        if self.first_nonfinite_event is not None:
            return
        bad = False
        for key, value in event.items():
            if isinstance(value, dict) and value.get("all_samples_finite") is False:
                bad = True
            if isinstance(value, dict) and value.get("all_finite") is False:
                bad = True
        if bad:
            self.first_nonfinite_event = event

    def install(self) -> None:
        if "v_proj" not in self.modules:
            return
        v_name, v_mod = self.modules["v_proj"]

        def _fwd_v(mod, inputs, output, *, _name=v_name):
            self._sync()
            x = inputs[0] if inputs else None
            event = {
                "phase": "forward",
                "module": _name,
                "input": _finite_sample_stats(x, name="v_proj_input"),
                "output": _finite_sample_stats(
                    output if torch.is_tensor(output) else None, name="v_proj_output"
                ),
            }
            self.forward_events.append(event)
            self._note_nonfinite(event)

        def _bwd_v(mod, grad_input, grad_output, *, _name=v_name):
            self._sync()
            gout = None
            if grad_output:
                gout = grad_output[0] if grad_output[0] is not None else None
            event = {
                "phase": "backward",
                "module": _name,
                "grad_output": (
                    analyze_grad_tensor(gout)
                    if gout is not None
                    else {"grad_is_none": True}
                ),
                "grad_output_samples": _finite_sample_stats(
                    gout, name="v_proj_grad_output"
                ),
            }
            self.backward_events.append(event)
            self._note_nonfinite(event)

        self.handles.append(v_mod.register_forward_hook(_fwd_v))
        try:
            self.handles.append(v_mod.register_full_backward_hook(_bwd_v))
        except Exception:
            pass

        if "lora_A" in self.modules:
            a_name, a_mod = self.modules["lora_A"]

            def _fwd_a(mod, inputs, output, *, _name=a_name):
                self._sync()
                event = {
                    "phase": "forward_lora_A",
                    "module": _name,
                    "input": _finite_sample_stats(
                        inputs[0] if inputs else None, name="lora_A_input"
                    ),
                    "output": _finite_sample_stats(
                        output if torch.is_tensor(output) else None,
                        name="lora_A_intermediate",
                    ),
                }
                self.forward_events.append(event)
                self._note_nonfinite(event)

            self.handles.append(a_mod.register_forward_hook(_fwd_a))

        # LoRA B weight gradient (the first-NaN parameter from N1).
        for name, parameter in self.model.named_parameters():
            if LORA_B_WEIGHT_SUBSTR not in name:
                continue
            if not parameter.requires_grad:
                continue

            def _param_hook(grad, *, _name=name):
                self._sync()
                stats = analyze_grad_tensor(grad)
                event = {
                    "phase": "lora_B_weight_grad",
                    "parameter": _name,
                    "grad_stats": stats,
                    "grad_samples": _finite_sample_stats(grad, name="lora_B_grad"),
                }
                self.lora_b_grad_events.append(event)
                self._note_nonfinite(event)
                return grad

            self.handles.append(parameter.register_hook(_param_hook))

    def close(self) -> None:
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()

    def report(self) -> Dict[str, Any]:
        return {
            "target_substr": LAYER34_VPROJ_SUBSTR,
            "lora_b_weight_substr": LORA_B_WEIGHT_SUBSTR,
            "modules_found": {key: name for key, (name, _m) in self.modules.items()},
            "forward_event_count": len(self.forward_events),
            "backward_event_count": len(self.backward_events),
            "lora_b_grad_event_count": len(self.lora_b_grad_events),
            "forward_events": self.forward_events[-8:],
            "backward_events": self.backward_events[-8:],
            "lora_b_grad_events": self.lora_b_grad_events[-8:],
            "first_nonfinite_event": self.first_nonfinite_event,
        }


def filter_unpacked_for_layer34(
    selective_summary: Optional[Dict[str, Any]],
    saved_tensor_log: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Select pack/unpack catalog rows whose module context touches layer-34 v_proj."""
    log = list(saved_tensor_log or [])
    matched = [
        entry
        for entry in log
        if LAYER34_VPROJ_SUBSTR in str(entry.get("module_context") or "")
    ]
    first_nf = None
    if isinstance(selective_summary, dict):
        cand = selective_summary.get("first_nonfinite_unpack")
        if isinstance(cand, dict) and LAYER34_VPROJ_SUBSTR in str(
            cand.get("module_context") or ""
        ):
            first_nf = cand
    return {
        "matched_saved_tensor_entries": matched[:64],
        "matched_count": len(matched),
        "storage_ids": sorted(
            {
                int(e["shared_storage_id"])
                for e in matched
                if e.get("shared_storage_id") is not None
            }
        ),
        "first_nonfinite_unpack_for_layer34": first_nf,
    }
