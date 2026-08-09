"""Diagnostic probe for layers.0.mlp.down_proj LoRA vs base paths."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


def _tensor_stats(t: Optional[torch.Tensor], *, name: str) -> Dict[str, Any]:
    if t is None or not torch.is_tensor(t):
        return {"name": name, "present": False}
    x = t.detach()
    finite = torch.isfinite(x)
    return {
        "name": name,
        "present": True,
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "storage_offset": int(x.storage_offset()),
        "dtype": str(x.dtype).replace("torch.", ""),
        "device": str(x.device),
        "numel": int(x.numel()),
        "finite_count": int(finite.sum().item()),
        "abs_max": float(x[finite].abs().max().item()) if finite.any() else None,
        "abs_min": float(x[finite].abs().min().item()) if finite.any() else None,
        "l2": float(x[finite].float().norm().item()) if finite.any() else None,
        "storage_data_ptr": int(x.untyped_storage().data_ptr()),
    }


def _find_down_proj(model: nn.Module) -> Optional[tuple]:
    for name, module in model.named_modules():
        if name.endswith("layers.0.mlp.down_proj") or name.endswith(
            "model.layers.0.mlp.down_proj"
        ):
            return name, module
        if "layers.0.mlp.down_proj" in name and name.endswith("down_proj"):
            return name, module
    return None


class DownProjLoraProbe:
    """Capture base/LoRA IO and lora_B weight grad for layers.0.mlp.down_proj."""

    def __init__(self, model: nn.Module, *, sync_cuda: bool = True) -> None:
        self.model = model
        self.sync_cuda = bool(sync_cuda)
        self.records: List[Dict[str, Any]] = []
        self._handles: List[Any] = []
        self._found = _find_down_proj(model)
        self.lora_B_weight = None
        self.lora_B_name = None

    def install(self) -> None:
        if self._found is None:
            return
        name, module = self._found
        # Prefer PEFT ModuleDict adapters.
        lora_A = getattr(module, "lora_A", None)
        lora_B = getattr(module, "lora_B", None)
        dropout = getattr(module, "lora_dropout", None)
        adapters = []
        if isinstance(lora_A, nn.ModuleDict):
            adapters = list(lora_A.keys())
        elif lora_A is not None:
            adapters = ["default"]

        def _sync() -> None:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

        def _base_hook(_mod, inputs, output):
            _sync()
            inp = inputs[0] if inputs else None
            self.records.append(
                {
                    "event": "base_down_proj",
                    "module": name,
                    "input": _tensor_stats(inp, name="base_input"),
                    "output": _tensor_stats(
                        output if torch.is_tensor(output) else None, name="base_output"
                    ),
                }
            )

        self._handles.append(module.register_forward_hook(_base_hook))

        for adapter in adapters:
            a_mod = lora_A[adapter] if isinstance(lora_A, nn.ModuleDict) else lora_A
            b_mod = lora_B[adapter] if isinstance(lora_B, nn.ModuleDict) else lora_B
            d_mod = (
                dropout[adapter]
                if isinstance(dropout, nn.ModuleDict)
                else dropout
            )
            if a_mod is not None:

                def _a_hook(_mod, inputs, output, adapter=adapter):
                    _sync()
                    inp = inputs[0] if inputs else None
                    self.records.append(
                        {
                            "event": "lora_A",
                            "adapter": adapter,
                            "input": _tensor_stats(inp, name="lora_A_input"),
                            "output": _tensor_stats(
                                output if torch.is_tensor(output) else None,
                                name="lora_A_output",
                            ),
                        }
                    )

                self._handles.append(a_mod.register_forward_hook(_a_hook))
            if d_mod is not None and isinstance(d_mod, nn.Module):

                def _d_hook(_mod, inputs, output, adapter=adapter):
                    _sync()
                    self.records.append(
                        {
                            "event": "lora_dropout",
                            "adapter": adapter,
                            "input": _tensor_stats(
                                inputs[0] if inputs else None, name="dropout_input"
                            ),
                            "output": _tensor_stats(
                                output if torch.is_tensor(output) else None,
                                name="dropout_output",
                            ),
                        }
                    )

                self._handles.append(d_mod.register_forward_hook(_d_hook))
            if b_mod is not None:
                self.lora_B_weight = b_mod.weight
                self.lora_B_name = f"{name}.lora_B.{adapter}.weight"

                def _b_fwd(_mod, inputs, output, adapter=adapter):
                    _sync()
                    self.records.append(
                        {
                            "event": "lora_B_forward",
                            "adapter": adapter,
                            "input": _tensor_stats(
                                inputs[0] if inputs else None, name="lora_B_input"
                            ),
                            "output": _tensor_stats(
                                output if torch.is_tensor(output) else None,
                                name="lora_B_output",
                            ),
                        }
                    )

                def _b_bwd(_mod, grad_input, grad_output, adapter=adapter):
                    _sync()
                    gout = grad_output[0] if grad_output else None
                    self.records.append(
                        {
                            "event": "lora_B_backward",
                            "adapter": adapter,
                            "grad_output": _tensor_stats(
                                gout, name="lora_B_grad_output"
                            ),
                        }
                    )

                self._handles.append(b_mod.register_forward_hook(_b_fwd))
                self._handles.append(b_mod.register_full_backward_hook(_b_bwd))

    def capture_lora_B_weight_grad(self) -> Dict[str, Any]:
        if self.lora_B_weight is None:
            return {"present": False}
        g = self.lora_B_weight.grad
        out = _tensor_stats(g, name="lora_B_weight_grad")
        out["parameter"] = self.lora_B_name
        return out

    def report(self) -> Dict[str, Any]:
        return {
            "target": "layers.0.mlp.down_proj",
            "found": self._found is not None,
            "module_name": self._found[0] if self._found else None,
            "records": self.records[:64],
            "lora_B_weight_grad": self.capture_lora_B_weight_grad(),
        }

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def compare_down_proj_probes(
    a_report: Dict[str, Any], b_report: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare compact stats from A vs B down_proj probes."""
    def _by_event(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in report.get("records") or []:
            key = f"{row.get('event')}:{row.get('adapter', '')}"
            out[key] = row
        return out

    a_map = _by_event(a_report)
    b_map = _by_event(b_report)
    keys = sorted(set(a_map) | set(b_map))
    comparisons = []
    for key in keys:
        ar = a_map.get(key) or {}
        br = b_map.get(key) or {}
        entry: Dict[str, Any] = {"event_key": key}
        for field in ("input", "output", "grad_output"):
            sa = ar.get(field) or {}
            sb = br.get(field) or {}
            if not sa.get("present") and not sb.get("present"):
                continue
            entry[field] = {
                "a": sa,
                "b": sb,
                "shape_match": sa.get("shape") == sb.get("shape"),
                "stride_match": sa.get("stride") == sb.get("stride"),
                "l2_abs_diff": (
                    abs(float(sa.get("l2") or 0.0) - float(sb.get("l2") or 0.0))
                    if sa.get("l2") is not None and sb.get("l2") is not None
                    else None
                ),
            }
        comparisons.append(entry)
    ga = (a_report.get("lora_B_weight_grad") or {})
    gb = (b_report.get("lora_B_weight_grad") or {})
    return {
        "events_compared": comparisons,
        "lora_B_weight_grad": {
            "a": ga,
            "b": gb,
            "l2_abs_diff": (
                abs(float(ga.get("l2") or 0.0) - float(gb.get("l2") or 0.0))
                if ga.get("l2") is not None and gb.get("l2") is not None
                else None
            ),
            "shape_match": ga.get("shape") == gb.get("shape"),
        },
    }
