"""LoRA-specific diagnostics for production-cached enable_grad replay.

Diagnostic-only helpers. Does not alter GRPO objective, rewards, or default
training cache semantics. Optional PEFT forward patches are opt-in via CLI.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LORA_ISOLATION_GROUPS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "mlp",
    "layer0",
    "layer3",
    "all",
)

WORKAROUND_MODES = (
    "none",
    "contiguous",
    "fp32_lora",
    "contiguous_fp32_lora",
)


def cuda_memory_snapshot(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "device": str(device),
            "allocated_bytes": None,
            "reserved_bytes": None,
            "max_allocated_bytes": None,
            "max_reserved_bytes": None,
        }
    index = device.index if device.index is not None else torch.cuda.current_device()
    free_b, total_b = torch.cuda.mem_get_info(index)
    return {
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        "free_bytes": int(free_b),
        "total_bytes": int(total_b),
        "approx_used_fraction_of_total": float(1.0 - (free_b / max(total_b, 1))),
    }


def rich_tensor_meta(value: Any, *, name: str = "tensor") -> Any:
    if not torch.is_tensor(value):
        return {"name": name, "type": type(value).__name__}
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
        "grad_fn": type(value.grad_fn).__name__ if value.grad_fn is not None else None,
        "is_contiguous": bool(value.is_contiguous()),
        "stride": list(value.stride()),
        "storage_offset": int(value.storage_offset())
        if value.device.type != "meta"
        else None,
        "numel": int(value.numel()),
    }


def estimate_autograd_graph_nodes(roots: Sequence[torch.Tensor], *, limit: int = 200000) -> Dict[str, Any]:
    """Traverse grad_fn graph; count nodes and AccumulateGrad leaves."""
    seen: Set[int] = set()
    stack: List[Any] = []
    trainable_leaves = 0
    for root in roots:
        if torch.is_tensor(root) and root.grad_fn is not None:
            stack.append(root.grad_fn)
    while stack:
        node = stack.pop()
        if node is None:
            continue
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        if len(seen) >= limit:
            break
        if type(node).__name__ == "AccumulateGrad":
            variable = getattr(node, "variable", None)
            if variable is not None and bool(getattr(variable, "requires_grad", False)):
                trainable_leaves += 1
            continue
        next_fns = getattr(node, "next_functions", None)
        if not next_fns:
            continue
        for child, _idx in next_fns:
            if child is not None and id(child) not in seen:
                stack.append(child)
    return {
        "root_count": len(roots),
        "approx_graph_nodes": len(seen),
        "truncated": len(seen) >= limit,
        "trainable_accumulate_grad_nodes": trainable_leaves,
    }


def analyze_past_key_values_grad(past_key_values) -> Dict[str, Any]:
    """Report whether carried KV tensors retain autograd links (diagnostic)."""
    if past_key_values is None:
        return {"present": False}
    tensors: List[torch.Tensor] = []
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for key, value in zip(past_key_values.key_cache, past_key_values.value_cache):
            if torch.is_tensor(key):
                tensors.append(key)
            if torch.is_tensor(value):
                tensors.append(value)
    elif isinstance(past_key_values, (tuple, list)):
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                continue
            if torch.is_tensor(layer[0]):
                tensors.append(layer[0])
            if torch.is_tensor(layer[1]):
                tensors.append(layer[1])
    else:
        return {"present": True, "type": type(past_key_values).__name__, "unparsed": True}

    any_requires_grad = any(bool(t.requires_grad) for t in tensors)
    any_grad_fn = any(t.grad_fn is not None for t in tensors)
    graph = (
        estimate_autograd_graph_nodes(tensors)
        if any_grad_fn
        else {
            "approx_graph_nodes": 0,
            "trainable_accumulate_grad_nodes": 0,
            "truncated": False,
            "root_count": 0,
        }
    )
    sample = []
    for index, tensor in enumerate(tensors[:4]):
        sample.append(rich_tensor_meta(tensor, name=f"kv_{index}"))
    carries = bool(
        any_grad_fn and int(graph.get("trainable_accumulate_grad_nodes", 0)) > 0
    )
    # Fallback: retained grad_fn on requires_grad KV is already evidence of live
    # history even if AccumulateGrad discovery fails on some PyTorch builds.
    if any_grad_fn and any_requires_grad:
        carries = True
    return {
        "present": True,
        "num_tensors": len(tensors),
        "any_requires_grad": any_requires_grad,
        "any_grad_fn": any_grad_fn,
        "carries_grad_into_trainable_params": carries,
        "autograd_graph": graph,
        "sample_tensors": sample,
        "shapes": [list(t.shape) for t in tensors[:8]],
    }


def _module_qualified_name(root: nn.Module, target: nn.Module) -> Optional[str]:
    for name, module in root.named_modules():
        if module is target:
            return name
    return None


def iter_lora_linear_modules(model) -> List[Tuple[str, nn.Module]]:
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except Exception:  # pragma: no cover
        LoraLinear = tuple()  # type: ignore[assignment]
    found: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if LoraLinear and isinstance(module, LoraLinear):
            found.append((name, module))
            continue
        # Fallback: duck-type PEFT LoRA linear.
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(
            module, "base_layer"
        ):
            found.append((name, module))
    return found


def lora_name_matches_isolation(module_name: str, isolation: str) -> bool:
    isolation = str(isolation)
    lower = module_name.lower()
    if isolation == "all":
        return True
    if isolation == "mlp":
        return any(
            token in lower
            for token in (".mlp.", "gate_proj", "up_proj", "down_proj")
        )
    if isolation == "layer0":
        return "layers.0." in lower
    if isolation == "layer3":
        return "layers.3." in lower
    # attention projections: match q_proj/k_proj/... path segments only.
    return f".{isolation}." in lower or lower.endswith(f".{isolation}") or (
        f"{isolation}.lora_" in lower
    )


def apply_lora_isolation(model, isolation: str) -> Dict[str, Any]:
    """Keep only selected LoRA params trainable; freeze projector and other LoRA."""
    if isolation not in LORA_ISOLATION_GROUPS:
        raise RuntimeError(f"unknown lora isolation {isolation!r}")
    enabled: List[str] = []
    disabled: List[str] = []
    for name, parameter in model.named_parameters():
        if name.startswith("mlp1."):
            parameter.requires_grad_(False)
            continue
        if "lora_" not in name.lower():
            parameter.requires_grad_(False)
            continue
        keep = lora_name_matches_isolation(name, isolation)
        parameter.requires_grad_(keep)
        (enabled if keep else disabled).append(name)
    return {
        "lora_isolation": isolation,
        "enabled_lora_param_count": len(enabled),
        "disabled_lora_param_count": len(disabled),
        "first_enabled_lora_param": enabled[0] if enabled else None,
        "enabled_lora_params_sample": enabled[:12],
    }


def truncate_trace_scored_blocks(trace, max_scored_blocks: Optional[int]):
    """Diagnostic: keep only the first N scored_for_grpo blocks (and their actions)."""
    if max_scored_blocks is None:
        return trace, {"truncated": False}
    if int(max_scored_blocks) < 1:
        raise RuntimeError("max_scored_blocks must be >= 1")
    kept_blocks = []
    scored = 0
    generated: List[int] = []
    for block in trace.blocks:
        if block.scored_for_grpo:
            if scored >= int(max_scored_blocks):
                break
            scored += 1
        kept_blocks.append(block)
        generated.extend(list(block.action_token_ids))
    # Rebuild a shallow copy-like object using the dataclass constructor fields.
    new_trace = type(trace)(
        prompt_token_ids=list(trace.prompt_token_ids),
        generated_token_ids=generated,
        blocks=kept_blocks,
        sampling=trace.sampling,
        stopped_on_eos=False,
        truncated=True,
        decoded_text=getattr(trace, "decoded_text", None),
        decoder_path=getattr(trace, "decoder_path", "hybrid"),
        reward_branch=getattr(trace, "reward_branch", "none"),
        committed_final_box_norm_1000=getattr(
            trace, "committed_final_box_norm_1000", None
        ),
        has_unambiguous_committed_box=getattr(
            trace, "has_unambiguous_committed_box", False
        ),
        fallback_triggered=getattr(trace, "fallback_triggered", False),
        rejected_pbd_proposals=list(
            getattr(trace, "rejected_pbd_proposals", []) or []
        ),
    )
    return new_trace, {
        "truncated": True,
        "max_scored_blocks": int(max_scored_blocks),
        "kept_blocks": len(kept_blocks),
        "kept_scored_blocks": scored,
        "kept_generated_len": len(generated),
    }


def _lora_delta(
    module: nn.Module,
    x: torch.Tensor,
    *,
    contiguous_input: bool,
    fp32_lora: bool,
) -> torch.Tensor:
    active = list(getattr(module, "active_adapters", []) or [])
    if not active:
        active = list(getattr(module, "lora_A", {}).keys())
    delta = None
    x_work = x.contiguous() if contiguous_input else x
    for adapter in active:
        if adapter not in module.lora_A:
            continue
        lora_A = module.lora_A[adapter]
        lora_B = module.lora_B[adapter]
        dropout = module.lora_dropout[adapter]
        scaling = module.scaling[adapter]
        x_a = dropout(x_work)
        if fp32_lora:
            x_a = x_a.float()
            a_w = lora_A.weight.float()
            b_w = lora_B.weight.float()
            mid = F.linear(x_a, a_w)
            out = F.linear(mid, b_w) * float(scaling)
            out = out.to(dtype=x.dtype)
        else:
            x_a = x_a.to(lora_A.weight.dtype)
            out = lora_B(lora_A(x_a)) * scaling
            out = out.to(dtype=x.dtype)
        delta = out if delta is None else delta + out
    if delta is None:
        return torch.zeros_like(x)
    return delta


@contextmanager
def peft_lora_forward_workaround(
    model,
    *,
    mode: str = "none",
    probe_sink: Optional[Dict[str, Any]] = None,
    probe_layer_index: Optional[int] = None,
    probe_block_getter=None,
) -> Iterator[Dict[str, Any]]:
    """Optionally patch PEFT LoRA Linear.forward for diagnostics / safe math.

    Modes:
      none: observe only (optional probe logging)
      contiguous: x = x.contiguous() before LoRA A/B
      fp32_lora: LoRA A/B in fp32, cast delta back
      contiguous_fp32_lora: both
    """
    if mode not in WORKAROUND_MODES:
        raise RuntimeError(f"unknown workaround mode {mode!r}")
    contiguous_input = mode in {"contiguous", "contiguous_fp32_lora"}
    fp32_lora = mode in {"fp32_lora", "contiguous_fp32_lora"}
    modules = iter_lora_linear_modules(model)
    originals = []
    report = {
        "workaround_mode": mode,
        "patched_module_count": len(modules),
        "contiguous_input": contiguous_input,
        "fp32_lora": fp32_lora,
    }

    def _layer_index_from_name(name: str) -> Optional[int]:
        marker = ".layers."
        if marker not in name:
            return None
        try:
            return int(name.split(marker, 1)[1].split(".", 1)[0])
        except Exception:
            return None

    for name, module in modules:
        original = module.forward

        def make_forward(mod, mod_name, orig):
            def forward(x, *args, **kwargs):
                adapter_names = kwargs.get("adapter_names", None)
                if getattr(mod, "disable_adapters", False) or getattr(mod, "merged", False):
                    return orig(x, *args, **kwargs)
                if adapter_names is not None:
                    return orig(x, *args, **kwargs)

                layer_index = _layer_index_from_name(mod_name)
                should_probe = (
                    probe_sink is not None
                    and probe_layer_index is not None
                    and layer_index == int(probe_layer_index)
                    and callable(probe_block_getter)
                    and probe_block_getter() is not None
                )
                if should_probe:
                    key = f"layer{layer_index}:{mod_name}"
                    entry = probe_sink.setdefault(key, {})
                    entry["input"] = rich_tensor_meta(x, name="input")
                    try:
                        with torch.no_grad():
                            base_out = mod.base_layer(x)
                        entry["base_output"] = rich_tensor_meta(base_out, name="base_out")
                        # Intermediate A/B shapes under current workaround settings.
                        x_work = x.contiguous() if contiguous_input else x
                        for adapter in list(getattr(mod, "active_adapters", []) or []):
                            if adapter not in mod.lora_A:
                                continue
                            a = mod.lora_A[adapter]
                            b = mod.lora_B[adapter]
                            x_a = x_work.to(a.weight.dtype)
                            if fp32_lora:
                                mid = F.linear(x_a.float(), a.weight.float())
                            else:
                                mid = a(x_a)
                            entry[f"lora_A_{adapter}"] = rich_tensor_meta(
                                mid, name=f"lora_A_{adapter}"
                            )
                            entry[f"lora_A_weight_{adapter}"] = rich_tensor_meta(
                                a.weight, name=f"A_w_{adapter}"
                            )
                            entry[f"lora_B_weight_{adapter}"] = rich_tensor_meta(
                                b.weight, name=f"B_w_{adapter}"
                            )
                    except Exception as exc:  # pragma: no cover
                        entry["probe_error"] = f"{type(exc).__name__}: {exc}"

                if mode == "none":
                    return orig(x, *args, **kwargs)

                # Exact PEFT math with safer casts/contiguity.
                result = mod.base_layer(x, *args, **kwargs)
                torch_result_dtype = result.dtype
                delta = _lora_delta(
                    mod,
                    x,
                    contiguous_input=contiguous_input,
                    fp32_lora=fp32_lora,
                )
                result = result + delta
                if should_probe:
                    probe_sink[f"layer{layer_index}:{mod_name}"][
                        "patched_output"
                    ] = rich_tensor_meta(result, name="patched_out")
                return result.to(torch_result_dtype)

            return forward

        originals.append((module, original))
        module.forward = make_forward(module, name, original)  # type: ignore[method-assign]

    try:
        yield report
    finally:
        for module, original in originals:
            module.forward = original  # type: ignore[method-assign]


def exact_lora_delta_reference(
    x: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    *,
    scaling: float,
    contiguous_input: bool = False,
    fp32_lora: bool = False,
) -> torch.Tensor:
    """Reference LoRA delta used by CPU equivalence tests."""
    x_work = x.contiguous() if contiguous_input else x
    if fp32_lora:
        mid = F.linear(x_work.float(), a_weight.float())
        out = F.linear(mid, b_weight.float()) * float(scaling)
        return out.to(dtype=x.dtype)
    mid = F.linear(x_work.to(a_weight.dtype), a_weight)
    out = F.linear(mid, b_weight) * float(scaling)
    return out.to(dtype=x.dtype)


def peft_style_lora_forward(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    a_weight: torch.Tensor,
    b_weight: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    """Match peft.tuners.lora.Linear math (no dropout): base(x)+B(A(x))*scale."""
    result = F.linear(x, base_weight)
    x_a = x.to(a_weight.dtype)
    delta = F.linear(F.linear(x_a, a_weight), b_weight) * float(scaling)
    return (result + delta.to(result.dtype)).to(result.dtype)
