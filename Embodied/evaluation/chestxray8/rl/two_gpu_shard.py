"""Exact two-GPU, live-KV decoder sharding for LocateAnything-3B.

This module deliberately shards *one* decoder invocation across devices instead
of using a device map with per-forward hooks.  The only differentiable transfer
is the hidden state between decoder layers 17 and 18; a custom identity-Jacobian
transport makes a transient FP32 CPU stage because direct BF16 *and* FP32 P2P
copies return zeros on the audited A4000 pair. The CPU value is not retained by
autograd. The cache object is shared by the decoder loop, but each layer writes
only its own K/V entry, which stays on the assigned device across replay blocks.

This is the thesis backend's placement mechanism.  It is not activation
offload, does not detach K/V, and does not checkpoint / recompute layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch


def _numeric_tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    """Synchronous debug-only finite summary without changing tensor values."""
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "all_finite": bool(finite.all().item()),
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(detached).sum().item()),
        "positive_inf_count": int(torch.isposinf(detached).sum().item()),
        "negative_inf_count": int(torch.isneginf(detached).sum().item()),
        "finite_min": float(finite_values.min().float().item())
        if finite_values.numel()
        else None,
        "finite_max": float(finite_values.max().float().item())
        if finite_values.numel()
        else None,
    }


class _ExactBF16PeerCopy(torch.autograd.Function):
    """Identity Jacobian with transient CPU staging for floating A4000 P2P."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, target: torch.device) -> torch.Tensor:
        ctx.source_device = value.device
        ctx.source_dtype = value.dtype
        # Autograd Function.forward executes without recording its internals,
        # so neither FP32 temporary is retained with the trajectory graph.
        if value.dtype == torch.bfloat16:
            return _bf16_values_via_fp32(value, target)
        return value.detach().to("cpu").to(target, non_blocking=False)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.source_dtype == torch.bfloat16:
            grad_input = _bf16_values_via_fp32(grad_output, ctx.source_device)
        else:
            grad_input = grad_output.detach().to("cpu").to(
                ctx.source_device, dtype=ctx.source_dtype, non_blocking=False
            )
        return grad_input, None


def _copy_across_shards(value: torch.Tensor, target: torch.device) -> torch.Tensor:
    """Value-exact transport around broken A4000 floating P2P copies."""
    if value.device == target:
        return value
    if value.is_floating_point():
        return _ExactBF16PeerCopy.apply(value, target)
    return value.detach().to("cpu").to(target, non_blocking=False)


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor values without relying on NumPy BF16 support."""
    cpu = value.detach().to("cpu").contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _bf16_values_via_fp32(
    value: torch.Tensor, target: str | torch.device
) -> torch.Tensor:
    """Copy BF16 via transient CPU FP32, avoiding broken BF16/FP32 P2P."""
    if value.dtype != torch.bfloat16:
        raise TypeError("FP32-mediated initialization transport requires BF16 input")
    return value.detach().float().to("cpu").to(
        torch.device(target), non_blocking=False
    ).to(torch.bfloat16)


def _copy_initialization_value_exact(
    value: torch.Tensor,
    target: str | torch.device,
    *,
    label: str,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Move one initialization tensor and synchronously prove value equality.

    This function is deliberately separate from ``_copy_across_shards``:
    model-state migration is not part of an autograd graph, while activation
    transport must retain the custom identity Jacobian.
    """
    target_device = torch.device(target)
    source_device = value.device
    source_cpu = value.detach().to("cpu").clone()
    source_hash = _tensor_sha256(source_cpu)
    if source_device == target_device:
        copied = value
        transport = "same_device_noop"
    elif value.dtype == torch.bfloat16:
        copied = _bf16_values_via_fp32(value, target_device)
        transport = "bf16_to_fp32_cpu_stage_to_target_to_bf16_synchronous"
    else:
        copied = value.detach().to("cpu").to(target_device, non_blocking=False)
        transport = "synchronous_cpu_stage_non_bf16"
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    target_cpu = copied.detach().to("cpu")
    exact = bool(torch.equal(source_cpu, target_cpu))
    target_hash = _tensor_sha256(target_cpu)
    report = {
        "label": label,
        "source_device": str(source_device),
        "target_device": str(copied.device),
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "source_stride": list(value.stride()),
        "target_stride": list(copied.stride()),
        "source_layout": str(value.layout),
        "target_layout": str(copied.layout),
        "transport": transport,
        "source_sha256": source_hash,
        "target_sha256": target_hash,
        "exact_value_equality": exact,
    }
    if not exact:
        raise RuntimeError(
            "initialization-time shard migration changed tensor values: "
            + json.dumps(report, sort_keys=True)
        )
    return copied, report


def _move_module_initialization_exact(
    module: torch.nn.Module,
    target: str | torch.device,
    *,
    label: str,
) -> Dict[str, Any]:
    """Apply value-checked initialization transport to all params/buffers."""
    records: List[Dict[str, Any]] = []

    def move(value: torch.Tensor) -> torch.Tensor:
        copied, record = _copy_initialization_value_exact(
            value, target, label=f"{label}.tensor_{len(records)}"
        )
        records.append(record)
        return copied

    module._apply(move)
    return {
        "label": label,
        "target_device": str(torch.device(target)),
        "tensor_count": len(records),
        "bf16_peer_tensor_count": sum(
            item["transport"]
            == "bf16_to_fp32_cpu_stage_to_target_to_bf16_synchronous"
            for item in records
        ),
        "all_exact": all(item["exact_value_equality"] for item in records),
        "tensors": records,
    }


def diagnose_direct_bf16_peer_copy(
    source_device: str | torch.device,
    target_device: str | torch.device,
) -> Dict[str, Any]:
    """Classify the historical direct-BF16 backend without mutating a model."""
    source_device = torch.device(source_device)
    target_device = torch.device(target_device)
    fixture_cpu = torch.tensor(
        [
            -9984.0,
            -39.0,
            -1.0,
            -0.0,
            0.0,
            1.0,
            29.125,
            1024.0,
            float("inf"),
            float("-inf"),
        ],
        dtype=torch.bfloat16,
    )
    # Diagnose host ingress separately. The peer-copy fixture itself must be
    # known-good before it is used to classify the 0->1 transport.
    direct_host_ingress = fixture_cpu.to(source_device, non_blocking=False)
    safe_host_ingress = (
        fixture_cpu.float()
        .to(source_device, non_blocking=False)
        .to(torch.bfloat16)
    )
    if source_device.type == "cuda":
        torch.cuda.synchronize(source_device)
    direct_host_cpu = direct_host_ingress.to("cpu")
    safe_host_cpu = safe_host_ingress.to("cpu")
    source = safe_host_ingress
    direct = source.to(target_device, non_blocking=False)
    safe = _bf16_values_via_fp32(source, target_device)
    source_fp32 = source.float()
    fp32_peer = source_fp32.to(target_device, non_blocking=False)
    cpu_staged = (
        source_fp32.to("cpu")
        .to(target_device, non_blocking=False)
        .to(torch.bfloat16)
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    direct_cpu = direct.to("cpu")
    safe_cpu = safe.to("cpu")
    fp32_peer_cpu = fp32_peer.to("cpu")
    cpu_staged_cpu = cpu_staged.to("cpu")
    direct_exact = bool(torch.equal(direct_cpu, fixture_cpu))
    safe_exact = bool(torch.equal(safe_cpu, fixture_cpu))
    return {
        "source_device": str(source_device),
        "target_device": str(target_device),
        "fixture_values": fixture_cpu.tolist(),
        "source_values_after_safe_ingress": source.to("cpu").tolist(),
        "direct_host_ingress_values": direct_host_cpu.tolist(),
        "safe_host_ingress_values": safe_host_cpu.tolist(),
        "direct_values": direct_cpu.tolist(),
        "safe_values": safe_cpu.tolist(),
        "fp32_peer_values": fp32_peer_cpu.tolist(),
        "cpu_staged_values": cpu_staged_cpu.tolist(),
        "direct_sha256": _tensor_sha256(direct_cpu),
        "safe_sha256": _tensor_sha256(safe_cpu),
        "fp32_peer_sha256": _tensor_sha256(fp32_peer_cpu),
        "cpu_staged_sha256": _tensor_sha256(cpu_staged_cpu),
        "source_sha256": _tensor_sha256(fixture_cpu),
        "direct_host_ingress_exact": bool(torch.equal(direct_host_cpu, fixture_cpu)),
        "safe_host_ingress_exact": bool(torch.equal(safe_host_cpu, fixture_cpu)),
        "historical_direct_backend_status": (
            "available_for_diagnostic_comparison"
            if direct_exact
            else "unavailable_direct_bf16_peer_copy_corrupts_values"
        ),
        "direct_exact": direct_exact,
        "safe_exact": safe_exact,
        "fp32_peer_exact": bool(torch.equal(fp32_peer_cpu, fixture_cpu.float())),
        "cpu_staged_exact": bool(torch.equal(cpu_staged_cpu, fixture_cpu)),
        "cuda_can_device_access_peer": bool(
            torch.cuda.can_device_access_peer(
                int(source_device.index or 0), int(target_device.index or 0)
            )
        ),
    }


@dataclass(frozen=True)
class DecoderShardLayout:
    """Deterministic contiguous decoder assignment (default: 18 / 18)."""

    first_device: torch.device
    second_device: torch.device
    first_layer_count: int = 18

    def device_for_layer(self, index: int) -> torch.device:
        return self.first_device if index < self.first_layer_count else self.second_device


def materialize_shard_position_ids(
    position_ids: torch.Tensor,
    first_device: str | torch.device,
    second_device: str | torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Own position values independently on CPU and on each decoder shard."""
    position_ids_cpu = position_ids.detach().to("cpu").clone()
    early_position_ids = (
        position_ids_cpu.to(torch.device(first_device), dtype=torch.long, non_blocking=False)
        .contiguous().clone()
    )
    late_position_ids = (
        position_ids_cpu.to(torch.device(second_device), dtype=torch.long, non_blocking=False)
        .contiguous().clone()
    )
    return position_ids_cpu, early_position_ids, late_position_ids


@dataclass(frozen=True)
class ResolvedQwenDecoder:
    """One authoritative LocateAnything/PEFT resolution for a shard operation."""

    qwen: torch.nn.Module
    decoder: torch.nn.Module
    qwen_path: str
    decoder_path: str

    @property
    def embed_tokens(self) -> torch.nn.Module:
        return self.decoder.embed_tokens

    @property
    def norm(self) -> torch.nn.Module:
        return self.decoder.norm

    @property
    def lm_head(self) -> torch.nn.Module:
        return self.qwen.lm_head

    def path(self, suffix: str) -> str:
        return f"{self.decoder_path}.{suffix}"


def resolve_locateanything_qwen_decoder(model: torch.nn.Module) -> ResolvedQwenDecoder:
    """Find the real decoder once, by module identity rather than PEFT paths.

    Current remote-code/PEFT variants expose the decoder beneath one of the
    paths mentioned in the experiment notes.  Walking ``named_modules`` avoids
    assuming which wrapper depth is active and gives movement and validation
    the *same Python module objects*.
    """
    named = list(model.named_modules())
    paths = {id(module): ("model" if not name else f"model.{name}") for name, module in named}
    decoder_candidates = [
        module
        for _, module in named
        if hasattr(module, "layers")
        and isinstance(getattr(module, "layers"), torch.nn.ModuleList)
        and len(module.layers) == 36
        and hasattr(module, "embed_tokens")
        and hasattr(module, "norm")
    ]
    matches: List[ResolvedQwenDecoder] = []
    for decoder in decoder_candidates:
        qwen_candidates = [
            module
            for _, module in named
            if getattr(module, "model", None) is decoder and hasattr(module, "lm_head")
        ]
        for qwen in qwen_candidates:
            matches.append(
                ResolvedQwenDecoder(
                    qwen=qwen,
                    decoder=decoder,
                    qwen_path=paths[id(qwen)],
                    decoder_path=paths[id(decoder)],
                )
            )
    if len(matches) == 1:
        return matches[0]
    candidate_paths = [paths[id(module)] for module in decoder_candidates]
    match_paths = [f"{item.qwen_path} -> {item.decoder_path}" for item in matches]
    raise RuntimeError(
        "could not uniquely resolve LocateAnything Qwen decoder root; "
        f"decoder_candidates={candidate_paths}, qwen_decoder_matches={match_paths}"
    )


def _resolve_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """Backward-compatible qwen-root accessor for diagnostic callers."""
    return resolve_locateanything_qwen_decoder(model).qwen


def _single_device(module: torch.nn.Module, label: str) -> str:
    devices = {str(parameter.device) for parameter in module.parameters(recurse=True)}
    if not devices:
        return "parameterless"
    if len(devices) != 1:
        raise RuntimeError(f"{label} spans multiple devices: {sorted(devices)}")
    return next(iter(devices))


def _first_parameter_device(module: torch.nn.Module, label: str) -> str:
    try:
        return str(next(module.parameters()).device)
    except StopIteration as exc:
        raise RuntimeError(f"{label} has no parameters to validate") from exc


def _weight_identity(left: torch.nn.Parameter, right: torch.nn.Parameter) -> Dict[str, Any]:
    same_object = left is right
    same_storage = False
    if left.device == right.device:
        try:
            same_storage = (
                left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
            )
        except RuntimeError:
            same_storage = False
    return {
        "same_parameter_object": same_object,
        "same_storage": same_storage,
        "left_device": str(left.device),
        "right_device": str(right.device),
    }


def two_gpu_layout_report(
    resolved: ResolvedQwenDecoder,
    layout: DecoderShardLayout,
    *,
    tied_weight_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Concrete path/device log; safe to emit before validation fails."""
    layers = list(resolved.decoder.layers)
    selected = {
        "decoder_root": (resolved.decoder_path, resolved.decoder),
        "embed_tokens": (resolved.path("embed_tokens"), resolved.embed_tokens),
        "layer_0": (resolved.path("layers.0"), layers[0]),
        "layer_17": (resolved.path("layers.17"), layers[17]),
        "layer_18": (resolved.path("layers.18"), layers[18]),
        "layer_35": (resolved.path("layers.35"), layers[35]),
        "final_norm": (resolved.path("norm"), resolved.norm),
        "lm_head": (f"{resolved.qwen_path}.lm_head", resolved.lm_head),
    }
    modules: Dict[str, Any] = {}
    for label, (path, module) in selected.items():
        devices = sorted({str(parameter.device) for parameter in module.parameters(recurse=True)})
        modules[label] = {
            "path": path,
            "class": f"{module.__class__.__module__}.{module.__class__.__name__}",
            "parameter_devices": devices,
            "first_parameter_device": (
                _first_parameter_device(module, label) if devices else "parameterless"
            ),
        }
    return {
        "resolved_qwen_path": resolved.qwen_path,
        "resolved_decoder_path": resolved.decoder_path,
        "expected_early_device": str(layout.first_device),
        "expected_late_device": str(layout.second_device),
        "modules": modules,
        "tied_weight_policy": tied_weight_policy,
        "original_decoder_forward_signature": getattr(
            resolved.decoder, "_chestxray8_two_gpu_original_forward_signature", None
        ),
    }


def _forward_signature_report(forward) -> Dict[str, Any]:
    signature = inspect.signature(forward)
    return {
        "text": str(signature),
        "parameters": [
            {
                "name": parameter.name,
                "kind": parameter.kind.name,
                "default": None
                if parameter.default is inspect.Parameter.empty
                else repr(parameter.default),
            }
            for parameter in signature.parameters.values()
        ],
        "accepts_var_kwargs": any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ),
    }


def _sharded_argument_handling_report(signature: Dict[str, Any]) -> Dict[str, Any]:
    """Auditable original-vs-sharded contract comparison for probe JSON."""
    original = [item["name"] for item in signature["parameters"]]
    return {
        "original_parameter_names": original,
        "input_ids": "unchanged; required on cuda:0, as original embedding input",
        "attention_mask": "original LocateAnything mask construction on cuda:0; copied only at layer 18",
        "position_ids": "original construction/reshape on cuda:0; copied only at layer 18",
        "past_key_values": "original DynamicCache/legacy conversion and live cache updates unchanged",
        "inputs_embeds": "original branch preserved; required on cuda:0",
        "use_cache": "original default and legacy-cache return behavior preserved",
        "output_attentions": "original layer argument, collection, and return behavior preserved",
        "output_hidden_states": "original collection and return behavior preserved",
        "return_dict": "original tuple/BaseModelOutputWithPast behavior preserved",
        "cache_position": (
            "rejected when non-null because this resolved original decoder does not accept it"
            if "cache_position" not in original
            else "accepted by the sharded wrapper"
        ),
        "visual_features": "passed to original image_processing before layer 0; moved to cuda:0 only in that original-consumed branch",
        "additional_kwargs": "strictly rejected unless present in the inspected original signature; no kwargs are silently dropped",
        "only_math_change": "differentiable hidden-state and non-grad mask/position copies at layer 17->18",
    }


def _cache_layer_records(past_key_values: Any) -> Iterable[Tuple[int, torch.Tensor]]:
    """Yield legacy K/V tensors without copying or changing cache ownership."""
    if not isinstance(past_key_values, (tuple, list)):
        return
    for layer_index, pair in enumerate(past_key_values):
        if not isinstance(pair, (tuple, list)) or len(pair) < 2:
            continue
        for tensor in pair[:2]:
            if isinstance(tensor, torch.Tensor):
                yield layer_index, tensor


def assert_two_gpu_live_cache_layout(
    model: torch.nn.Module,
    layout: DecoderShardLayout,
    *,
    resolved: Optional[ResolvedQwenDecoder] = None,
    tied_weight_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate the physical placement, including injected LoRA modules."""
    resolved = resolved or resolve_locateanything_qwen_decoder(model)
    qwen = resolved.qwen
    decoder = resolved.decoder
    layers = list(decoder.layers)
    if len(layers) != 2 * layout.first_layer_count:
        raise RuntimeError(
            "18/18 shard requires exactly 36 decoder layers; got "
            f"{len(layers)}"
        )
    expected_first, expected_second = str(layout.first_device), str(layout.second_device)
    for index, layer in enumerate(layers):
        expected = expected_first if index < layout.first_layer_count else expected_second
        actual = _single_device(layer, f"decoder layer {index}")
        if actual != expected:
            raise RuntimeError(
                f"decoder layer {index} is on {actual}, expected {expected}"
            )
    # Compare the actual embedding parameter, never the PEFT/wrapper module.
    if _first_parameter_device(resolved.embed_tokens, "embed_tokens") != expected_first:
        raise RuntimeError("embed_tokens must be colocated with early decoder layers")
    if _first_parameter_device(resolved.norm, "decoder norm") != expected_second:
        raise RuntimeError("decoder norm must be colocated with late decoder layers")
    if _first_parameter_device(resolved.lm_head, "lm_head") != expected_second:
        raise RuntimeError("lm_head must be colocated with late decoder layers")

    lora_parameters = [
        (name, parameter)
        for name, parameter in qwen.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    if not lora_parameters:
        raise RuntimeError("LoRA Case B trainability missing after two-GPU placement")
    misplaced_lora: List[str] = []
    for name, parameter in lora_parameters:
        # PEFT preserves the base path (model.layers.N...) in parameter names.
        layer_index = next(
            (index for index in range(len(layers)) if f"layers.{index}." in name),
            None,
        )
        if layer_index is None:
            misplaced_lora.append(f"{name}: cannot infer decoder layer")
            continue
        expected = str(layout.device_for_layer(layer_index))
        if str(parameter.device) != expected:
            misplaced_lora.append(f"{name}: {parameter.device}, expected {expected}")
    if misplaced_lora:
        raise RuntimeError("LoRA/base colocation failure: " + "; ".join(misplaced_lora[:8]))
    report = two_gpu_layout_report(
        resolved,
        layout,
        tied_weight_policy=tied_weight_policy,
    )
    report.update({
        "num_decoder_layers": len(layers),
        "split": [layout.first_layer_count, len(layers) - layout.first_layer_count],
        "early_device": expected_first,
        "late_device": expected_second,
        "lora_trainable_parameter_tensors": len(lora_parameters),
        "lora_trainable_parameter_elements": int(
            sum(parameter.numel() for _, parameter in lora_parameters)
        ),
    })
    return report


def _install_sharded_decoder_forward(
    decoder: torch.nn.Module,
    layout: DecoderShardLayout,
    *,
    original_forward=None,
    original_signature: Optional[Dict[str, Any]] = None,
) -> None:
    """Install the Qwen2 decoder loop with an autograd-preserving boundary."""
    prior = getattr(decoder, "_chestxray8_two_gpu_layout", None)
    requested = (str(layout.first_device), str(layout.second_device), layout.first_layer_count)
    if prior is not None:
        if tuple(prior) != requested:
            raise RuntimeError("decoder is already sharded with a different layout")
        return
    if getattr(decoder, "gradient_checkpointing", False):
        raise RuntimeError(
            "two-GPU live-cache sharding requires gradient_checkpointing=false; "
            "mutable cached replay cannot be checkpointed"
        )
    if getattr(decoder, "_attn_implementation", "sdpa") != "sdpa":
        raise RuntimeError("two-GPU thesis backend requires production SDPA")

    # Imports are intentionally local: the exact remote-code transformers
    # version is resolved at process runtime rather than at module import time.
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    original_forward = original_forward or decoder.forward
    original_signature = original_signature or _forward_signature_report(original_forward)
    original_names = {
        item["name"] for item in original_signature["parameters"] if item["name"] != "self"
    }
    # The pinned remote implementation has precisely this decoder contract.
    # Refuse an unreviewed source drift rather than silently dropping a new
    # remote-code argument while attempting exact replay.
    implemented = {
        "input_ids", "visual_features", "image_token_index", "attention_mask",
        "position_ids", "past_key_values", "inputs_embeds", "use_cache",
        "output_attentions", "output_hidden_states", "return_dict", "cache_position",
    }
    unsupported = sorted(original_names - implemented)
    if unsupported:
        raise RuntimeError(
            "two-GPU decoder contract has unimplemented original arguments: "
            f"{unsupported}; refusing to drop remote-code semantics"
        )
    source_globals = getattr(getattr(original_forward, "__func__", original_forward), "__globals__", {})
    required_source_helpers = (
        "find_prefix_seq_length_by_pe",
        "_prepare_4d_causal_attention_mask",
        "update_causal_mask_for_one_gen_window_2d",
        "update_causal_mask_with_pad_non_visible_2d",
        "create_block_diff_mask_by_pe_4d",
    )
    missing_helpers = [name for name in required_source_helpers if name not in source_globals]
    if missing_helpers:
        raise RuntimeError(
            "could not copy original LocateAnything decoder mask logic; missing "
            f"remote helpers {missing_helpers}"
        )

    first_device, second_device = layout.first_device, layout.second_device
    split = layout.first_layer_count

    # These are the functions used by the original resolved remote-code forward,
    # not reimplementations imported from a different Qwen revision.
    find_prefix_seq_length_by_pe = source_globals["find_prefix_seq_length_by_pe"]
    prepare_4d_causal_attention_mask = source_globals["_prepare_4d_causal_attention_mask"]
    update_causal_mask_for_one_gen_window_2d = source_globals[
        "update_causal_mask_for_one_gen_window_2d"
    ]
    update_causal_mask_with_pad_non_visible_2d = source_globals[
        "update_causal_mask_with_pad_non_visible_2d"
    ]
    create_block_diff_mask_by_pe_4d = source_globals["create_block_diff_mask_by_pe_4d"]

    # The remote Qwen attention indexes RoPE tables as ``cos[position_ids]``.
    # Install a local, synchronous guard around that exact call site so an
    # invalid index is reported before CUDA launches IndexKernel.
    rotary_owner = decoder
    remote_layer_globals = getattr(getattr(decoder.layers[0].self_attn.forward, "__func__", decoder.layers[0].self_attn.forward), "__globals__", {})
    original_apply_rotary = remote_layer_globals.get("apply_rotary_pos_emb")
    if original_apply_rotary is None:
        raise RuntimeError("resolved Qwen attention has no apply_rotary_pos_emb global")
    if not getattr(decoder, "_chestxray8_rotary_guard_installed", False):
        def checked_apply_rotary(q, k, cos, sin, position_ids, unsqueeze_dim=1):
            ctx = dict(getattr(rotary_owner, "_chestxray8_rotary_context", {}) or {})
            # .item() intentionally synchronizes before the indexing kernel.
            pos_min, pos_max = int(position_ids.min().item()), int(position_ids.max().item())
            report = {**ctx, "layer_index": int(ctx.get("layer_index", -1)),
                "immutable_cpu_position_values": ctx.get("immutable_cpu_position_values"),
                "layer_local_position_ids": {"values": position_ids.detach().cpu().tolist(),
                    "shape": list(position_ids.shape), "dtype": str(position_ids.dtype),
                    "device": str(position_ids.device), "min": pos_min, "max": pos_max},
                "cos_shape": list(cos.shape), "sin_shape": list(sin.shape),
                "current_token_sequence_length": int(q.shape[-2]),
                "attention_mask_shape": ctx.get("attention_mask_shape"),
                "past_kv_sequence_length": ctx.get("past_kv_sequence_length"),
                "expected_first_position": ctx.get("expected_first_position"),
                "expected_last_position": ctx.get("expected_last_position"),
                "model_max_position_embeddings": int(rotary_owner.config.max_position_embeddings)}
            if not (pos_min >= 0 and pos_max < cos.shape[0]):
                report["status"] = "invalid_rotary_position_ids"
                destination = ctx.get("diagnostic_json_path")
                if destination:
                    Path(destination).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                raise AssertionError("rotary position-id bounds violation: " + json.dumps(report, sort_keys=True))
            # Keep the explicit invariants adjacent to the original CUDA index.
            assert position_ids.min().item() >= 0
            assert position_ids.max().item() < cos.shape[0]
            if (
                bool(getattr(rotary_owner, "_chestxray8_debug_position_ids", False))
                and not torch.is_grad_enabled()
                and int(ctx.get("layer_index", -1)) == split
            ):
                rotary_numeric = {
                    "query_before_rope": _numeric_tensor_summary(q),
                    "key_before_rope": _numeric_tensor_summary(k),
                    "cos": _numeric_tensor_summary(cos),
                    "sin": _numeric_tensor_summary(sin),
                }
                rotated_q, rotated_k = original_apply_rotary(
                    q, k, cos, sin, position_ids, unsqueeze_dim
                )
                rotary_numeric.update(
                    {
                        "query_after_rope": _numeric_tensor_summary(rotated_q),
                        "key_after_rope": _numeric_tensor_summary(rotated_k),
                    }
                )
                rotary_owner._chestxray8_layer18_rotary_numeric_report = rotary_numeric
                return rotated_q, rotated_k
            return original_apply_rotary(q, k, cos, sin, position_ids, unsqueeze_dim)
        remote_layer_globals["apply_rotary_pos_emb"] = checked_apply_rotary
        decoder._chestxray8_rotary_guard_installed = True

    def forward_sharded(
        self,
        input_ids=None,
        visual_features=None,
        image_token_index=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        if kwargs:
            unknown = sorted(kwargs)
            raise TypeError(
                "sharded decoder received kwargs not accepted by the original "
                f"LocateAnything decoder: {unknown}; original={original_signature['text']}"
            )
        if cache_position is not None and "cache_position" not in original_names:
            raise TypeError(
                "cache_position was provided, but the resolved original decoder "
                "does not accept it"
            )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            if input_ids.device != first_device:
                raise RuntimeError(f"input_ids must start on {first_device}, got {input_ids.device}")
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            if inputs_embeds.device != first_device:
                raise RuntimeError(
                    f"inputs_embeds must start on {first_device}, got {inputs_embeds.device}"
                )

        consumes_visual_before_layer_0 = inputs_embeds is None and visual_features is not None
        visual_report: Dict[str, Any] = {
            "received_kwarg_names": sorted(
                name
                for name, value in {
                    "input_ids": input_ids,
                    "visual_features": visual_features,
                    "image_token_index": image_token_index,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "inputs_embeds": inputs_embeds,
                    "use_cache": use_cache,
                    "output_attentions": output_attentions,
                    "output_hidden_states": output_hidden_states,
                    "return_dict": return_dict,
                    "cache_position": cache_position,
                }.items()
                if value is not None
            ),
            "visual_features": None,
            "visual_features_target_device": str(first_device),
            "visual_features_consumed": (
                "before_layer_0_via_image_processing"
                if consumes_visual_before_layer_0
                else "not_consumed_by_original_decoder_for_this_call"
            ),
            "extra_kwargs": [],
        }
        if isinstance(visual_features, torch.Tensor):
            visual_report["visual_features"] = {
                "shape": list(visual_features.shape),
                "dtype": str(visual_features.dtype),
                "received_device": str(visual_features.device),
            }
            # The original consumes visual features only through
            # ``image_processing`` when inputs_embeds is absent.  Preserve that
            # branch exactly; its move is differentiable and a same-device no-op
            # in the production replay.
            if consumes_visual_before_layer_0:
                visual_features = visual_features.to(first_device, non_blocking=True)
            visual_report["visual_features"]["moved_device"] = str(visual_features.device)
        self._chestxray8_two_gpu_last_argument_report = visual_report
        argument_history = getattr(self, "_chestxray8_two_gpu_argument_reports", None)
        if argument_history is None:
            argument_history = []
            self._chestxray8_two_gpu_argument_reports = argument_history
        argument_history.append(visual_report)

        use_legacy_cache = False
        past_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = (
                    DynamicCache()
                    if past_key_values is None
                    else DynamicCache.from_legacy_cache(past_key_values)
                )
            past_length = int(past_key_values.get_seq_length())

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + seq_length,
                dtype=torch.long,
                device="cpu",
            ).unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        position_ids_cpu, early_position_ids, late_position_ids = materialize_shard_position_ids(
            position_ids, first_device, second_device
        )
        entry_context = dict(getattr(self, "_chestxray8_rotary_context", {}) or {})
        forward_call_id = int(getattr(self, "_chestxray8_forward_call_count", 0)) + 1
        self._chestxray8_forward_call_count = forward_call_id
        uses_pbd_position_pattern = str(entry_context.get("branch", "")).startswith("PBD")
        canonical_ar_cpu = torch.arange(
            past_length, past_length + seq_length, dtype=torch.long, device="cpu"
        ).unsqueeze(0)
        if bool(getattr(self, "_chestxray8_debug_position_ids", False)):
            assert torch.equal(early_position_ids.detach().cpu(), position_ids_cpu)
            assert torch.equal(late_position_ids.detach().cpu(), position_ids_cpu)
            if not uses_pbd_position_pattern and entry_context.get("branch") in {"NTP fallback", "AR/NTP"}:
                if not torch.equal(position_ids_cpu, canonical_ar_cpu):
                    raise AssertionError(
                        f"AR/NTP forward-entry positions differ from {past_length}..{past_length + seq_length - 1}"
                    )
        # Original mask/image logic executes on the early shard.
        position_ids = early_position_ids
        if inputs_embeds is None:
            # Exact original LocateAnything visual injection.  The ChestX-ray8
            # safe-image patch replaces only this method's implementation and
            # keeps the same before-layer-0 contract.
            inputs_embeds = self.image_processing(
                input_ids, visual_features, image_token_index
            )

        if attention_mask is not None and self._attn_implementation == "magi" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                )
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        x0_len = find_prefix_seq_length_by_pe(position_ids).to(device=device)

        def prepare_block_mask_for_inference(mask):
            mask = prepare_4d_causal_attention_mask(
                mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_length,
                sliding_window=self.config.sliding_window,
            )
            if seq_length == 1 or (
                input_ids is not None and input_ids[0][-1].item() != self.text_mask_token_id
            ):
                return mask
            if mask is None or len(mask.shape) != 4:
                return mask
            if use_cache:
                update_mask = lambda ids, old: update_causal_mask_for_one_gen_window_2d(
                    ids, old, block_size=self.block_size, use_cache=use_cache,
                    causal_attn=self.causal_attn,
                )
            else:
                update_mask = lambda ids, old: update_causal_mask_with_pad_non_visible_2d(
                    ids, old, block_size=self.block_size,
                    text_mask_token_id=self.text_mask_token_id,
                    causal_attn=self.causal_attn,
                )
            return torch.stack(
                [update_mask(input_ids[b], mask[b][0]).unsqueeze(0) for b in range(mask.shape[0])],
                dim=0,
            )

        def prepare_block_mask_for_training():
            block_mask, _ = create_block_diff_mask_by_pe_4d(
                block_size=self.block_size,
                x0_len_list=x0_len,
                position_ids=position_ids,
                causal_attn=self.causal_attn,
            )
            return block_mask

        if self._attn_implementation == "magi":
            # This branch remains in the copied contract, although the thesis
            # backend validates SDPA before installing the patch.
            build_magi_ranges = source_globals["build_magi_ranges"]
            ar_decode = seq_length == 1 or (
                input_ids is not None and input_ids[0][-1].item() != self.text_mask_token_id
            )
            attention_mask = build_magi_ranges(
                kv_len=seq_length + past_length, q_len=seq_length,
                block_size=self.block_size, ar_decode=ar_decode, device=device,
            )
        elif self._attn_implementation == "sdpa":
            attention_mask = (
                prepare_block_mask_for_training()
                if self.training
                else prepare_block_mask_for_inference(attention_mask)
            )
        else:
            raise NotImplementedError(f"{self._attn_implementation=}")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.device != first_device:
            attention_mask = attention_mask.to(first_device, non_blocking=True)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        layer_attention_mask = attention_mask
        layer_position_ids = early_position_ids
        hidden_finite_report = []
        numeric_rollout_debug = bool(
            getattr(self, "_chestxray8_debug_position_ids", False)
        ) and not torch.is_grad_enabled()
        if numeric_rollout_debug:
            hidden_finite_report.append({"stage": "before_layer_0", "device": str(hidden_states.device),
                                         "all_finite": bool(torch.isfinite(hidden_states).all().item())})
        for index, decoder_layer in enumerate(self.layers):
            if index == split:
                # Crucially, this custom identity-Jacobian operation is the sole
                # cross-device activation transport. Its internal CPU value is
                # transient and is not retained with the autograd graph.
                boundary_event = {
                    "layer_boundary": f"{split - 1}->{split}",
                    "hidden_before_device": str(hidden_states.device),
                    "hidden_before_requires_grad": bool(hidden_states.requires_grad),
                    "hidden_before_grad_fn": (
                        type(hidden_states.grad_fn).__name__
                        if hidden_states.grad_fn is not None
                        else None
                    ),
                }
                debug_boundary = numeric_rollout_debug
                hidden_cpu_before_transfer = (
                    hidden_states.detach().cpu().clone() if debug_boundary else None
                )
                mask_cpu_before_transfer = (
                    layer_attention_mask.detach().cpu().clone()
                    if debug_boundary and isinstance(layer_attention_mask, torch.Tensor)
                    else None
                )
                hidden_states = _copy_across_shards(hidden_states, second_device)
                boundary_event.update(
                    {
                        "hidden_after_device": str(hidden_states.device),
                        "hidden_after_requires_grad": bool(hidden_states.requires_grad),
                        "hidden_after_grad_fn": (
                            type(hidden_states.grad_fn).__name__
                            if hidden_states.grad_fn is not None
                            else None
                        ),
                    }
                )
                boundary_history = getattr(
                    self, "_chestxray8_two_gpu_boundary_events", None
                )
                if boundary_history is None:
                    boundary_history = []
                    self._chestxray8_two_gpu_boundary_events = boundary_history
                boundary_history.append(boundary_event)
                if isinstance(layer_attention_mask, torch.Tensor):
                    layer_attention_mask = _copy_across_shards(
                        layer_attention_mask, second_device
                    )
                if debug_boundary:
                    torch.cuda.synchronize(second_device)
                    boundary_event["hidden_after_transfer_numeric"] = _numeric_tensor_summary(hidden_states)
                    boundary_event["hidden_transfer_value_preserved"] = bool(
                        torch.equal(hidden_states.detach().cpu(), hidden_cpu_before_transfer)
                    )
                    if isinstance(layer_attention_mask, torch.Tensor):
                        boundary_event["attention_mask_after_transfer_numeric"] = _numeric_tensor_summary(layer_attention_mask)
                        boundary_event["attention_mask_transfer_value_preserved"] = bool(
                            torch.equal(layer_attention_mask.detach().cpu(), mask_cpu_before_transfer)
                        )
                # Do not transfer or reuse early-shard position storage.  The
                # late shard owns its independent copy from forward entry.
                layer_position_ids = late_position_ids
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_past_length = 0
            if use_cache:
                layer_past_length = int(past_key_values.get_seq_length(index))
            context = dict(getattr(self, "_chestxray8_rotary_context", {}) or {})
            context.update({"layer_index": index, "attention_mask_shape": list(layer_attention_mask.shape) if isinstance(layer_attention_mask, torch.Tensor) else None,
                            "forward_call_id": forward_call_id,
                            "past_kv_sequence_length": past_length,
                            "layer_observed_past_kv_sequence_length": layer_past_length,
                            "expected_first_position": past_length,
                            "expected_last_position": past_length + seq_length - 1,
                            "immutable_cpu_position_values": position_ids_cpu.tolist()})
            self._chestxray8_rotary_context = context
            if bool(getattr(self, "_chestxray8_debug_position_ids", False)):
                expected_local = early_position_ids if index < split else late_position_ids
                assert layer_position_ids.data_ptr() == expected_local.data_ptr()
                assert torch.equal(layer_position_ids.detach().cpu(), position_ids_cpu)
            diagnostic_handles = []
            layer18_numeric: Optional[Dict[str, Any]] = None
            if numeric_rollout_debug and index == split:
                layer18_numeric = {
                    "hidden_input": _numeric_tensor_summary(hidden_states),
                    "attention_mask": _numeric_tensor_summary(layer_attention_mask)
                    if isinstance(layer_attention_mask, torch.Tensor)
                    else None,
                    "position_ids": _numeric_tensor_summary(layer_position_ids),
                    "past_key": _numeric_tensor_summary(past_key_values[index][0])
                    if use_cache and past_key_values.get_seq_length(index) > 0
                    else None,
                    "past_value": _numeric_tensor_summary(past_key_values[index][1])
                    if use_cache and past_key_values.get_seq_length(index) > 0
                    else None,
                }

                def capture_output(name):
                    def hook(_module, _inputs, output):
                        tensor = output[0] if isinstance(output, (tuple, list)) else output
                        if isinstance(tensor, torch.Tensor):
                            layer18_numeric[name] = _numeric_tensor_summary(tensor)
                    return hook

                def capture_input(name):
                    def hook(_module, inputs):
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            layer18_numeric[name] = _numeric_tensor_summary(inputs[0])
                    return hook

                for name, module in (
                    ("after_input_layernorm", decoder_layer.input_layernorm),
                    ("q_projection", decoder_layer.self_attn.q_proj),
                    ("k_projection", decoder_layer.self_attn.k_proj),
                    ("v_projection", decoder_layer.self_attn.v_proj),
                    ("self_attention_output", decoder_layer.self_attn),
                    ("after_post_attention_layernorm", decoder_layer.post_attention_layernorm),
                    ("mlp_output", decoder_layer.mlp),
                ):
                    diagnostic_handles.append(module.register_forward_hook(capture_output(name)))
                diagnostic_handles.append(
                    decoder_layer.self_attn.o_proj.register_forward_pre_hook(
                        capture_input("sdpa_output_before_o_projection")
                    )
                )
            try:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=layer_position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            finally:
                for handle in diagnostic_handles:
                    handle.remove()
            if layer18_numeric is not None:
                layer18_numeric["rotary"] = getattr(
                    self, "_chestxray8_layer18_rotary_numeric_report", None
                )
                self._chestxray8_layer18_numeric_report = layer18_numeric
            hidden_states = layer_outputs[0]
            if numeric_rollout_debug:
                hidden_finite_report.append(
                    {
                        "stage": f"after_layer_{index}",
                        "layer_index": index,
                        **_numeric_tensor_summary(hidden_states),
                    }
                )
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if numeric_rollout_debug:
            hidden_finite_report.append({"stage": "after_final_norm_before_lm_head", "device": str(hidden_states.device),
                "all_finite": bool(torch.isfinite(hidden_states).all().item()),
                "nan_count": int(torch.isnan(hidden_states).sum().item()),
                "positive_inf_count": int(torch.isposinf(hidden_states).sum().item()),
                "negative_inf_count": int(torch.isneginf(hidden_states).sum().item())})
            self._chestxray8_last_hidden_finite_report = {
                "forward_call_id": forward_call_id, "layers": hidden_finite_report,
                "first_nonfinite_stage": next((item["stage"] for item in hidden_finite_report if not item["all_finite"]), None)}
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )
        if not return_dict:
            return tuple(
                value
                for value in (hidden_states, next_cache, all_hidden_states, all_self_attns)
                if value is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    decoder._chestxray8_two_gpu_original_forward = original_forward
    decoder._chestxray8_two_gpu_original_forward_signature = original_signature
    decoder.forward = MethodType(forward_sharded, decoder)
    decoder._chestxray8_two_gpu_layout = requested


def shard_locateanything_decoder_two_gpu(
    model: torch.nn.Module,
    *,
    first_device: str | torch.device = "cuda:0",
    second_device: str | torch.device = "cuda:1",
    first_layer_count: int = 18,
    on_pre_forward_layout: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Place LocateAnything's decoder 18/18 and patch its exact live-cache loop.

    Call this after LoRA injection.  Moving a decoder layer recursively moves
    its PEFT LoRA modules with the frozen base projections, guaranteeing
    colocated forward/backward execution.
    """
    layout = DecoderShardLayout(
        first_device=torch.device(first_device),
        second_device=torch.device(second_device),
        first_layer_count=int(first_layer_count),
    )
    if layout.first_device.type != "cuda" or layout.second_device.type != "cuda":
        raise ValueError("two-GPU live-cache sharding requires two CUDA devices")
    if layout.first_device == layout.second_device:
        raise ValueError("two-GPU live-cache sharding requires distinct CUDA devices")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices are required (CUDA_VISIBLE_DEVICES=2,3)")
    # Resolve once.  Every move, tied-weight decision, validation and forward
    # patch below uses these exact objects; wrapper depth is never re-inferred.
    resolved = resolve_locateanything_qwen_decoder(model)
    qwen = resolved.qwen
    decoder = resolved.decoder
    layers = list(decoder.layers)
    original_decoder_forward = decoder.forward
    original_decoder_signature = _forward_signature_report(original_decoder_forward)
    if len(layers) != 36 or layout.first_layer_count != 18:
        raise RuntimeError(
            "LocateAnything-3B thesis layout is fixed to 36 layers split 18/18; "
            f"got {len(layers)} layers and split {layout.first_layer_count}"
        )

    # The caller initially loads everything on cuda:0.  Vision/projector remain
    # there, so image injection and the first embedding lookup need no transfer.
    migration_reports: List[Dict[str, Any]] = []
    # The loader contract places the complete model on the first shard.  Do
    # not perform (or pay to hash) same-device no-op moves for embeddings and
    # layers 0..17; validate their placement below.  Every tensor that actually
    # crosses to the late shard is value-checked here before forward install.
    for index, layer in enumerate(layers[layout.first_layer_count :], start=layout.first_layer_count):
        migration_reports.append(
            _move_module_initialization_exact(
                layer,
                layout.second_device,
                label=resolved.path(f"layers.{index}"),
            )
        )
    migration_reports.append(
        _move_module_initialization_exact(
            resolved.norm,
            layout.second_device,
            label=resolved.path("norm"),
        )
    )

    tie_before = _weight_identity(resolved.lm_head.weight, resolved.embed_tokens.weight)
    if tie_before["same_parameter_object"] or tie_before["same_storage"]:
        # A parameter cannot be tied across CUDA devices.  The base lm_head is
        # frozen in Case B, so a value-identical independent clone is exact for
        # the probe and preserves the desired late-device logits placement.
        source = resolved.lm_head.weight.detach()
        cloned_weight, clone_report = _copy_initialization_value_exact(
            source,
            layout.second_device,
            label=f"{resolved.qwen_path}.lm_head.weight_tied_clone",
        )
        resolved.lm_head.weight = torch.nn.Parameter(
            cloned_weight.clone(),
            requires_grad=bool(resolved.lm_head.weight.requires_grad),
        )
        migration_reports.append(
            {
                "label": f"{resolved.qwen_path}.lm_head_tied_clone",
                "target_device": str(layout.second_device),
                "tensor_count": 1,
                "bf16_peer_tensor_count": int(
                    clone_report["transport"]
                    == "bf16_to_fp32_cpu_stage_to_target_to_bf16_synchronous"
                ),
                "all_exact": bool(clone_report["exact_value_equality"]),
                "tensors": [clone_report],
            }
        )
        tied_weight_policy: Dict[str, Any] = {
            "policy": "untie_lm_head_clone_on_late_device",
            "before": tie_before,
            "reason": "tied embedding/lm_head storage cannot span cuda:0 and cuda:1",
        }
    else:
        tied_weight_policy = {
            "policy": "independent_lm_head_on_late_device",
            "before": tie_before,
        }
    if not (tie_before["same_parameter_object"] or tie_before["same_storage"]):
        migration_reports.append(
            _move_module_initialization_exact(
                resolved.lm_head,
                layout.second_device,
                label=f"{resolved.qwen_path}.lm_head",
            )
        )
    tied_weight_policy["after"] = _weight_identity(
        resolved.lm_head.weight, resolved.embed_tokens.weight
    )
    embed_rows = int(resolved.embed_tokens.weight.shape[0])
    lm_head_rows = int(resolved.lm_head.weight.shape[0])
    configured_vocab = int(getattr(resolved.qwen.config, "vocab_size", embed_rows))
    vocabulary_invariants = {
        "embedding_rows": embed_rows,
        "lm_head_rows": lm_head_rows,
        "configured_vocab_size": configured_vocab,
        "embedding_lm_head_shape_equal": bool(
            resolved.embed_tokens.weight.shape == resolved.lm_head.weight.shape
        ),
        "all_equal": embed_rows == lm_head_rows == configured_vocab,
    }
    if not vocabulary_invariants["all_equal"]:
        raise RuntimeError(
            "embedding/lm-head/config vocabulary invariant failed: "
            + json.dumps(vocabulary_invariants, sort_keys=True)
        )
    lora_state = [
        (name, parameter)
        for name, parameter in qwen.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]
    lora_invariants = {
        "tensor_count": len(lora_state),
        "lora_a_tensor_count": sum("lora_A" in name for name, _ in lora_state),
        "lora_b_tensor_count": sum("lora_B" in name for name, _ in lora_state),
        "all_finite": all(bool(torch.isfinite(value).all().item()) for _, value in lora_state),
        "all_lora_b_exactly_zero": all(
            bool(torch.count_nonzero(value).item() == 0)
            for name, value in lora_state
            if "lora_B" in name
        ),
        "dtypes": sorted({str(value.dtype) for _, value in lora_state}),
    }
    if not (
        lora_invariants["tensor_count"] == 504
        and lora_invariants["lora_a_tensor_count"] == 252
        and lora_invariants["lora_b_tensor_count"] == 252
        and lora_invariants["all_finite"]
        and lora_invariants["all_lora_b_exactly_zero"]
    ):
        raise RuntimeError(
            "LoRA initialization invariant failed: "
            + json.dumps(lora_invariants, sort_keys=True)
        )

    # This callback runs after all explicit moves and immediately before the
    # validator/forward patch.  The feasibility JSON therefore includes exact
    # paths/devices even when validation raises.
    pre_forward_report = two_gpu_layout_report(
        resolved,
        layout,
        tied_weight_policy=tied_weight_policy,
    )
    pre_forward_report["original_decoder_forward_signature"] = original_decoder_signature
    pre_forward_report["decoder_argument_contract"] = _sharded_argument_handling_report(
        original_decoder_signature
    )
    pre_forward_report["initialization_migration"] = {
        "transport_policy": "BF16->FP32->CPU stage->target->BF16; synchronous; no autograd",
        "module_count": len(migration_reports),
        "tensor_count": sum(item["tensor_count"] for item in migration_reports),
        "bf16_peer_tensor_count": sum(
            item["bf16_peer_tensor_count"] for item in migration_reports
        ),
        "all_exact": all(item["all_exact"] for item in migration_reports),
        "modules": migration_reports,
    }
    pre_forward_report["vocabulary_invariants"] = vocabulary_invariants
    pre_forward_report["lora_initialization_invariants"] = lora_invariants
    if on_pre_forward_layout is not None:
        on_pre_forward_layout(pre_forward_report)
    if (
        tied_weight_policy["after"]["same_parameter_object"]
        or tied_weight_policy["after"]["same_storage"]
    ):
        raise RuntimeError("lm_head and embed_tokens remain tied after two-device placement")
    _install_sharded_decoder_forward(
        decoder,
        layout,
        original_forward=original_decoder_forward,
        original_signature=original_decoder_signature,
    )
    report = assert_two_gpu_live_cache_layout(
        model,
        layout,
        resolved=resolved,
        tied_weight_policy=tied_weight_policy,
    )
    report["decoder_argument_contract"] = _sharded_argument_handling_report(
        original_decoder_signature
    )
    report["initialization_migration"] = pre_forward_report[
        "initialization_migration"
    ]
    report["vocabulary_invariants"] = vocabulary_invariants
    report["lora_initialization_invariants"] = lora_invariants
    report.update(
        {
            "backend": "live_production_cached_two_gpu_shard",
            "carry_cache": True,
            "detached_kv": False,
            "truncated_bptt": False,
            "bfix": False,
            "attn_implementation": "sdpa",
            "cross_device_boundary": "decoder_layers.17_to_18",
        }
    )
    return report


def inspect_legacy_cache_devices(
    past_key_values: Any,
    layout: DecoderShardLayout,
) -> Dict[str, Any]:
    """Diagnostics only: render every returned legacy K/V pair directly."""
    rendered: List[Dict[str, Any]] = []
    if not isinstance(past_key_values, (tuple, list)):
        return {
            "cache_type": type(past_key_values).__name__,
            "layers": rendered,
            "all_layers_live_and_colocated": False,
        }
    for layer_index, pair in enumerate(past_key_values):
        if not isinstance(pair, (tuple, list)) or len(pair) < 2:
            rendered.append(
                {
                    "layer_index": layer_index,
                    "malformed_cache_entry": True,
                }
            )
            continue
        key, value = pair[0], pair[1]
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            rendered.append(
                {
                    "layer_index": layer_index,
                    "malformed_cache_entry": True,
                }
            )
            continue
        expected = str(layout.device_for_layer(layer_index))
        rendered.append(
            {
                "layer_index": layer_index,
                "expected_owner_device": expected,
                "key_device": str(key.device),
                "value_device": str(value.device),
                "key_requires_grad": bool(key.requires_grad),
                "value_requires_grad": bool(value.requires_grad),
                "key_grad_fn": type(key.grad_fn).__name__ if key.grad_fn is not None else None,
                "value_grad_fn": type(value.grad_fn).__name__ if value.grad_fn is not None else None,
                "sequence_length": int(key.size(2)),
                "key_value_shape_match": list(key.shape) == list(value.shape),
                "storage_device_consistent": (
                    str(key.device) == expected and str(value.device) == expected
                ),
            }
        )
    all_live = len(rendered) == 36 and all(
        item.get("storage_device_consistent", False)
        and item.get("key_requires_grad", False)
        and item.get("value_requires_grad", False)
        and item.get("key_grad_fn") is not None
        and item.get("value_grad_fn") is not None
        for item in rendered
    )
    return {
        "cache_type": type(past_key_values).__name__,
        "layers": rendered,
        "all_layers_live_and_colocated": all_live,
    }
