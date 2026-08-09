"""Priority-B selective saved-tensor CPU offload (exact autograd links).

Uses ``torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook)`` to
offload large activation storages while:
  - never detaching tensors,
  - never calling ``contiguous()`` on unpack by default,
  - protecting attention mask/bias only when confirmed by attention-module
    context + tensor semantics / argument identity (not bare stride rules),
  - optionally deduplicating overlapping views (OFF by default: CUDA storage
    pointers can be reused after free; each save event gets a unique save_id),
  - releasing restored GPU storages when unpack products become unreachable.

Diagnostic / investigation only. Not a thesis training backend.
"""

from __future__ import annotations

import hashlib
import math
import threading
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn


_TLS = threading.local()

# Identity-guard levels for short-fixture A/B oracle diagnostics.
IDENTITY_GUARD_LEVELS = ("G0", "G1", "G2", "G3")

# Diagnostic thresholds for prediction tables (bytes).
DEFAULT_THRESHOLD_SWEEP = (1 << 20, 512 << 10, 256 << 10)

_ATTN_NAME_TOKENS = (
    "self_attn",
    "attention",
    "attn",
    "sdpa",
)
_ATTN_ARG_NAME_TOKENS = (
    "attention_mask",
    "attn_mask",
    "attn_bias",
    "attention_bias",
    "causal_mask",
    "key_padding_mask",
)


def _tls_set(name: str, default_factory):
    value = getattr(_TLS, name, None)
    if value is None:
        value = default_factory()
        setattr(_TLS, name, value)
    return value


def _current_module_context() -> Optional[str]:
    stack = getattr(_TLS, "module_stack", None)
    if not stack:
        return None
    return stack[-1]


def push_module_context(name: str) -> None:
    stack = getattr(_TLS, "module_stack", None)
    if stack is None:
        _TLS.module_stack = [name]
    else:
        stack.append(name)


def pop_module_context() -> None:
    stack = getattr(_TLS, "module_stack", None)
    if stack:
        stack.pop()


def is_attention_module_context(context: Optional[str]) -> bool:
    if not context:
        return False
    ctx = context.lower()
    # Prefer path segments so "latent" alone does not match.
    parts = ctx.replace("-", "_").split(".")
    joined = ".".join(parts)
    if any(tok in parts for tok in ("self_attn", "attention", "sdpa")):
        return True
    if any(f".{tok}." in f".{joined}." for tok in ("self_attn", "attention", "sdpa")):
        return True
    # Leaf names like "q_proj" live under self_attn.*; require attn token in path.
    return any(tok in joined for tok in ("self_attn", "attention", "sdpa"))


def tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().nbytes())


def tensor_logical_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def storage_data_ptr(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def matches_old_stride1_only_rule(tensor: torch.Tensor) -> bool:
    """Legacy over-broad rule (B1 failure): protect whenever stride(1) % 4 != 0."""
    if not torch.is_tensor(tensor) or tensor.ndim < 2:
        return False
    return int(tensor.stride(1)) % 4 != 0


def register_confirmed_attn_tensor(tensor: Any) -> None:
    """Record argument identity / storage identity for attention mask/bias."""
    if not torch.is_tensor(tensor):
        return
    ids: Set[int] = _tls_set("protected_tensor_ids", set)
    ptrs: Set[int] = _tls_set("protected_storage_ptrs", set)
    ids.add(id(tensor))
    try:
        ptrs.add(storage_data_ptr(tensor))
    except Exception:
        pass


def _clear_confirmed_attn_registry() -> None:
    _TLS.protected_tensor_ids = set()
    _TLS.protected_storage_ptrs = set()


def _has_qk_mask_bias_shape(tensor: torch.Tensor) -> Tuple[bool, str]:
    """Return whether shape looks like attention mask/bias (not KV head-dim)."""
    if tensor.ndim == 4:
        _b, h, q, k = (int(x) for x in tensor.shape)
        # Bias/mask: (B,H,Q,K) with sequence-like K. KV/activations: (B,H,S,D) D<=256.
        if h <= 128 and k > 256 and 1 <= q <= 16384:
            return True, "float4d_or_bool4d_qk"
        return False, "not_qk_4d"
    if tensor.ndim == 3:
        _b, q, k = (int(x) for x in tensor.shape)
        if q > 1 and k > 256 and q <= 16384:
            return True, "qk_3d"
        return False, "not_qk_3d"
    if tensor.ndim == 2:
        q, k = (int(x) for x in tensor.shape)
        if q >= 1 and k > 256:
            return True, "qk_2d"
        return False, "not_qk_2d"
    return False, "rank_unsupported"


def classify_attention_protection(
    tensor: torch.Tensor,
    *,
    module_context: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Confirm attention mask/bias protection.

    Returns ``(protect, reason, kind)`` where ``kind`` is
    ``confirmed_attention_bias``, ``confirmed_attention_mask``, or ``""``.

    ``stride1_not_multiple_of_4`` is **never** a standalone protect reason.
    """
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return False, "empty_or_non_tensor", ""

    ctx = module_context if module_context is not None else _current_module_context()
    in_attn = is_attention_module_context(ctx)

    ids = getattr(_TLS, "protected_tensor_ids", None)
    ptrs = getattr(_TLS, "protected_storage_ptrs", None)
    if not isinstance(ids, set):
        ids = set()
    if not isinstance(ptrs, set):
        ptrs = set()
    identity_hit = id(tensor) in ids
    try:
        storage_hit = storage_data_ptr(tensor) in ptrs
    except Exception:
        storage_hit = False

    shape_ok, shape_reason = _has_qk_mask_bias_shape(tensor)
    is_bool = tensor.dtype == torch.bool
    is_float = bool(tensor.dtype.is_floating_point)

    if identity_hit or storage_hit:
        if not (is_bool or is_float):
            return False, "identity_hit_but_bad_dtype", ""
        kind = "confirmed_attention_mask" if is_bool else "confirmed_attention_bias"
        how = "argument_identity" if identity_hit else "storage_identity"
        return True, f"{how}+{shape_reason}", kind

    if not in_attn:
        return False, "not_attention_module_context", ""

    if not (is_bool or is_float):
        return False, "attention_context_bad_dtype", ""

    if is_bool and tensor.ndim >= 2 and shape_ok:
        return True, f"attention_context+bool_mask+{shape_reason}", "confirmed_attention_mask"
    if is_bool and tensor.ndim >= 2 and identity_hit:
        return True, "attention_context+bool_mask+identity", "confirmed_attention_mask"
    # Bool padding masks may be (B, K) with K>256 already covered; also (B,1,1,K).
    if is_bool and tensor.ndim >= 2 and int(tensor.size(-1)) > 256:
        return True, "attention_context+bool_mask+seq_k", "confirmed_attention_mask"

    if is_float and shape_ok:
        return True, f"attention_context+float_bias+{shape_reason}", "confirmed_attention_bias"

    return False, "attention_context_but_not_mask_bias_semantics", ""


def looks_like_attention_bias_or_mask(tensor: torch.Tensor) -> Tuple[bool, str]:
    """Backward-compatible wrapper; no bare stride1 protection."""
    protect, reason, _kind = classify_attention_protection(tensor)
    return protect, reason


def describe_tensor(
    tensor: torch.Tensor,
    *,
    category: str,
    offloaded: bool,
    module_context: Optional[str],
    shared_storage_id: Optional[int] = None,
    first_seen_storage: Optional[bool] = None,
    protection_kind: Optional[str] = None,
    old_stride1_rule: Optional[bool] = None,
) -> Dict[str, Any]:
    storage_nbytes = tensor_storage_nbytes(tensor)
    logical_nbytes = tensor_logical_nbytes(tensor)
    strides = tuple(int(s) for s in tensor.stride())
    stride1_aligned = True
    if tensor.ndim >= 2:
        stride1_aligned = (int(strides[1]) % 4) == 0
    return {
        "category": category,
        "protection_kind": protection_kind,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "stride": list(strides),
        "storage_offset": int(tensor.storage_offset()),
        "storage_nbytes": storage_nbytes,
        "logical_nbytes": logical_nbytes,
        "numel": int(tensor.numel()),
        "device": str(tensor.device),
        "is_contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
        "offloaded": bool(offloaded),
        "module_context": module_context,
        "shared_storage_id": shared_storage_id,
        "first_seen_storage": first_seen_storage,
        "stride1_multiple_of_4": stride1_aligned,
        "old_stride1_only_rule": (
            matches_old_stride1_only_rule(tensor)
            if old_stride1_rule is None
            else bool(old_stride1_rule)
        ),
    }


def linear_to_multi_index(lin: int, shape: Sequence[int]) -> Tuple[int, ...]:
    coords: List[int] = []
    remaining = int(lin)
    for size in reversed(tuple(int(s) for s in shape)):
        if size <= 0:
            coords.append(0)
            continue
        coords.append(int(remaining % size))
        remaining //= int(size)
    return tuple(reversed(coords))


def deterministic_sample_lin_indices(numel: int, max_samples: int = 8) -> Tuple[int, ...]:
    n = int(numel)
    if n <= 0:
        return ()
    if n <= max_samples:
        return tuple(range(n))
    # Ends + midpoints + uniform stride — no tensor allocation.
    picks = {0, 1, max(0, n // 4), max(0, n // 2), max(0, (3 * n) // 4), n - 2, n - 1}
    step = max(1, n // max_samples)
    picks.update(range(0, n, step))
    ordered = sorted(i for i in picks if 0 <= i < n)
    return tuple(ordered[:max_samples])


def sample_tensor_values(
    tensor: torch.Tensor,
    *,
    max_samples: int = 8,
    lin_indices: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[float, ...], Tuple[int, ...], Tuple[bool, ...]]:
    """Sample values via multi-index ``.item()`` only (no flatten / large temps)."""
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return (), (), ()
    shape = tuple(int(s) for s in tensor.shape)
    if lin_indices is None:
        lin_indices = deterministic_sample_lin_indices(int(tensor.numel()), max_samples)
    values: List[float] = []
    finite_flags: List[bool] = []
    used: List[int] = []
    # Detach for reads only; does not detach the live autograd tensor returned to PyTorch.
    src = tensor.detach()
    for lin in lin_indices:
        coords = linear_to_multi_index(int(lin), shape)
        # Scalar read — allocation-free for the purpose of verification.
        scalar = src[coords]
        if scalar.dtype == torch.bool:
            value = float(scalar.item())
        else:
            value = float(scalar.float().item())
        values.append(value)
        finite_flags.append(bool(math.isfinite(value)))
        used.append(int(lin))
    return tuple(values), tuple(used), tuple(finite_flags)


def cuda_storage_device_index(target: torch.device) -> int:
    """Convert ``torch.device('cuda:N')`` metadata to an int for ``Storage.cuda``."""
    if not isinstance(target, torch.device):
        raise TypeError(f"expected torch.device, got {type(target)!r}")
    if target.type != "cuda":
        raise ValueError(f"expected cuda device, got {target!r}")
    device_index = target.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return int(device_index)


def predict_threshold_outcomes(
    pack_catalog: Sequence[Dict[str, Any]],
    thresholds: Sequence[int] = DEFAULT_THRESHOLD_SWEEP,
) -> List[Dict[str, Any]]:
    """Predict unique offload / protected GPU bytes for each threshold."""
    rows: List[Dict[str, Any]] = []
    # Unique storages: key -> record
    storages: Dict[str, Dict[str, Any]] = {}
    for entry in pack_catalog:
        key = str(entry.get("storage_key"))
        if key not in storages:
            storages[key] = entry
    for threshold in thresholds:
        unique_offload = 0
        unique_offload_bytes = 0
        protected_bias = 0
        protected_bias_bytes = 0
        protected_mask = 0
        protected_mask_bytes = 0
        below = 0
        below_bytes = 0
        other = 0
        other_bytes = 0
        for entry in storages.values():
            nbytes = int(entry.get("storage_nbytes") or 0)
            kind = entry.get("protection_kind") or ""
            device_type = str(entry.get("device_type") or "")
            if device_type and device_type != "cuda":
                other += 1
                other_bytes += nbytes
                continue
            if kind == "confirmed_attention_bias":
                protected_bias += 1
                protected_bias_bytes += nbytes
            elif kind == "confirmed_attention_mask":
                protected_mask += 1
                protected_mask_bytes += nbytes
            elif nbytes >= int(threshold):
                unique_offload += 1
                unique_offload_bytes += nbytes
            else:
                below += 1
                below_bytes += nbytes
        protected_gpu_bytes = (
            protected_bias_bytes + protected_mask_bytes + below_bytes + other_bytes
        )
        rows.append(
            {
                "threshold_bytes": int(threshold),
                "predicted_unique_offloaded_storages": unique_offload,
                "predicted_unique_offloaded_bytes": unique_offload_bytes,
                "predicted_protected_gpu_bytes": protected_gpu_bytes,
                "predicted_protected_attention_bias_bytes": protected_bias_bytes,
                "predicted_protected_attention_mask_bytes": protected_mask_bytes,
                "predicted_protected_below_threshold_bytes": below_bytes,
                "predicted_other_resident_bytes": other_bytes,
                "predicted_protected_attention_bias_storages": protected_bias,
                "predicted_protected_attention_mask_storages": protected_mask,
                "predicted_below_threshold_storages": below,
            }
        )
    return rows


def recommend_threshold(
    predictions: Sequence[Dict[str, Any]],
    *,
    gpu_budget_bytes: int = 24 * (1 << 30),
    safety_margin_bytes: int = 2 * (1 << 30),
) -> Dict[str, Any]:
    """Pick the most aggressive threshold whose predicted protected GPU bytes fit.

    "Safest" against OOM among the sweep: lowest threshold that keeps
    predicted protected GPU bytes under ``gpu_budget - margin``. If none fit,
    return the lowest threshold (maximum offload).
    """
    if not predictions:
        return {
            "threshold_bytes": 256 << 10,
            "reason": "empty_predictions_default_256KiB",
        }
    ordered = sorted(predictions, key=lambda r: int(r["threshold_bytes"]))
    limit = int(gpu_budget_bytes) - int(safety_margin_bytes)
    fitting = [
        row
        for row in ordered
        if int(row["predicted_protected_gpu_bytes"]) <= limit
    ]
    # Prefer lowest threshold among those that fit (most offload headroom).
    chosen = fitting[0] if fitting else ordered[0]
    return {
        "threshold_bytes": int(chosen["threshold_bytes"]),
        "predicted_unique_offloaded_bytes": int(
            chosen["predicted_unique_offloaded_bytes"]
        ),
        "predicted_protected_gpu_bytes": int(chosen["predicted_protected_gpu_bytes"]),
        "fits_budget_with_margin": bool(fitting),
        "budget_limit_bytes": limit,
        "reason": (
            "lowest_threshold_with_predicted_protected_gpu_under_budget_margin"
            if fitting
            else "no_threshold_fit_use_most_aggressive_offload"
        ),
    }


@dataclass
class _CpuStorageEntry:
    cpu_storage: torch.UntypedStorage
    nbytes: int
    dtype: torch.dtype
    original_device: torch.device
    save_id: int
    storage_ptr_at_save: int
    tensor_version: Optional[int] = None
    cpu_checksum: Optional[str] = None


@dataclass
class _PackedOffload:
    kind: str  # "offload"
    storage_id: int
    save_id: int
    size: Tuple[int, ...]
    stride: Tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    module_context: Optional[str] = None
    storage_ptr_at_save: int = 0
    tensor_version: Optional[int] = None
    cpu_checksum: Optional[str] = None
    # Scalar samples only (CPU, tiny); never retain full tensors / flat copies
    # unless full_unpack_verify retains a diagnostic reference.
    sample_lin_indices: Optional[Tuple[int, ...]] = None
    value_fingerprint: Optional[Tuple[float, ...]] = None
    pack_sample_isfinite: Optional[Tuple[bool, ...]] = None
    reference_cpu: Optional[torch.Tensor] = None
    layout_note: str = (
        "restore via Tensor.set_(storage, offset, size, stride); "
        "no contiguous() by default"
    )


@dataclass
class _PackedResident:
    kind: str  # "resident"
    tensor: torch.Tensor
    reason: str
    protection_kind: str = ""


@dataclass
class SelectiveOffloadStats:
    pack_calls: int = 0
    unpack_calls: int = 0
    offloaded_tensors: int = 0
    resident_tensors: int = 0
    protected_attn_bias_or_mask: int = 0
    protected_below_threshold: int = 0
    unique_storages_offloaded: int = 0
    unique_offloaded_storage_bytes: int = 0
    logical_offloaded_tensor_bytes: int = 0
    gpu_resident_protected_tensors: int = 0
    gpu_resident_protected_bytes: int = 0
    protected_attention_bias_tensors: int = 0
    protected_attention_bias_bytes: int = 0
    protected_attention_mask_tensors: int = 0
    protected_attention_mask_bytes: int = 0
    protected_below_threshold_bytes: int = 0
    other_resident_tensors: int = 0
    other_resident_bytes: int = 0
    old_stride1_rule_hits: int = 0
    old_stride1_only_now_offloaded: int = 0
    old_stride1_only_now_offloaded_logical_bytes: int = 0
    old_stride1_only_now_offloaded_storage_bytes: int = 0
    old_stride1_only_now_below_threshold: int = 0
    old_stride1_only_now_below_threshold_logical_bytes: int = 0
    unpack_value_mismatches: int = 0
    unpack_stride_changes: int = 0
    unpack_nonfinite_count: int = 0
    pack_nonfinite_sample_count: int = 0
    sdpa_identity_pack_calls: int = 0
    sdpa_protected_tensors: int = 0
    sdpa_protected_unique_storages: int = 0
    sdpa_protected_unique_bytes: int = 0
    lora_identity_pack_calls: int = 0
    mlp_identity_pack_calls: int = 0
    global_identity_pack_calls: int = 0
    storage_dedup_hits: int = 0
    unique_save_events: int = 0
    full_unpack_value_mismatches: int = 0
    full_unpack_checksum_mismatches: int = 0
    peak_live_unpacked_storages: int = 0
    peak_live_unpacked_bytes: int = 0
    peak_allocated_bytes_at_unpack: Optional[int] = None
    peak_unpack_call_index: Optional[int] = None
    category_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_calls": self.pack_calls,
            "unpack_calls": self.unpack_calls,
            "offloaded_tensors": self.offloaded_tensors,
            "resident_tensors": self.resident_tensors,
            "protected_attn_bias_or_mask": self.protected_attn_bias_or_mask,
            "protected_below_threshold": self.protected_below_threshold,
            "unique_storages_offloaded": self.unique_storages_offloaded,
            "unique_offloaded_storage_bytes": self.unique_offloaded_storage_bytes,
            "logical_offloaded_tensor_bytes": self.logical_offloaded_tensor_bytes,
            "gpu_resident_protected_tensors": self.gpu_resident_protected_tensors,
            "gpu_resident_protected_bytes": self.gpu_resident_protected_bytes,
            "protected_bytes_by_class": {
                "confirmed_attention_bias": self.protected_attention_bias_bytes,
                "confirmed_attention_mask": self.protected_attention_mask_bytes,
                "below_threshold": self.protected_below_threshold_bytes,
                "other_resident": self.other_resident_bytes,
            },
            "protected_tensors_by_class": {
                "confirmed_attention_bias": self.protected_attention_bias_tensors,
                "confirmed_attention_mask": self.protected_attention_mask_tensors,
                "below_threshold": self.protected_below_threshold,
                "other_resident": self.other_resident_tensors,
            },
            "old_stride1_rule_hits": self.old_stride1_rule_hits,
            "old_stride1_only_now_offloaded": self.old_stride1_only_now_offloaded,
            "old_stride1_only_now_offloaded_logical_bytes": (
                self.old_stride1_only_now_offloaded_logical_bytes
            ),
            "old_stride1_only_now_offloaded_storage_bytes": (
                self.old_stride1_only_now_offloaded_storage_bytes
            ),
            "old_stride1_only_now_below_threshold": (
                self.old_stride1_only_now_below_threshold
            ),
            "old_stride1_only_now_below_threshold_logical_bytes": (
                self.old_stride1_only_now_below_threshold_logical_bytes
            ),
            "unpack_value_mismatches": self.unpack_value_mismatches,
            "unpack_stride_changes": self.unpack_stride_changes,
            "unpack_nonfinite_count": self.unpack_nonfinite_count,
            "pack_nonfinite_sample_count": self.pack_nonfinite_sample_count,
            "sdpa_identity_pack_calls": self.sdpa_identity_pack_calls,
            "sdpa_protected_tensors": self.sdpa_protected_tensors,
            "sdpa_protected_unique_storages": self.sdpa_protected_unique_storages,
            "sdpa_protected_unique_bytes": self.sdpa_protected_unique_bytes,
            "lora_identity_pack_calls": self.lora_identity_pack_calls,
            "mlp_identity_pack_calls": self.mlp_identity_pack_calls,
            "global_identity_pack_calls": self.global_identity_pack_calls,
            "storage_dedup_hits": self.storage_dedup_hits,
            "unique_save_events": self.unique_save_events,
            "full_unpack_value_mismatches": self.full_unpack_value_mismatches,
            "full_unpack_checksum_mismatches": self.full_unpack_checksum_mismatches,
            "protected_bytes_by_class_with_sdpa": {
                "confirmed_attention_bias": self.protected_attention_bias_bytes,
                "confirmed_attention_mask": self.protected_attention_mask_bytes,
                "sdpa_context_identity_resident": self.sdpa_protected_unique_bytes,
                "below_threshold": self.protected_below_threshold_bytes,
                "other_resident": self.other_resident_bytes,
            },
            "peak_live_unpacked_storages": self.peak_live_unpacked_storages,
            "peak_live_unpacked_bytes": self.peak_live_unpacked_bytes,
            "peak_allocated_bytes_at_unpack": self.peak_allocated_bytes_at_unpack,
            "peak_unpack_call_index": self.peak_unpack_call_index,
            "category_counts": dict(self.category_counts),
            "storage_dedup_saved_logical_bytes": max(
                0,
                self.logical_offloaded_tensor_bytes
                - self.unique_offloaded_storage_bytes,
            ),
        }


class SelectiveSavedTensorOffload:
    """Pack/unpack hooks with selective CPU offload + contextual attn protection."""

    def __init__(
        self,
        *,
        threshold_bytes: int = 1 << 20,
        pin_memory: bool = True,
        protect_attn_bias: bool = True,
        verify_unpack_values: bool = False,
        full_unpack_verify: bool = False,
        allow_storage_dedup: bool = False,
        max_log_entries: int = 4000,
        offload_devices: Optional[Sequence[str]] = None,
        track_unpack_memory: bool = True,
    ) -> None:
        self.threshold_bytes = int(threshold_bytes)
        self.pin_memory = bool(pin_memory)
        self.protect_attn_bias = bool(protect_attn_bias)
        self.verify_unpack_values = bool(verify_unpack_values)
        # Short-fixture default: retain full CPU reference + checksum per save.
        self.full_unpack_verify = bool(full_unpack_verify)
        # Pointer-only dedupe is unsafe (allocator reuse / in-place mutation).
        # Short-fixture oracle disables it; each save event gets a unique payload.
        self.allow_storage_dedup = bool(allow_storage_dedup)
        self.max_log_entries = int(max_log_entries)
        self.offload_devices = set(offload_devices or ("cuda",))
        self.track_unpack_memory = bool(track_unpack_memory)
        self.stats = SelectiveOffloadStats()
        self.tensor_log: List[Dict[str, Any]] = []
        self.pack_catalog: List[Dict[str, Any]] = []
        self.unpack_telemetry: List[Dict[str, Any]] = []
        self.full_verify_log: List[Dict[str, Any]] = []
        self.first_nonfinite_unpack: Optional[Dict[str, Any]] = None
        self.first_unpack_value_mismatch: Optional[Dict[str, Any]] = None
        self.first_full_unpack_mismatch: Optional[Dict[str, Any]] = None
        self._cpu_storages: Dict[int, _CpuStorageEntry] = {}
        # Live restored GPU storages: storage_id -> (storage, nbytes, live_tensor_ids)
        self._live_gpu: Dict[int, Tuple[torch.UntypedStorage, int, Set[int]]] = {}
        self._next_storage_id = 1
        self._next_save_id = 1
        # Optional dedupe key includes tensor version when available.
        self._ptr_to_storage_id: Dict[Tuple[Any, ...], int] = {}
        self._stride1_offload_storage_keys: Set[str] = set()
        self._sdpa_storage_keys: Set[str] = set()

    @staticmethod
    def _tensor_version(tensor: torch.Tensor) -> Optional[int]:
        try:
            return int(tensor._version)
        except Exception:
            return None

    @staticmethod
    def _cpu_checksum(tensor: torch.Tensor) -> str:
        """Deterministic checksum of the logical tensor bytes on CPU."""
        cpu = tensor.detach().to("cpu").contiguous()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha1(raw).hexdigest()
        flat = cpu.reshape(-1).float()
        finite = torch.isfinite(flat)
        return (
            f"sha1={digest};numel={int(flat.numel())};"
            f"finite={int(finite.sum().item())};"
            f"sumabs={float(flat[finite].abs().sum().item()) if finite.any() else 0.0}"
        )

    def note_identity_pack(self, tensor: torch.Tensor, *, kind: str) -> None:
        """Record a tensor kept resident by nested identity hooks."""
        if kind == "sdpa":
            self.note_sdpa_identity_pack(tensor)
            return
        if kind == "lora":
            self.stats.lora_identity_pack_calls += 1
        elif kind == "mlp":
            self.stats.mlp_identity_pack_calls += 1
        elif kind == "global":
            self.stats.global_identity_pack_calls += 1
        if torch.is_tensor(tensor):
            self._log(
                describe_tensor(
                    tensor,
                    category=f"protected_{kind}_identity_resident",
                    offloaded=False,
                    module_context=_current_module_context(),
                    protection_kind=f"{kind}_context_identity_resident",
                )
            )

    def note_sdpa_identity_pack(self, tensor: torch.Tensor) -> None:
        """Record a tensor kept GPU-resident by nested SDPA identity hooks.

        Must not copy or reconstruct the tensor — efficient SDPA backward needs
        the original CUDA allocation/alignment.
        """
        self.stats.sdpa_identity_pack_calls += 1
        if not torch.is_tensor(tensor) or tensor.device.type != "cuda":
            return
        self.stats.sdpa_protected_tensors += 1
        key = self._storage_key_str(tensor)
        if key not in self._sdpa_storage_keys:
            self._sdpa_storage_keys.add(key)
            nbytes = tensor_storage_nbytes(tensor)
            self.stats.sdpa_protected_unique_storages += 1
            self.stats.sdpa_protected_unique_bytes += nbytes
        module_context = _current_module_context()
        self._log(
            describe_tensor(
                tensor,
                category="protected_sdpa_identity_resident",
                offloaded=False,
                module_context=module_context,
                protection_kind="sdpa_context_identity_resident",
            )
        )

    def _bump_category(self, category: str) -> None:
        self.stats.category_counts[category] = (
            self.stats.category_counts.get(category, 0) + 1
        )

    def _log(self, entry: Dict[str, Any]) -> None:
        # Metadata only — never store tensor objects.
        if len(self.tensor_log) < self.max_log_entries:
            self.tensor_log.append(entry)
        self._bump_category(str(entry.get("category")))

    def _storage_key(self, tensor: torch.Tensor) -> Tuple[int, int, str]:
        storage = tensor.untyped_storage()
        return (int(storage.data_ptr()), int(storage.nbytes()), str(tensor.device))

    def _storage_key_str(self, tensor: torch.Tensor) -> str:
        ptr, nbytes, device = self._storage_key(tensor)
        return f"{device}:{ptr}:{nbytes}"

    def _copy_storage_to_cpu(self, tensor: torch.Tensor) -> torch.UntypedStorage:
        storage = tensor.untyped_storage()
        cpu = storage.cpu()
        if self.pin_memory and torch.cuda.is_available():
            try:
                cpu = cpu.pin_memory()
            except RuntimeError:
                pass
        return cpu

    def _live_unpacked_bytes(self) -> int:
        return int(sum(nbytes for _s, nbytes, ids in self._live_gpu.values() if ids))

    def _release_live_tensor(self, storage_id: int, tensor_id: int) -> None:
        entry = self._live_gpu.get(storage_id)
        if entry is None:
            return
        storage, nbytes, ids = entry
        ids.discard(tensor_id)
        if not ids:
            # Drop strong ref to restored GPU storage.
            self._live_gpu.pop(storage_id, None)
            del storage

    def _register_live_tensor(
        self,
        storage_id: int,
        gpu_storage: torch.UntypedStorage,
        nbytes: int,
        tensor: torch.Tensor,
    ) -> None:
        tensor_id = id(tensor)
        if storage_id in self._live_gpu:
            storage, existing_nbytes, ids = self._live_gpu[storage_id]
            ids.add(tensor_id)
            nbytes = existing_nbytes
        else:
            self._live_gpu[storage_id] = (gpu_storage, nbytes, {tensor_id})
        weakref.finalize(
            tensor, SelectiveSavedTensorOffload._release_live_tensor_static, self, storage_id, tensor_id
        )
        live_n = len(self._live_gpu)
        live_b = self._live_unpacked_bytes()
        if live_n > self.stats.peak_live_unpacked_storages:
            self.stats.peak_live_unpacked_storages = live_n
        if live_b > self.stats.peak_live_unpacked_bytes:
            self.stats.peak_live_unpacked_bytes = live_b

    @staticmethod
    def _release_live_tensor_static(
        owner: "SelectiveSavedTensorOffload", storage_id: int, tensor_id: int
    ) -> None:
        owner._release_live_tensor(storage_id, tensor_id)

    def pack_hook(self, tensor: torch.Tensor) -> Any:
        self.stats.pack_calls += 1
        module_context = _current_module_context()

        if not torch.is_tensor(tensor):
            self.stats.resident_tensors += 1
            self.stats.other_resident_tensors += 1
            self._log(
                {
                    "category": "resident_non_tensor",
                    "protection_kind": "other_resident",
                    "offloaded": False,
                    "module_context": module_context,
                    "type": type(tensor).__name__,
                }
            )
            return _PackedResident("resident", tensor, "non_tensor", "other_resident")

        old_stride1 = matches_old_stride1_only_rule(tensor)
        if old_stride1:
            self.stats.old_stride1_rule_hits += 1

        device_type = tensor.device.type
        storage_nbytes = tensor_storage_nbytes(tensor)
        logical_nbytes = tensor_logical_nbytes(tensor)
        storage_key = self._storage_key_str(tensor)

        if device_type not in self.offload_devices:
            self.stats.resident_tensors += 1
            self.stats.other_resident_tensors += 1
            self.stats.other_resident_bytes += logical_nbytes
            entry = describe_tensor(
                tensor,
                category="resident_non_cuda",
                offloaded=False,
                module_context=module_context,
                protection_kind="other_resident",
                old_stride1_rule=old_stride1,
            )
            self._log(entry)
            self.pack_catalog.append(
                {
                    "storage_key": storage_key,
                    "storage_nbytes": storage_nbytes,
                    "logical_nbytes": logical_nbytes,
                    "protection_kind": "other_resident",
                    "device_type": device_type,
                    "old_stride1_only_rule": old_stride1,
                    "offloaded": False,
                }
            )
            return _PackedResident("resident", tensor, "non_cuda", "other_resident")

        # Belt-and-suspenders: nested identity regions keep GPU-resident.
        identity_kind = None
        if bool(getattr(_TLS, "inside_global_identity", False)):
            identity_kind = "global"
        elif bool(getattr(_TLS, "inside_mlp_identity", False)):
            identity_kind = "mlp"
        elif bool(getattr(_TLS, "inside_lora", False)):
            identity_kind = "lora"
        elif bool(getattr(_TLS, "inside_sdpa", False)):
            identity_kind = "sdpa"
        if identity_kind is not None:
            self.stats.resident_tensors += 1
            self.stats.gpu_resident_protected_tensors += 1
            self.stats.gpu_resident_protected_bytes += logical_nbytes
            self.note_identity_pack(tensor, kind=identity_kind)
            protection_kind = f"{identity_kind}_context_identity_resident"
            self.pack_catalog.append(
                {
                    "storage_key": storage_key,
                    "storage_nbytes": storage_nbytes,
                    "logical_nbytes": logical_nbytes,
                    "protection_kind": protection_kind,
                    "device_type": device_type,
                    "old_stride1_only_rule": old_stride1,
                    "offloaded": False,
                    "module_context": module_context,
                }
            )
            return _PackedResident(
                "resident",
                tensor,
                f"inside_{identity_kind}_tls",
                protection_kind,
            )

        protect = False
        protect_reason = ""
        protection_kind = ""
        if self.protect_attn_bias:
            protect, protect_reason, protection_kind = classify_attention_protection(
                tensor, module_context=module_context
            )

        if protect:
            self.stats.resident_tensors += 1
            self.stats.protected_attn_bias_or_mask += 1
            self.stats.gpu_resident_protected_tensors += 1
            self.stats.gpu_resident_protected_bytes += logical_nbytes
            if protection_kind == "confirmed_attention_mask":
                self.stats.protected_attention_mask_tensors += 1
                self.stats.protected_attention_mask_bytes += logical_nbytes
            else:
                self.stats.protected_attention_bias_tensors += 1
                self.stats.protected_attention_bias_bytes += logical_nbytes
            category = f"protected_{protection_kind}:{protect_reason}"
            self._log(
                describe_tensor(
                    tensor,
                    category=category,
                    offloaded=False,
                    module_context=module_context,
                    protection_kind=protection_kind,
                    old_stride1_rule=old_stride1,
                )
            )
            self.pack_catalog.append(
                {
                    "storage_key": storage_key,
                    "storage_nbytes": storage_nbytes,
                    "logical_nbytes": logical_nbytes,
                    "protection_kind": protection_kind,
                    "device_type": device_type,
                    "old_stride1_only_rule": old_stride1,
                    "offloaded": False,
                    "module_context": module_context,
                }
            )
            return _PackedResident(
                "resident", tensor, protect_reason, protection_kind
            )

        if storage_nbytes < self.threshold_bytes and logical_nbytes < self.threshold_bytes:
            if old_stride1:
                self.stats.old_stride1_only_now_below_threshold += 1
                self.stats.old_stride1_only_now_below_threshold_logical_bytes += (
                    logical_nbytes
                )
            self.stats.resident_tensors += 1
            self.stats.protected_below_threshold += 1
            self.stats.protected_below_threshold_bytes += logical_nbytes
            self.stats.gpu_resident_protected_tensors += 1
            self.stats.gpu_resident_protected_bytes += logical_nbytes
            self._log(
                describe_tensor(
                    tensor,
                    category="protected_below_threshold",
                    offloaded=False,
                    module_context=module_context,
                    protection_kind="below_threshold",
                    old_stride1_rule=old_stride1,
                )
            )
            self.pack_catalog.append(
                {
                    "storage_key": storage_key,
                    "storage_nbytes": storage_nbytes,
                    "logical_nbytes": logical_nbytes,
                    "protection_kind": "below_threshold",
                    "device_type": device_type,
                    "old_stride1_only_rule": old_stride1,
                    "offloaded": False,
                    "module_context": module_context,
                }
            )
            return _PackedResident(
                "resident", tensor, "below_threshold", "below_threshold"
            )

        # Previously stride1-only protected tensors that are now actually offloaded.
        if old_stride1:
            self.stats.old_stride1_only_now_offloaded += 1
            self.stats.old_stride1_only_now_offloaded_logical_bytes += logical_nbytes
            if storage_key not in self._stride1_offload_storage_keys:
                self._stride1_offload_storage_keys.add(storage_key)
                self.stats.old_stride1_only_now_offloaded_storage_bytes += storage_nbytes

        save_id = self._next_save_id
        self._next_save_id += 1
        self.stats.unique_save_events += 1
        storage_ptr = int(tensor.untyped_storage().data_ptr())
        tensor_version = self._tensor_version(tensor)
        # Safe key (when dedupe enabled): pointer alone is insufficient because
        # CUDA allocator pointers can be reused after a tensor is released.
        dedupe_key = (
            storage_ptr,
            int(storage_nbytes),
            str(tensor.device),
            str(tensor.dtype),
            tensor_version,
        )
        first_seen = True
        if self.allow_storage_dedup and dedupe_key in self._ptr_to_storage_id:
            storage_id = self._ptr_to_storage_id[dedupe_key]
            first_seen = False
            self.stats.storage_dedup_hits += 1
        else:
            storage_id = self._next_storage_id
            self._next_storage_id += 1
            cpu_storage = self._copy_storage_to_cpu(tensor)
            checksum = None
            if self.full_unpack_verify or self.verify_unpack_values:
                try:
                    checksum = self._cpu_checksum(tensor)
                except Exception:
                    checksum = None
            self._cpu_storages[storage_id] = _CpuStorageEntry(
                cpu_storage=cpu_storage,
                nbytes=storage_nbytes,
                dtype=tensor.dtype,
                original_device=tensor.device,
                save_id=save_id,
                storage_ptr_at_save=storage_ptr,
                tensor_version=tensor_version,
                cpu_checksum=checksum,
            )
            if self.allow_storage_dedup:
                self._ptr_to_storage_id[dedupe_key] = storage_id
            self.stats.unique_storages_offloaded += 1
            self.stats.unique_offloaded_storage_bytes += storage_nbytes

        # Always keep tiny deterministic samples for non-finite / corruption probes.
        sample_vals, sample_idxs, sample_finite = sample_tensor_values(
            tensor, max_samples=8
        )
        if sample_finite and not all(sample_finite):
            self.stats.pack_nonfinite_sample_count += 1

        reference_cpu = None
        checksum = self._cpu_storages[storage_id].cpu_checksum
        if self.full_unpack_verify:
            # Retain logical view for exact allclose at unpack (short fixture).
            reference_cpu = tensor.detach().to("cpu").clone()
            if checksum is None:
                try:
                    checksum = self._cpu_checksum(tensor)
                except Exception:
                    checksum = None

        self.stats.offloaded_tensors += 1
        self.stats.logical_offloaded_tensor_bytes += logical_nbytes
        self._log(
            describe_tensor(
                tensor,
                category="offloaded_large_activation",
                offloaded=True,
                module_context=module_context,
                shared_storage_id=storage_id,
                first_seen_storage=first_seen,
                protection_kind=None,
                old_stride1_rule=old_stride1,
            )
        )
        self.pack_catalog.append(
            {
                "save_id": save_id,
                "storage_key": storage_key,
                "storage_nbytes": storage_nbytes,
                "logical_nbytes": logical_nbytes,
                "protection_kind": None,
                "device_type": device_type,
                "old_stride1_only_rule": old_stride1,
                "offloaded": True,
                "module_context": module_context,
                "shared_storage_id": storage_id,
                "storage_ptr_at_save": storage_ptr,
                "storage_offset": int(tensor.storage_offset()),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "tensor_version": tensor_version,
                "cpu_checksum": checksum,
                "storage_dedup_hit": not first_seen,
            }
        )
        return _PackedOffload(
            kind="offload",
            storage_id=storage_id,
            save_id=save_id,
            size=tuple(int(x) for x in tensor.size()),
            stride=tuple(int(x) for x in tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            dtype=tensor.dtype,
            device=tensor.device,
            module_context=module_context,
            storage_ptr_at_save=storage_ptr,
            tensor_version=tensor_version,
            cpu_checksum=checksum,
            sample_lin_indices=sample_idxs,
            value_fingerprint=sample_vals,
            pack_sample_isfinite=sample_finite,
            reference_cpu=reference_cpu,
        )

    def unpack_hook(self, packed: Any) -> torch.Tensor:
        self.stats.unpack_calls += 1
        unpack_index = self.stats.unpack_calls

        mem_before = None
        if self.track_unpack_memory and torch.cuda.is_available():
            try:
                mem_before = int(torch.cuda.memory_allocated())
            except Exception:
                mem_before = None

        if isinstance(packed, _PackedResident):
            telemetry = {
                "unpack_index": unpack_index,
                "kind": "resident",
                "protection_kind": packed.protection_kind,
                "allocated_before": mem_before,
                "allocated_after": mem_before,
                "live_unpacked_storages": len(self._live_gpu),
                "live_unpacked_bytes": self._live_unpacked_bytes(),
            }
            self.unpack_telemetry.append(telemetry)
            return packed.tensor

        if not isinstance(packed, _PackedOffload):
            if torch.is_tensor(packed):
                return packed
            raise TypeError(f"unknown packed saved tensor type: {type(packed)!r}")

        storage_id = packed.storage_id
        entry = self._cpu_storages[storage_id]
        if int(getattr(packed, "save_id", -1)) != int(entry.save_id) and not self.allow_storage_dedup:
            # Unique-save mode: packed save_id must match the CPU payload event.
            raise RuntimeError(
                f"save_id mismatch on unpack: packed={packed.save_id} "
                f"entry={entry.save_id} storage_id={storage_id}"
            )
        live = self._live_gpu.get(storage_id)
        live_storage_reused = bool(live is not None and live[2])
        if live_storage_reused:
            gpu_storage, nbytes, _ids = live  # type: ignore[misc]
        else:
            target = packed.device
            if target.type == "cuda":
                device_index = cuda_storage_device_index(target)
                gpu_storage = entry.cpu_storage.cuda(
                    device=device_index,
                    non_blocking=bool(self.pin_memory),
                )
            else:
                gpu_storage = entry.cpu_storage
            nbytes = int(entry.nbytes)

        out = torch.empty(0, dtype=packed.dtype, device=packed.device)
        out.set_(
            gpu_storage,
            packed.storage_offset,
            torch.Size(packed.size),
            packed.stride,
        )
        self._register_live_tensor(storage_id, gpu_storage, nbytes, out)

        if tuple(int(s) for s in out.stride()) != packed.stride:
            self.stats.unpack_stride_changes += 1

        if self.full_unpack_verify and packed.reference_cpu is not None:
            try:
                restored_cpu = out.detach().to("cpu")
                if restored_cpu.shape != packed.reference_cpu.shape:
                    mismatch = True
                else:
                    mismatch = not bool(
                        torch.equal(
                            restored_cpu.contiguous().view(torch.uint8),
                            packed.reference_cpu.contiguous().view(torch.uint8),
                        )
                    )
                checksum_now = self._cpu_checksum(out)
                checksum_mismatch = (
                    packed.cpu_checksum is not None
                    and checksum_now != packed.cpu_checksum
                )
                if mismatch:
                    self.stats.full_unpack_value_mismatches += 1
                if checksum_mismatch:
                    self.stats.full_unpack_checksum_mismatches += 1
                verify_entry = {
                    "save_id": packed.save_id,
                    "storage_id": storage_id,
                    "module_context": packed.module_context,
                    "shape": list(packed.size),
                    "stride": list(packed.stride),
                    "storage_offset": int(packed.storage_offset),
                    "dtype": str(packed.dtype).replace("torch.", ""),
                    "storage_ptr_at_save": packed.storage_ptr_at_save,
                    "tensor_version": packed.tensor_version,
                    "cpu_checksum_pack": packed.cpu_checksum,
                    "cpu_checksum_unpack": checksum_now,
                    "value_mismatch": mismatch,
                    "checksum_mismatch": checksum_mismatch,
                }
                if len(self.full_verify_log) < self.max_log_entries:
                    self.full_verify_log.append(verify_entry)
                if (mismatch or checksum_mismatch) and self.first_full_unpack_mismatch is None:
                    self.first_full_unpack_mismatch = verify_entry
                    self.stats.unpack_value_mismatches += 1
                    if self.first_unpack_value_mismatch is None:
                        self.first_unpack_value_mismatch = dict(verify_entry)
            except Exception as exc:
                if self.first_full_unpack_mismatch is None:
                    self.first_full_unpack_mismatch = {
                        "save_id": packed.save_id,
                        "error": str(exc),
                    }

        restored_vals: Optional[Tuple[float, ...]] = None
        restored_finite: Optional[Tuple[bool, ...]] = None
        sample_mismatch = False
        sample_nonfinite = False
        if packed.sample_lin_indices is not None:
            restored_vals, _idxs, restored_finite = sample_tensor_values(
                out,
                lin_indices=packed.sample_lin_indices,
            )
            if restored_finite and not all(restored_finite):
                sample_nonfinite = True
                self.stats.unpack_nonfinite_count += 1
                if self.first_nonfinite_unpack is None:
                    self.first_nonfinite_unpack = {
                        "unpack_index": unpack_index,
                        "storage_id": storage_id,
                        "shape": list(packed.size),
                        "stride": list(packed.stride),
                        "storage_offset": int(packed.storage_offset),
                        "dtype": str(packed.dtype).replace("torch.", ""),
                        "module_context": packed.module_context,
                        "sample_lin_indices": list(packed.sample_lin_indices),
                        "original_sampled_values": list(packed.value_fingerprint or ()),
                        "pack_sample_isfinite": list(packed.pack_sample_isfinite or ()),
                        "restored_sampled_values": list(restored_vals),
                        "restored_sample_isfinite": list(restored_finite),
                        "live_storage_reused": live_storage_reused,
                    }
            if (
                self.verify_unpack_values
                and packed.value_fingerprint is not None
                and restored_vals != packed.value_fingerprint
            ):
                sample_mismatch = True
                self.stats.unpack_value_mismatches += 1
                if self.first_unpack_value_mismatch is None:
                    self.first_unpack_value_mismatch = {
                        "unpack_index": unpack_index,
                        "storage_id": storage_id,
                        "shape": list(packed.size),
                        "stride": list(packed.stride),
                        "storage_offset": int(packed.storage_offset),
                        "dtype": str(packed.dtype).replace("torch.", ""),
                        "module_context": packed.module_context,
                        "sample_lin_indices": list(packed.sample_lin_indices or ()),
                        "original_sampled_values": list(packed.value_fingerprint),
                        "restored_sampled_values": list(restored_vals),
                        "live_storage_reused": live_storage_reused,
                    }

        mem_after = None
        if self.track_unpack_memory and torch.cuda.is_available():
            try:
                mem_after = int(torch.cuda.memory_allocated())
            except Exception:
                mem_after = None
        if mem_after is not None:
            prev_peak = self.stats.peak_allocated_bytes_at_unpack
            if prev_peak is None or mem_after > prev_peak:
                self.stats.peak_allocated_bytes_at_unpack = mem_after
                self.stats.peak_unpack_call_index = unpack_index

        self.unpack_telemetry.append(
            {
                "unpack_index": unpack_index,
                "kind": "offload",
                "storage_id": storage_id,
                "storage_nbytes": nbytes,
                "allocated_before": mem_before,
                "allocated_after": mem_after,
                "live_unpacked_storages": len(self._live_gpu),
                "live_unpacked_bytes": self._live_unpacked_bytes(),
                "sample_mismatch": sample_mismatch,
                "sample_nonfinite": sample_nonfinite,
            }
        )
        return out

    def classification_report(self) -> Dict[str, Any]:
        stride_only_offloaded = [
            e
            for e in self.pack_catalog
            if e.get("old_stride1_only_rule") and bool(e.get("offloaded"))
        ]
        stride_only_below = [
            e
            for e in self.pack_catalog
            if e.get("old_stride1_only_rule")
            and e.get("protection_kind") == "below_threshold"
        ]
        uniq_off = {
            e["storage_key"]: int(e.get("storage_nbytes") or 0)
            for e in stride_only_offloaded
        }
        uniq_below = {
            e["storage_key"]: int(e.get("storage_nbytes") or 0)
            for e in stride_only_below
        }
        return {
            "old_stride1_rule_hits": self.stats.old_stride1_rule_hits,
            "old_stride1_only_now_offloaded_tensors": (
                self.stats.old_stride1_only_now_offloaded
            ),
            "old_stride1_only_now_offloaded_unique_storages": len(uniq_off),
            "old_stride1_only_now_offloaded_unique_storage_bytes": int(
                sum(uniq_off.values())
            ),
            "old_stride1_only_now_offloaded_logical_bytes": (
                self.stats.old_stride1_only_now_offloaded_logical_bytes
            ),
            "old_stride1_only_now_below_threshold_tensors": (
                self.stats.old_stride1_only_now_below_threshold
            ),
            "old_stride1_only_now_below_threshold_unique_storage_bytes": int(
                sum(uniq_below.values())
            ),
            "protected_bytes_by_class": {
                "confirmed_attention_bias": self.stats.protected_attention_bias_bytes,
                "confirmed_attention_mask": self.stats.protected_attention_mask_bytes,
                "below_threshold": self.stats.protected_below_threshold_bytes,
                "other_resident": self.stats.other_resident_bytes,
            },
            "note": (
                "stride1_not_multiple_of_4 is classification-only; it no longer "
                "protects tensors by itself. 'now_offloaded' counts former "
                "stride1-only hits that are actually offloaded at the active "
                "threshold."
            ),
        }

    def threshold_prediction_report(
        self, thresholds: Sequence[int] = DEFAULT_THRESHOLD_SWEEP
    ) -> Dict[str, Any]:
        predictions = predict_threshold_outcomes(self.pack_catalog, thresholds)
        recommendation = recommend_threshold(predictions)
        return {
            "predictions": predictions,
            "recommendation": recommendation,
            "classification": self.classification_report(),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "threshold_bytes": self.threshold_bytes,
            "pin_memory": self.pin_memory,
            "protect_attn_bias": self.protect_attn_bias,
            "verify_unpack_values": self.verify_unpack_values,
            "full_unpack_verify": self.full_unpack_verify,
            "allow_storage_dedup": self.allow_storage_dedup,
            "stats": self.stats.to_dict(),
            "classification": self.classification_report(),
            "threshold_predictions": self.threshold_prediction_report(),
            "first_nonfinite_unpack": self.first_nonfinite_unpack,
            "first_unpack_value_mismatch": self.first_unpack_value_mismatch,
            "first_full_unpack_mismatch": self.first_full_unpack_mismatch,
            "full_verify_log_tail": self.full_verify_log[-32:],
            "sdpa_protection": {
                "method": "nested_identity_saved_tensors_hooks_around_sdpa",
                "reconstructs_via_cpu_storage": False,
                "unique_bytes": self.stats.sdpa_protected_unique_bytes,
                "unique_storages": self.stats.sdpa_protected_unique_storages,
                "identity_pack_calls": self.stats.sdpa_identity_pack_calls,
                "estimated_non_sdpa_offload_still_needed": True,
                "fit_estimate_1x24gb": {
                    "sdpa_resident_unique_bytes": self.stats.sdpa_protected_unique_bytes,
                    "unique_offloaded_storage_bytes": (
                        self.stats.unique_offloaded_storage_bytes
                    ),
                    "note": (
                        "SDPA-saved tensors stay on GPU with original storage; "
                        "only non-SDPA activations are offloaded."
                    ),
                },
            },
            "tensor_log_entries": len(self.tensor_log),
            "tensor_log_truncated": self.max_log_entries,
            "unpack_telemetry_entries": len(self.unpack_telemetry),
            "live_unpacked_storages": len(self._live_gpu),
            "live_unpacked_bytes": self._live_unpacked_bytes(),
            "layout_reconstruction": {
                "method": "Tensor.set_(untyped_storage, storage_offset, size, stride)",
                "calls_contiguous_by_default": False,
                "values_exact_when_storage_copy_exact": True,
                "stride_preserved_by_construction": True,
                "sdpa_tensors_use_identity_no_cpu_roundtrip": True,
                "gpu_storage_lifetime": (
                    "weakref-tracked per unpacked tensor; storage dropped when "
                    "no live unpack products remain"
                ),
            },
        }

    def clear_gpu_restore_cache(self) -> None:
        self._live_gpu.clear()


def _capture_attn_args(args: Any, kwargs: Dict[str, Any]) -> None:
    for key, value in kwargs.items():
        key_l = str(key).lower()
        if any(tok in key_l for tok in ("mask", "bias")):
            register_confirmed_attn_tensor(value)
    # Positional tensors that already look like q/k masks inside attn modules.
    for value in args:
        if torch.is_tensor(value) and (
            value.dtype == torch.bool or value.dtype.is_floating_point
        ):
            shape_ok, _ = _has_qk_mask_bias_shape(value)
            if shape_ok or (value.dtype == torch.bool and value.ndim >= 2):
                register_confirmed_attn_tensor(value)


@contextmanager
def module_context_hooks(model: nn.Module) -> Iterator[None]:
    """Push module names and capture attention mask/bias argument identities."""
    handles = []
    _clear_confirmed_attn_registry()

    def _pre(mod: nn.Module, args: Any, kwargs: Any = None) -> None:
        name = getattr(mod, "_selective_offload_name", mod.__class__.__name__)
        push_module_context(name)
        if is_attention_module_context(name):
            if kwargs is None:
                # Older hook signature (mod, args) — kwargs unavailable.
                _capture_attn_args(args, {})
            else:
                _capture_attn_args(args, dict(kwargs))

    def _post(mod: nn.Module, _inputs: Any, _output: Any) -> None:
        pop_module_context()

    for name, module in model.named_modules():
        module._selective_offload_name = name or module.__class__.__name__  # type: ignore[attr-defined]
        # Prefer with_kwargs when available for attention_mask / attn_bias identity.
        try:
            handles.append(
                module.register_forward_pre_hook(_pre, with_kwargs=True)  # type: ignore[call-arg]
            )
        except TypeError:
            handles.append(module.register_forward_pre_hook(_pre))
        handles.append(module.register_forward_hook(_post))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for _name, module in model.named_modules():
            if hasattr(module, "_selective_offload_name"):
                delattr(module, "_selective_offload_name")
        _TLS.module_stack = []
        _clear_confirmed_attn_registry()


def _identity_pack(
    manager: Optional[SelectiveSavedTensorOffload],
    tensor: Any,
    *,
    kind: str,
) -> Any:
    if manager is not None and torch.is_tensor(tensor):
        manager.note_identity_pack(tensor, kind=kind)
    return tensor


def _identity_unpack(tensor: Any) -> Any:
    return tensor


def _sdpa_identity_pack(manager: Optional[SelectiveSavedTensorOffload], tensor: Any) -> Any:
    """Identity pack: keep the exact CUDA tensor object for efficient SDPA backward."""
    return _identity_pack(manager, tensor, kind="sdpa")


def _sdpa_identity_unpack(tensor: Any) -> Any:
    return _identity_unpack(tensor)


@contextmanager
def _nested_identity_region(
    manager: Optional[SelectiveSavedTensorOffload],
    *,
    kind: str,
    tls_attr: str,
) -> Iterator[None]:
    prev = bool(getattr(_TLS, tls_attr, False))
    setattr(_TLS, tls_attr, True)
    try:
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: _identity_pack(manager, t, kind=kind),
            _identity_unpack,
        ):
            yield
    finally:
        setattr(_TLS, tls_attr, prev)


@contextmanager
def patch_sdpa_nested_identity_hooks(
    manager: Optional[SelectiveSavedTensorOffload] = None,
) -> Iterator[None]:
    """Wrap ``scaled_dot_product_attention`` with nested identity saved_tensors_hooks."""
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention

    def _wrapped_sdpa(*args: Any, **kwargs: Any):
        with _nested_identity_region(manager, kind="sdpa", tls_attr="inside_sdpa"):
            return original(*args, **kwargs)

    F.scaled_dot_product_attention = _wrapped_sdpa  # type: ignore[assignment]
    torch.nn.functional.scaled_dot_product_attention = _wrapped_sdpa  # type: ignore[attr-defined]
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original  # type: ignore[assignment]
        torch.nn.functional.scaled_dot_product_attention = original  # type: ignore[attr-defined]
        _TLS.inside_sdpa = False


def _iter_lora_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    try:
        from peft.tuners.lora import Linear as LoraLinear
    except Exception:  # pragma: no cover
        LoraLinear = tuple()  # type: ignore[assignment]
    found: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if LoraLinear and isinstance(module, LoraLinear):
            found.append((name, module))
        elif hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            found.append((name, module))
    return found


def _iter_mlp_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Complete decoder/vision MLP containers (gate/up/act/mul/down live inside)."""
    found: List[Tuple[str, nn.Module]] = []
    seen: Set[int] = set()
    for name, module in model.named_modules():
        mid = id(module)
        if mid in seen:
            continue
        parts = name.split(".")
        leaf = parts[-1] if parts else ""
        # Exact leaf "mlp" covers Qwen2MLP / LlamaMLP-style modules.
        is_mlp_leaf = leaf == "mlp"
        is_swiglu = (
            hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
            and hasattr(module, "act_fn")
        )
        if is_mlp_leaf or is_swiglu:
            seen.add(mid)
            found.append((name, module))
    return found


@contextmanager
def patch_module_forward_identity_hooks(
    modules: Sequence[Tuple[str, nn.Module]],
    manager: Optional[SelectiveSavedTensorOffload],
    *,
    kind: str,
    tls_attr: str,
) -> Iterator[None]:
    """Wrap selected module.forward with nested identity saved_tensors_hooks."""
    originals: List[Tuple[nn.Module, Any]] = []
    for _name, module in modules:
        original = module.forward

        def _make(orig):
            def _wrapped(*args: Any, **kwargs: Any):
                with _nested_identity_region(manager, kind=kind, tls_attr=tls_attr):
                    return orig(*args, **kwargs)

            return _wrapped

        module.forward = _make(original)  # type: ignore[method-assign]
        originals.append((module, original))
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original  # type: ignore[method-assign]
        setattr(_TLS, tls_attr, False)


def _iter_lora_branch_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """lora_A / lora_B / dropout only — not the base frozen linear."""
    branches: List[Tuple[str, nn.Module]] = []
    for parent_name, parent in _iter_lora_modules(model):
        for attr in ("lora_A", "lora_B", "lora_dropout"):
            container = getattr(parent, attr, None)
            if container is None:
                continue
            if isinstance(container, nn.ModuleDict):
                for adapter, mod in container.items():
                    if isinstance(mod, nn.Module):
                        branches.append((f"{parent_name}.{attr}.{adapter}", mod))
            elif isinstance(container, nn.Module):
                branches.append((f"{parent_name}.{attr}", container))
    return branches


@contextmanager
def patch_lora_nested_identity_hooks(
    model: nn.Module,
    manager: Optional[SelectiveSavedTensorOffload] = None,
) -> Iterator[None]:
    """G1: nested identity hooks around PEFT LoRA A/dropout/B branches only."""
    with patch_module_forward_identity_hooks(
        _iter_lora_branch_modules(model),
        manager,
        kind="lora",
        tls_attr="inside_lora",
    ):
        yield


@contextmanager
def patch_mlp_nested_identity_hooks(
    model: nn.Module,
    manager: Optional[SelectiveSavedTensorOffload] = None,
) -> Iterator[None]:
    """G2: nested identity hooks around complete MLP modules.

    Covers gate_proj, up_proj, activation, elementwise multiply, and down_proj
    because those ops execute inside the parent ``mlp.forward``.
    """
    with patch_module_forward_identity_hooks(
        _iter_mlp_modules(model),
        manager,
        kind="mlp",
        tls_attr="inside_mlp_identity",
    ):
        yield


@contextmanager
def patch_global_identity_hooks(
    manager: Optional[SelectiveSavedTensorOffload] = None,
) -> Iterator[None]:
    """G3 helper: mark global identity TLS (outer hooks are identity)."""
    with _nested_identity_region(
        manager, kind="global", tls_attr="inside_global_identity"
    ):
        yield


@contextmanager
def selective_saved_tensor_offload_context(
    manager: SelectiveSavedTensorOffload,
    model: Optional[nn.Module] = None,
    *,
    identity_guard_level: str = "G0",
) -> Iterator[SelectiveSavedTensorOffload]:
    """Install selective pack/unpack hooks + nested identity protection.

    Guard levels (short-fixture diagnostics):
      G0: SDPA only (default production candidate)
      G1: SDPA + all LoRA branches
      G2: SDPA + all LoRA branches + complete MLP modules
      G3: global identity saved_tensors_hooks for ALL saves (no CPU offload);
          same B wrapper/process, but pack/unpack never copy to CPU
    """
    if not hasattr(torch.autograd.graph, "saved_tensors_hooks"):
        raise RuntimeError(
            "torch.autograd.graph.saved_tensors_hooks unavailable in this PyTorch"
        )
    level = str(identity_guard_level or "G0").upper()
    if level not in IDENTITY_GUARD_LEVELS:
        raise RuntimeError(
            f"unknown identity_guard_level={identity_guard_level!r}; "
            f"expected one of {IDENTITY_GUARD_LEVELS}"
        )
    # G3: keep module-context tracking for B-wrapper parity, but never install
    # selective pack/unpack or nested SDPA/LoRA/MLP identity patches.
    ctx_model = module_context_hooks(model) if model is not None else nullcontext()
    with ctx_model:
        if level == "G3":
            with torch.autograd.graph.saved_tensors_hooks(
                lambda t: _identity_pack(manager, t, kind="global"),
                _identity_unpack,
            ):
                yield manager
            # Hard invariant: G3 must never have offloaded any tensor.
            if int(manager.stats.offloaded_tensors) != 0:
                raise RuntimeError(
                    "G3 violated no-offload invariant: "
                    f"offloaded_tensors={manager.stats.offloaded_tensors}"
                )
        else:
            lora_ctx = (
                patch_lora_nested_identity_hooks(model, manager)
                if level in ("G1", "G2") and model is not None
                else nullcontext()
            )
            mlp_ctx = (
                patch_mlp_nested_identity_hooks(model, manager)
                if level == "G2" and model is not None
                else nullcontext()
            )
            with patch_sdpa_nested_identity_hooks(manager):
                with lora_ctx:
                    with mlp_ctx:
                        with torch.autograd.graph.saved_tensors_hooks(
                            manager.pack_hook, manager.unpack_hook
                        ):
                            yield manager
    manager.clear_gpu_restore_cache()
