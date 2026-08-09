"""A/B reproducibility fingerprints for short-fixture gradient oracle.

Diagnostic only: log RNG / determinism / dropout / parameter / fixture state
so G3 A-vs-B mismatches can be attributed to offload vs process drift.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


DEFAULT_ORACLE_REPRO_SEED = 424242


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def tensor_checksum(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    flat = cpu.reshape(-1).float()
    finite = torch.isfinite(flat)
    return (
        f"sha1={_sha1_bytes(raw)};numel={int(flat.numel())};"
        f"finite={int(finite.sum().item())};"
        f"sumabs={float(flat[finite].abs().sum().item()) if finite.any() else 0.0}"
    )


def rng_state_hash(state: Any) -> Optional[str]:
    if state is None:
        return None
    if torch.is_tensor(state):
        return _sha1_bytes(state.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    if isinstance(state, (bytes, bytearray)):
        return _sha1_bytes(bytes(state))
    if isinstance(state, (list, tuple)):
        parts = []
        for item in state:
            h = rng_state_hash(item)
            parts.append(h or "none")
        return _sha1_bytes("|".join(parts).encode("utf-8"))
    try:
        return _sha1_bytes(repr(state).encode("utf-8"))
    except Exception:
        return None


def apply_identical_oracle_seed(seed: int) -> Dict[str, Any]:
    """Set identical CPU/CUDA RNG seeds immediately before model forward."""
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
        numpy_seeded = True
    except Exception:
        numpy_seeded = False
    torch.manual_seed(seed)
    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True
    return {
        "seed": seed,
        "random_seeded": True,
        "numpy_seeded": numpy_seeded,
        "torch_manual_seed": True,
        "cuda_manual_seed_all": cuda_seeded,
    }


def _lora_dropout_report(model: nn.Module) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        drop = getattr(module, "lora_dropout", None)
        if drop is None:
            continue
        if isinstance(drop, nn.ModuleDict):
            for adapter, mod in drop.items():
                p = getattr(mod, "p", None)
                rows.append(
                    {
                        "module": f"{name}.lora_dropout.{adapter}",
                        "p": float(p) if p is not None else None,
                        "training": bool(getattr(mod, "training", False)),
                        "type": type(mod).__name__,
                    }
                )
        elif isinstance(drop, nn.Module):
            p = getattr(drop, "p", None)
            rows.append(
                {
                    "module": f"{name}.lora_dropout",
                    "p": float(p) if p is not None else None,
                    "training": bool(drop.training),
                    "type": type(drop).__name__,
                }
            )
    return rows[:64]


def _trainable_param_checksums(model: nn.Module, limit: int = 16) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        rows.append(
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype).replace("torch.", ""),
                "checksum": tensor_checksum(param.data),
            }
        )
        if len(rows) >= int(limit):
            break
    return rows


def _fixture_tensor_checksums(
    decoder_kwargs: Optional[Dict[str, Any]],
    trace: Any,
) -> Dict[str, Any]:
    tensors: Dict[str, str] = {}
    if isinstance(decoder_kwargs, dict):
        for key, value in sorted(decoder_kwargs.items()):
            if torch.is_tensor(value):
                tensors[f"decoder_kwargs.{key}"] = tensor_checksum(value)
            elif isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
                for i, t in enumerate(value[:8]):
                    if torch.is_tensor(t):
                        tensors[f"decoder_kwargs.{key}[{i}]"] = tensor_checksum(t)
    prompt = getattr(trace, "prompt_token_ids", None)
    generated = getattr(trace, "generated_token_ids", None)
    if prompt is not None:
        tensors["trace.prompt_token_ids"] = _sha1_bytes(
            repr(list(prompt)).encode("utf-8")
        )
    if generated is not None:
        tensors["trace.generated_token_ids"] = _sha1_bytes(
            repr(list(generated)).encode("utf-8")
        )
    return tensors


def _sdpa_backend_flags() -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    try:
        # torch.backends.cuda.sdp_kernel context defaults / global enables
        for name in (
            "flash_sdp_enabled",
            "mem_efficient_sdp_enabled",
            "math_sdp_enabled",
            "cudnn_sdp_enabled",
        ):
            fn = getattr(torch.backends.cuda, name, None)
            if callable(fn):
                try:
                    flags[name] = bool(fn())
                except Exception as exc:
                    flags[name] = f"error:{exc}"
    except Exception as exc:
        flags["sdp_query_error"] = str(exc)
    return flags


def capture_ab_repro_fingerprint(
    *,
    model: nn.Module,
    device: torch.device,
    decoder_kwargs: Optional[Dict[str, Any]] = None,
    trace: Any = None,
    seed_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture pre-forward reproducibility state for A/B comparison."""
    torch_rng = None
    cuda_rng = None
    try:
        torch_rng = torch.get_rng_state()
    except Exception as exc:
        torch_rng = f"error:{exc}"
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            cuda_rng = torch.cuda.get_rng_state_all()
        except Exception as exc:
            cuda_rng = f"error:{exc}"

    det_alg = None
    det_alg_error = None
    try:
        det_alg = bool(torch.are_deterministic_algorithms_enabled())
    except Exception as exc:
        det_alg_error = str(exc)

    warn_only = None
    try:
        warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    except Exception:
        warn_only = None

    tf32_matmul = None
    tf32_cudnn = None
    try:
        tf32_matmul = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    try:
        tf32_cudnn = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass

    cudnn_benchmark = None
    cudnn_deterministic = None
    try:
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    except Exception:
        pass

    return {
        "seed_report": seed_report,
        "torch_cpu_rng_hash": rng_state_hash(torch_rng),
        "cuda_rng_hash": rng_state_hash(cuda_rng),
        "deterministic_algorithms_enabled": det_alg,
        "deterministic_algorithms_warn_only": warn_only,
        "deterministic_algorithms_error": det_alg_error,
        "tf32_matmul_allow": tf32_matmul,
        "tf32_cudnn_allow": tf32_cudnn,
        "cudnn_benchmark": cudnn_benchmark,
        "cudnn_deterministic": cudnn_deterministic,
        "model_training": bool(model.training),
        "lora_dropout": _lora_dropout_report(model),
        "trainable_param_checksums_head": _trainable_param_checksums(model, limit=16),
        "fixture_tensor_checksums": _fixture_tensor_checksums(decoder_kwargs, trace),
        "sdpa_backend_flags": _sdpa_backend_flags(),
        "device": str(device),
    }


def compare_ab_repro_fingerprints(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare A/B reproducibility fingerprints field-by-field."""
    keys = [
        "torch_cpu_rng_hash",
        "cuda_rng_hash",
        "deterministic_algorithms_enabled",
        "deterministic_algorithms_warn_only",
        "tf32_matmul_allow",
        "tf32_cudnn_allow",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "model_training",
        "sdpa_backend_flags",
        "fixture_tensor_checksums",
        "trainable_param_checksums_head",
        "lora_dropout",
    ]
    diffs: List[Dict[str, Any]] = []
    matches: Dict[str, bool] = {}
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        ok = av == bv
        matches[key] = ok
        if not ok:
            diffs.append({"field": key, "a": av, "b": bv})
    seed_a = (a.get("seed_report") or {}).get("seed")
    seed_b = (b.get("seed_report") or {}).get("seed")
    seed_match = seed_a == seed_b and seed_a is not None
    return {
        "seed_match": seed_match,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "field_matches": matches,
        "all_logged_fields_match": bool(seed_match and all(matches.values())),
        "num_diff_fields": len(diffs),
        "diff_fields": [d["field"] for d in diffs],
        "diffs_head": diffs[:8],
    }
