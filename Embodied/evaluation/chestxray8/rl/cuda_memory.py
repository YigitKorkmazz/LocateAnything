"""CUDA peak-memory logging helpers for memory-safe GRPO training."""

from __future__ import annotations

import gc
from typing import Any, Dict, Optional

import torch


def cuda_available(device: Optional[torch.device] = None) -> bool:
    if device is not None and device.type != "cuda":
        return False
    return torch.cuda.is_available()


def _device_index(device: Optional[torch.device] = None) -> Optional[int]:
    if device is not None and device.index is not None:
        return int(device.index)
    return None


def reset_peak_memory(device: Optional[torch.device] = None) -> None:
    if not cuda_available(device):
        return
    torch.cuda.reset_peak_memory_stats(_device_index(device))


def peak_allocated_mb(device: Optional[torch.device] = None) -> float:
    if not cuda_available(device):
        return 0.0
    return float(torch.cuda.max_memory_allocated(_device_index(device)) / (1024**2))


def current_allocated_mb(device: Optional[torch.device] = None) -> float:
    if not cuda_available(device):
        return 0.0
    return float(torch.cuda.memory_allocated(_device_index(device)) / (1024**2))


def peak_reserved_mb(device: Optional[torch.device] = None) -> float:
    if not cuda_available(device):
        return 0.0
    return float(torch.cuda.max_memory_reserved(_device_index(device)) / (1024**2))


def record_stage_peak(
    stages: Dict[str, Any],
    name: str,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    stats = {
        "peak_allocated_mb": peak_allocated_mb(device),
        "current_allocated_mb": current_allocated_mb(device),
    }
    stages[name] = stats
    return stats


def release_cuda_temporaries(
    *,
    empty_cache: bool = False,
    device: Optional[torch.device] = None,
) -> None:
    """Drop unreachable CUDA tensors.

    Default path only runs ``gc.collect()``. ``empty_cache`` is opt-in and must
    not be used inside per-block replay loops (it does not change math).
    """
    gc.collect()
    if empty_cache and cuda_available(device):
        torch.cuda.empty_cache()
