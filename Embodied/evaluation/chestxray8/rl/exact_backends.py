"""Exact live-cache backend registry and comparison table (thesis path).

Scientific requirement
----------------------
Full live-cache trajectory gradients for all scored PBD and NTP blocks:
``carry_cache=True``, model ``use_cache=True``, ``past_key_values`` **not**
detached, no Bfix, no truncated BPTT.

``truncated_bptt_surrogate`` (stop-grad past) is forward-exact but
gradient-inexact. It must never be the default thesis training backend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Scientific default: live production-cached autograd.
LIVE_PRODUCTION_CACHED = {
    "backend": "live_production_cached_autograd",
    "alias": "sequential_production_cached",
    "carry_cache": True,
    "stop_grad_past_key_values": False,
    "legacy_nocache_masks": False,
    "force_math_sdpa": False,
    "model_use_cache_for_mtp_masks": True,
    "gradient_checkpointing": False,
    "live_cached_autograd": True,
    "thesis_default": False,
    "forward_exact": True,
    "gradient_exact": True,  # by construction vs production live cache
    "status": "single_gpu_scientific_reference_not_feasible_on_24gb",
}

TRUNCATED_BPTT_SURROGATE = {
    "backend": "truncated_bptt_surrogate",
    "alias": "sequential_production_matching_truncated_bp",
    "carry_cache": True,
    "stop_grad_past_key_values": True,
    "legacy_nocache_masks": False,
    "force_math_sdpa": False,
    "model_use_cache_for_mtp_masks": True,
    "gradient_checkpointing": False,
    "live_cached_autograd": False,
    "thesis_default": False,
    "forward_exact": True,
    "gradient_exact": False,
    "status": "surrogate_only_not_thesis_default",
}

SAVE_ON_CPU_CANDIDATE = {
    "backend": "live_production_cached_save_on_cpu",
    "carry_cache": True,
    "stop_grad_past_key_values": False,
    "legacy_nocache_masks": False,
    "force_math_sdpa": False,
    "model_use_cache_for_mtp_masks": True,
    "gradient_checkpointing": False,
    "live_cached_autograd": True,
    "save_on_cpu": True,
    "pin_memory": True,
    "thesis_default": False,
    "forward_exact": True,  # A1: all 17 blocks matched production forward
    "gradient_exact": False,  # A1: backward failed attn_bias stride alignment
    "status": "priority_A_diagnostic_baseline_only",
    "notes": (
        "A1: peak ~12.4GB forward OK, detached_kv=false; backward failed "
        "`attn_bias is not correctly aligned (strideH)`; CPU RSS ~185GB. "
        "Keep only as broad-offload baseline, not a production backend."
    ),
}

SELECTIVE_OFFLOAD_CANDIDATE = {
    "backend": "live_production_cached_selective_saved_tensor_offload",
    "carry_cache": True,
    "stop_grad_past_key_values": False,
    "live_cached_autograd": True,
    "save_on_cpu": False,
    "selective_saved_tensors_hooks": True,
    "protect_attn_bias": True,
    "protect_attn_requires_context_or_identity": True,
    "stride1_standalone_protection": False,
    "threshold_bytes_default": 256 << 10,
    "thesis_default": False,
    "forward_exact": None,
    "gradient_exact": None,
    "status": "priority_B_under_investigation",
    "notes": (
        "Short A/B: no-offload BF16 finite; selective offload failed in "
        "ScaledDotProductEfficientAttentionBackward0. Nested identity "
        "saved_tensors_hooks now keep all SDPA-saved tensors GPU-resident "
        "with original storage (no CPU round-trip)."
    ),
}

IMMUTABLE_CACHE_CKPT_CANDIDATE = {
    "backend": "live_production_cached_immutable_cache_checkpoint",
    "carry_cache": True,
    "stop_grad_past_key_values": False,
    "live_cached_autograd": True,
    "gradient_checkpointing": "immutable_cache_inputs_only",
    "thesis_default": False,
    "forward_exact": None,
    "gradient_exact": None,
    "status": "priority_C_not_implemented",
}

MULTI_GPU_SHARD_CANDIDATE = {
    "backend": "live_production_cached_two_gpu_shard",
    "alias": "two_gpu_live_cached_autograd_18_18",
    "carry_cache": True,
    "stop_grad_past_key_values": False,
    "live_cached_autograd": True,
    "legacy_nocache_masks": False,
    "force_math_sdpa": False,
    "model_use_cache_for_mtp_masks": True,
    "gradient_checkpointing": False,
    "decoder_layers": 36,
    "decoder_split": [18, 18],
    "early_device": "cuda:0",
    "late_device": "cuda:1",
    "thesis_default": True,
    "forward_exact": None,
    "gradient_exact": None,
    "status": "thesis_default_pending_one_trajectory_feasibility",
    "notes": (
        "Autograd-preserving hidden-state copy at decoder layer 17→18; "
        "each layer's live KV remains on its assigned GPU. No saved-tensor "
        "offload, KV detach, Bfix, truncated BPTT, or forced math SDPA."
    ),
}

BFIX_REJECTED = {
    "backend": "bfix_cacheless_full_prefix",
    "carry_cache": False,
    "stop_grad_past_key_values": False,
    "thesis_default": False,
    "forward_exact": False,
    "gradient_exact": False,
    "status": "rejected_A_neq_Bfix",
}


ACCEPTANCE_CRITERIA = [
    "production per-block logp match",
    "total logp match",
    "selected LoRA gradient max/relative error within numerical tolerance",
    "optimizer update match",
    "PPO initialization ratio equals 1",
    "no detached KV",
    "no truncated BPTT",
    "no Bfix",
]


def empty_comparison_row(backend: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "backend": backend.get("backend"),
        "status": backend.get("status"),
        "thesis_default": bool(backend.get("thesis_default", False)),
        "forward_exactness": backend.get("forward_exact"),
        "gradient_exactness": backend.get("gradient_exact"),
        "optimizer_update_exactness": None,
        "peak_gpu_memory_bytes": None,
        "cpu_memory_bytes": None,
        "runtime_seconds": None,
        "feasible_1x24gb": None,
        "feasible_2x24gb": None,
        "detached_kv": bool(backend.get("stop_grad_past_key_values", False)),
        "truncated_bptt": backend.get("backend") == "truncated_bptt_surrogate",
        "bfix": backend.get("backend") == "bfix_cacheless_full_prefix",
        "acceptance_criteria": list(ACCEPTANCE_CRITERIA),
        "passes_acceptance": None,
        "notes": None,
    }


def exact_backend_comparison_table(
    measured: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return the exact-backend comparison table (fill measured cells when known)."""
    measured = measured or {}
    rows = []
    for backend in (
        LIVE_PRODUCTION_CACHED,
        SAVE_ON_CPU_CANDIDATE,
        SELECTIVE_OFFLOAD_CANDIDATE,
        IMMUTABLE_CACHE_CKPT_CANDIDATE,
        MULTI_GPU_SHARD_CANDIDATE,
        TRUNCATED_BPTT_SURROGATE,
        BFIX_REJECTED,
    ):
        row = empty_comparison_row(backend)
        key = str(backend["backend"])
        if key in measured:
            row.update(measured[key])
            row["backend"] = key
        # Known qualitative cells.
        if key == "truncated_bptt_surrogate":
            row["forward_exactness"] = True
            row["gradient_exactness"] = False
            row["optimizer_update_exactness"] = False
            row["feasible_1x24gb"] = True
            row["passes_acceptance"] = False
            row["notes"] = (
                "CPU tiny-model: forward_exact_match but truncated_bptt_surrogate "
                "(later-block grads through earlier KV diverge)."
            )
        if key == "bfix_cacheless_full_prefix":
            row["forward_exactness"] = False
            row["gradient_exactness"] = False
            row["passes_acceptance"] = False
            row["notes"] = "A != Bfix (~0.098); rejected."
        if key == "live_production_cached_autograd":
            row["forward_exactness"] = True
            row["gradient_exactness"] = True
            row["optimizer_update_exactness"] = True
            row["feasible_1x24gb"] = False
            row["feasible_2x24gb"] = None
            row["passes_acceptance"] = None
            row["notes"] = (
                "Scientific reference, but not the thesis execution backend: "
                "Case B live LoRA autograd OOMs / disguised invalid-argument "
                "near 24GB on one RTX 3090."
            )
        if key == "live_production_cached_two_gpu_shard":
            row["feasible_1x24gb"] = False
            row["feasible_2x24gb"] = None
            row["notes"] = MULTI_GPU_SHARD_CANDIDATE.get("notes")
        if key == "live_production_cached_save_on_cpu" and key not in measured:
            row["forward_exactness"] = True
            row["gradient_exactness"] = False
            row["optimizer_update_exactness"] = False
            row["peak_gpu_memory_bytes"] = int(12.4 * (1 << 30))
            row["cpu_memory_bytes"] = int(185 * (1 << 30))
            row["feasible_1x24gb"] = False
            row["passes_acceptance"] = False
            row["notes"] = SAVE_ON_CPU_CANDIDATE.get("notes")
        if (
            key == "live_production_cached_selective_saved_tensor_offload"
            and key not in measured
        ):
            row["notes"] = (
                "Priority-B: custom saved_tensors_hooks; protect attn_bias; "
                "offload large activations with shared-storage dedupe; "
                "restore via set_(storage, offset, size, stride)."
            )
        rows.append(row)
    return rows


SURROGATE_NOT_THESIS_DEFAULT_MSG = (
    "truncated_bptt_surrogate / stop-grad past_key_values is forward-exact but "
    "NOT gradient-equivalent to live production-cached autograd (CPU suite "
    "verdict=truncated_bptt_surrogate). It must not be used as the default "
    "thesis training backend. Scientific requirement: full live-cache "
    "trajectory gradients for all scored PBD/NTP blocks (no detached KV, "
    "no Bfix). Investigate save_on_cpu / selective offload / immutable-cache "
    "checkpointing / two-GPU live decoder sharding before training."
)
