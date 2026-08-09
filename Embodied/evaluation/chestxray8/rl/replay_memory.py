"""Eval-mode layer checkpointing and SDPA helpers for exact GRPO replay.

Hybrid/PBD RL replay already runs under ``model.eval()``. HF's built-in
``gradient_checkpointing`` only activates when ``self.training`` is True and
would switch LocateAnything onto train-mode block-diff masks. These helpers
checkpoint decoder layers while staying in eval mode so AR/MTP masks,
position IDs, and ``full_trajectory`` scoring stay unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint


def assert_model_eval_for_replay(model) -> None:
    if bool(getattr(model, "training", False)):
        raise RuntimeError(
            "GRPO replay must remain in eval mode (existing scientific path); "
            "refusing train-mode replay"
        )
    language = getattr(model, "language_model", None)
    if language is not None and bool(getattr(language, "training", False)):
        raise RuntimeError(
            "language_model must remain in eval mode during GRPO replay"
        )


def activation_dtype_name(model) -> str:
    dtype = getattr(getattr(model, "language_model", None), "dtype", None)
    if dtype is None:
        dtype = next(model.parameters()).dtype
    return str(dtype).replace("torch.", "")


def assert_low_precision_activations(model) -> str:
    name = activation_dtype_name(model)
    if name not in {"bfloat16", "float16"}:
        raise RuntimeError(
            f"GRPO activations must be bf16/fp16 for memory-safe training, got {name}"
        )
    return name


def _decoder_layers(model) -> List[torch.nn.Module]:
    language = getattr(model, "language_model", model)
    layers = None
    for attr in ("model", "base_model", "transformer"):
        candidate = getattr(language, attr, None)
        if candidate is None:
            continue
        nested = getattr(candidate, "model", None)
        if nested is not None and hasattr(nested, "layers"):
            layers = nested.layers
            break
        if hasattr(candidate, "layers"):
            layers = candidate.layers
            break
    if layers is None and hasattr(language, "layers"):
        layers = language.layers
    if layers is None:
        raise RuntimeError(
            "could not locate decoder layers for eval-mode checkpointing"
        )
    return list(layers)


@contextmanager
def eval_mode_layer_checkpointing(
    model,
    *,
    enabled: bool = True,
    allow_unsafe_mutable_kv_cache: bool = False,
) -> Iterator[None]:
    """Wrap decoder layer forwards with non-reentrant checkpointing in eval mode.

    Production-cached GRPO replay must keep this disabled. Checkpoint recompute
    is not replay-safe when layer forwards mutate ``past_key_values`` / DynamicCache
    while attention masks captured at save-time still reflect the earlier length.
    Enabling requires an explicit opt-in that is never used by the training path.
    """
    if not enabled:
        yield
        return
    if not allow_unsafe_mutable_kv_cache:
        raise RuntimeError(
            "eval-mode layer checkpointing refused: decoder forwards that depend "
            "on mutable past_key_values / KV-cache objects are not checkpoint-safe "
            "unless cache tensors and attention-mask state are immutable explicit "
            "checkpoint inputs. For sequential_production_cached GRPO set "
            "training.gradient_checkpointing=false. Do not silently fall back to "
            "cacheless Bfix / two-pass gradients."
        )
    assert_model_eval_for_replay(model)
    layers = _decoder_layers(model)
    originals: List[Tuple[torch.nn.Module, Any]] = []

    def _wrap(layer: torch.nn.Module):
        original = layer.forward

        def forward_with_checkpoint(*args, **kwargs):
            def run(*run_args, **run_kwargs):
                return original(*run_args, **run_kwargs)

            return checkpoint(
                run,
                *args,
                use_reentrant=False,
                **kwargs,
            )

        return forward_with_checkpoint

    try:
        for layer in layers:
            originals.append((layer, layer.forward))
            layer.forward = _wrap(layer)  # type: ignore[method-assign]
        yield
    finally:
        for layer, original in originals:
            layer.forward = original  # type: ignore[method-assign]


@contextmanager
def math_sdpa_for_grad_replay(enabled: bool = True) -> Iterator[None]:
    """Force math SDPA during gradient-bearing replay (odd-length mask safety)."""
    if not enabled or not torch.cuda.is_available():
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            yield
        return
    except Exception:
        pass
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
    ):
        yield
