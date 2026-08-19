"""Experimental exact layer checkpointing for differentiable cached replay.

This module is intentionally not wired into production training.  It exists for
the NTP replay exactness and memory oracle.  The key difference from ordinary
layer checkpointing is that backward recomputation receives immutable K/V
tensors and mutates a fresh local ``DynamicCache``.  The live outer cache is
updated exactly once, during the original forward.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache

from rl.replay_memory import _decoder_layers, assert_model_eval_for_replay


def _assign_dynamic_layer(layer, key: torch.Tensor, value: torch.Tensor) -> None:
    """Assign an already-computed cache value without adding a second cat op."""
    layer.keys = key
    layer.values = value
    layer.dtype = key.dtype
    layer.device = key.device
    layer.is_initialized = True


def _ensure_layer(cache: DynamicCache, layer_index: int):
    while len(cache.layers) <= layer_index:
        if cache.layer_class_to_replicate is None:
            raise RuntimeError("DynamicCache cannot create the requested layer")
        cache.layers.append(cache.layer_class_to_replicate())
    return cache.layers[layer_index]


def _prior_layer_kv(
    cache: DynamicCache, layer_index: int
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    if len(cache.layers) <= layer_index:
        return None
    layer = cache.layers[layer_index]
    if not bool(getattr(layer, "is_initialized", False)) or layer.get_seq_length() == 0:
        return None
    return layer.keys, layer.values


@contextmanager
def functional_kv_layer_checkpointing(
    model,
    *,
    enabled: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Checkpoint decoder layers without mutating saved cache state twice.

    Only the production oracle contract is accepted: eval mode, SDPA,
    ``use_cache=True``, no attention outputs, and a ``DynamicCache``.  K/V
    tensors remain differentiable explicit checkpoint inputs and outputs.
    """
    report: Dict[str, Any] = {
        "enabled": bool(enabled),
        "use_reentrant": False,
        "preserve_rng_state": True,
        "functional_kv": True,
        "wrapped_layer_count": 0,
        "checkpoint_calls": 0,
        "checkpoint_calls_by_layer": {},
        "recompute_uses_fresh_cache": True,
    }
    if not enabled:
        yield report
        return

    assert_model_eval_for_replay(model)
    layers = _decoder_layers(model)
    originals: List[Tuple[torch.nn.Module, Any]] = []

    def wrap(layer: torch.nn.Module, layer_index: int):
        original = layer.forward
        cache_config = getattr(layer.self_attn, "config", None)
        if cache_config is None:
            raise RuntimeError(f"decoder layer {layer_index} has no attention config")

        def forward_with_checkpoint(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            if kwargs:
                raise RuntimeError(
                    f"checkpoint oracle received unsupported layer kwargs: {sorted(kwargs)}"
                )
            if not bool(use_cache):
                raise RuntimeError("exact KV checkpoint oracle requires use_cache=True")
            if bool(output_attentions):
                raise RuntimeError("exact KV checkpoint oracle requires output_attentions=False")
            if not isinstance(past_key_value, DynamicCache):
                raise RuntimeError(
                    "exact KV checkpoint oracle requires transformers.DynamicCache"
                )
            if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                raise RuntimeError("attention mask must be a tensor or the exact None sentinel")
            if not isinstance(position_ids, torch.Tensor):
                raise RuntimeError("exact KV checkpoint oracle requires explicit position IDs")

            prior = _prior_layer_kv(past_key_value, layer_index)

            mask_is_none = attention_mask is None

            def run_layer(hidden, positions, *remaining_tensors):
                if mask_is_none:
                    mask = None
                    prior_tensors = remaining_tensors
                else:
                    mask = remaining_tensors[0]
                    prior_tensors = remaining_tensors[1:]
                local_cache = DynamicCache(config=cache_config)
                if prior_tensors:
                    if len(prior_tensors) != 2:
                        raise RuntimeError("expected explicit prior key and value tensors")
                    local_layer = _ensure_layer(local_cache, layer_index)
                    _assign_dynamic_layer(
                        local_layer, prior_tensors[0], prior_tensors[1]
                    )
                outputs = original(
                    hidden,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_value=local_cache,
                    output_attentions=False,
                    use_cache=True,
                )
                local_layer = _ensure_layer(local_cache, layer_index)
                return outputs[0], local_layer.keys, local_layer.values

            checkpoint_inputs = (hidden_states, position_ids)
            if attention_mask is not None:
                checkpoint_inputs += (attention_mask,)
            if prior is not None:
                checkpoint_inputs += prior
            next_hidden, next_key, next_value = checkpoint(
                run_layer,
                *checkpoint_inputs,
                use_reentrant=False,
                preserve_rng_state=True,
                determinism_check="default",
            )
            outer_layer = _ensure_layer(past_key_value, layer_index)
            _assign_dynamic_layer(outer_layer, next_key, next_value)
            report["checkpoint_calls"] += 1
            calls_by_layer = report["checkpoint_calls_by_layer"]
            calls_by_layer[layer_index] = calls_by_layer.get(layer_index, 0) + 1
            return next_hidden, past_key_value

        return forward_with_checkpoint

    try:
        for index, layer in enumerate(layers):
            originals.append((layer, layer.forward))
            layer.forward = wrap(layer, index)  # type: ignore[method-assign]
        report["wrapped_layer_count"] = len(layers)
        yield report
    finally:
        for layer, original in originals:
            layer.forward = original  # type: ignore[method-assign]
