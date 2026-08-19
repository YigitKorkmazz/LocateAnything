"""Static safety contracts for the isolated functional-KV checkpoint oracle."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rl.exact_kv_checkpoint import functional_kv_layer_checkpointing  # noqa: E402
import ntp_checkpoint_replay_oracle as oracle  # noqa: E402


def test_checkpoint_is_nonreentrant_and_functionalizes_mutable_kv():
    source = inspect.getsource(functional_kv_layer_checkpointing)
    assert "use_reentrant=False" in source
    assert "preserve_rng_state=True" in source
    assert "local_cache = DynamicCache" in source
    assert "prior_tensors[0], prior_tensors[1]" in source
    assert "_assign_dynamic_layer(outer_layer, next_key, next_value)" in source
    assert ".detach(" not in source


def test_oracle_preserves_replay_contract_and_never_steps_optimizer():
    source = inspect.getsource(oracle)
    assert "use_cache=True" in source
    assert "legacy_nocache_masks=False" in source
    assert "scaled_loss = loss / 8.0" in source
    assert "scaled_loss.backward()" in source
    assert "optimizer.step" not in source
    assert "detach_kv\": False" in source
    assert "truncated_bptt\": False" in source


def test_production_backend_is_imported_but_default_off_and_current_replay_only():
    production = (HERE / "two_gpu_g4_grpo_multistep_smoke.py").read_text(
        encoding="utf-8"
    )
    assert "from rl.exact_kv_checkpoint import functional_kv_layer_checkpointing" in production
    assert "EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = False" in production
    assert production.count("with functional_kv_layer_checkpointing(") == 1
    old_policy = production.index("with torch.no_grad():\n                old_logps")
    checkpoint = production.index("with functional_kv_layer_checkpointing(")
    backward = production.index("scaled.backward()", checkpoint)
    assert old_policy < checkpoint < backward


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} exact-KV checkpoint static tests passed")
