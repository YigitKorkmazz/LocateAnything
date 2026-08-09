#!/usr/bin/env python3
"""CPU unit tests for Priority-B selective saved-tensor offload helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.lora_grad_diagnostics import truncate_trace_scored_blocks  # noqa: E402
from rl.nan_grad_diagnostics import (  # noqa: E402
    aggregate_isfinite_flags,
    analyze_grad_tensor,
    assert_trainable_grads_finite,
)
from rl.short_ab_nan_compare import (  # noqa: E402
    _iter_decoder_layer_modules,
    interpret_ab_cd,
    restore_trainable_init,
    run_first_mlp_lora_sanity,
    snapshot_state_dict,
    snapshot_trainable_init,
)
from rl.short_sequence_fixture import build_short_rollout_trace  # noqa: E402
from rl.nan_grad_diagnostics import compare_grad_dicts  # noqa: E402
from rl.selective_saved_tensor_offload import (  # noqa: E402
    SelectiveSavedTensorOffload,
    _clear_confirmed_attn_registry,
    classify_attention_protection,
    cuda_storage_device_index,
    matches_old_stride1_only_rule,
    predict_threshold_outcomes,
    push_module_context,
    pop_module_context,
    recommend_threshold,
    register_confirmed_attn_tensor,
    sample_tensor_values,
    selective_saved_tensor_offload_context,
)


def test_cuda_device_metadata_to_storage_cuda_index() -> None:
    assert cuda_storage_device_index(torch.device("cuda:0")) == 0
    assert cuda_storage_device_index(torch.device("cuda:1")) == 1
    assert cuda_storage_device_index(torch.device("cuda", 0)) == 0
    if torch.cuda.is_available():
        idx = cuda_storage_device_index(torch.device("cuda"))
        assert idx == int(torch.cuda.current_device())
    try:
        cuda_storage_device_index(torch.device("cpu"))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_stride1_alone_does_not_protect() -> None:
    need = 10122 * 8 + 1465 * 7 + 8
    base = torch.zeros(need)
    weird = base.as_strided((1, 8, 7, 1465), (10122 * 8, 10122, 1465, 1))
    assert matches_old_stride1_only_rule(weird)
    protect, reason, kind = classify_attention_protection(
        weird, module_context="model.layers.0.mlp"
    )
    assert protect is False
    assert kind == ""
    assert "stride1" not in reason


def test_attention_context_qk_bias_protects() -> None:
    bias = torch.zeros(1, 1, 7, 1465)
    protect, reason, kind = classify_attention_protection(
        bias, module_context="model.layers.0.self_attn"
    )
    assert protect is True
    assert kind == "confirmed_attention_bias"
    assert "attention_context" in reason


def test_kv_activation_not_protected_outside_identity() -> None:
    kv = torch.randn(1, 8, 1400, 128)
    protect, _reason, kind = classify_attention_protection(
        kv, module_context="model.layers.0.self_attn"
    )
    assert protect is False
    assert kind == ""


def test_argument_identity_protects() -> None:
    _clear_confirmed_attn_registry()
    mask = torch.ones(1, 64, dtype=torch.bool)
    register_confirmed_attn_tensor(mask)
    protect, reason, kind = classify_attention_protection(
        mask, module_context="not_attention"
    )
    assert protect is True
    assert kind == "confirmed_attention_mask"
    assert "identity" in reason


def test_shared_storage_view_restore_preserves_stride() -> None:
    # Default: no storage-pointer dedupe — each save event is independent.
    mgr = SelectiveSavedTensorOffload(
        threshold_bytes=64,
        pin_memory=False,
        protect_attn_bias=False,
        verify_unpack_values=True,
        allow_storage_dedup=False,
        offload_devices=("cpu", "cuda"),
    )
    base = torch.randn(4, 64)
    v_a = base[:, :32]
    v_b = base[:, 32:]
    pa = mgr.pack_hook(v_a)
    pb = mgr.pack_hook(v_b)
    assert pa.kind == "offload" and pb.kind == "offload"
    assert pa.save_id != pb.save_id
    assert pa.storage_id != pb.storage_id
    assert mgr.stats.storage_dedup_hits == 0
    assert mgr.stats.unique_save_events == 2
    ra = mgr.unpack_hook(pa)
    rb = mgr.unpack_hook(pb)
    assert torch.equal(ra, v_a) and torch.equal(rb, v_b)
    assert tuple(ra.stride()) == tuple(v_a.stride())
    assert mgr.stats.unpack_value_mismatches == 0
    del ra, rb
    assert mgr.stats.peak_live_unpacked_storages >= 1

    # Optional dedupe path (unsafe across allocator reuse): shared storage_id.
    mgr_dedup = SelectiveSavedTensorOffload(
        threshold_bytes=64,
        pin_memory=False,
        protect_attn_bias=False,
        verify_unpack_values=True,
        allow_storage_dedup=True,
        offload_devices=("cpu", "cuda"),
    )
    base2 = torch.randn(4, 64)
    va2 = base2[:, :32]
    vb2 = base2[:, 32:]
    pa2 = mgr_dedup.pack_hook(va2)
    pb2 = mgr_dedup.pack_hook(vb2)
    assert pa2.storage_id == pb2.storage_id
    assert mgr_dedup.stats.storage_dedup_hits == 1
    ra2 = mgr_dedup.unpack_hook(pa2)
    rb2 = mgr_dedup.unpack_hook(pb2)
    assert torch.equal(ra2, va2) and torch.equal(rb2, vb2)


def test_dry_run_stride_only_reclassification_and_threshold_predict() -> None:
    _clear_confirmed_attn_registry()
    push_module_context("model.layers.0.mlp")
    try:
        mgr = SelectiveSavedTensorOffload(
            threshold_bytes=1024,
            pin_memory=False,
            protect_attn_bias=True,
            verify_unpack_values=False,
            offload_devices=("cpu", "cuda"),
        )
        # Non-qk activation with misaligned stride(1), not in attention context.
        # Max index ≈ (7+63)*10123 + 31 => need ~70*10123 elements.
        storage_elems = 10123 * 72
        base = torch.zeros(storage_elems)
        weird = base.as_strided((8, 64, 32), (10123, 10123, 1))
        assert matches_old_stride1_only_rule(weird)
        # k=32 <= 256 => not attention q/k semantics even if context were attn.
        packed = mgr.pack_hook(weird)
        assert packed.kind == "offload", getattr(packed, "reason", packed)
        report = mgr.classification_report()
        assert report["old_stride1_only_now_offloaded_tensors"] == 1
        assert report["old_stride1_only_now_offloaded_unique_storage_bytes"] > 0
        preds = predict_threshold_outcomes(
            mgr.pack_catalog, thresholds=(1 << 20, 512 << 10, 256 << 10)
        )
        assert len(preds) == 3
        rec = recommend_threshold(preds)
        assert rec["threshold_bytes"] in {1 << 20, 512 << 10, 256 << 10}
    finally:
        pop_module_context()


def test_allocation_free_sample_and_unpack_verify() -> None:
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    vals, idxs, finite = sample_tensor_values(t, max_samples=8)
    assert len(vals) == len(idxs) == len(finite)
    assert all(finite)
    # Spot-check first/last via multi-index path (no flatten required).
    assert vals[0] == float(t[0, 0, 0].item())

    mgr = SelectiveSavedTensorOffload(
        threshold_bytes=16,
        pin_memory=False,
        protect_attn_bias=False,
        verify_unpack_values=True,
        offload_devices=("cpu", "cuda"),
    )
    packed = mgr.pack_hook(t)
    restored = mgr.unpack_hook(packed)
    assert torch.equal(restored, t)
    assert mgr.stats.unpack_value_mismatches == 0
    assert mgr.first_nonfinite_unpack is None


def test_truncate_trace_keeps_at_least_three_scored_blocks() -> None:
    # Minimal duck-typed trace matching truncate_trace_scored_blocks expectations.
    from types import SimpleNamespace

    def _block(scored: bool, tokens: list[int]):
        return SimpleNamespace(scored_for_grpo=scored, action_token_ids=tokens)

    class FakeTrace:
        def __init__(self):
            self.prompt_token_ids = [1, 2, 3]
            self.generated_token_ids = [10, 11, 12, 13, 14, 15]
            self.blocks = [
                _block(True, [10]),
                _block(False, [11]),
                _block(True, [12]),
                _block(True, [13]),
                _block(True, [14]),
                _block(True, [15]),
            ]
            self.sampling = SimpleNamespace(block_size=4)
            self.decoder_path = "hybrid"
            self.reward_branch = "none"
            self.committed_final_box_norm_1000 = None
            self.has_unambiguous_committed_box = False
            self.fallback_triggered = False
            self.rejected_pbd_proposals = []
            self.decoded_text = None
            self.stopped_on_eos = False
            self.truncated = False

    # type(trace)(...) constructor: FakeTrace() takes no args matching dataclass -
    # truncate uses type(trace)(**fields). Provide a compatible constructor.
    class Trace:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    base = FakeTrace()
    # Rebuild as Trace so truncate can construct a copy.
    full = Trace(
        prompt_token_ids=base.prompt_token_ids,
        generated_token_ids=base.generated_token_ids,
        blocks=base.blocks,
        sampling=base.sampling,
        stopped_on_eos=False,
        truncated=False,
        decoded_text=None,
        decoder_path="hybrid",
        reward_branch="none",
        committed_final_box_norm_1000=None,
        has_unambiguous_committed_box=False,
        fallback_triggered=False,
        rejected_pbd_proposals=[],
    )
    short, report = truncate_trace_scored_blocks(full, 3)
    assert report["truncated"] is True
    assert report["kept_scored_blocks"] == 3
    assert sum(1 for b in short.blocks if b.scored_for_grpo) == 3


def test_interpret_ab_cd_notes() -> None:
    def _run(finite: bool) -> dict:
        return {
            "backward_status": "ok",
            "trainable_grad_report": {"all_trainable_grads_finite": finite},
        }

    runs = {
        "A_no_offload_bf16": _run(True),
        "B_selective_offload_bf16": _run(False),
        "C_no_offload_fp32": _run(True),
        "D_selective_offload_fp32": _run(True),
        "_compare_A_B": {"grads": {"within_tol": False}},
    }
    notes = interpret_ab_cd(runs)
    assert notes["A_bf16_finite"] is True
    assert notes["B_bf16_selective_finite"] is False
    assert any("SDPA" in n or "selective offload" in n for n in notes["notes"])
    assert any("precision-specific" in n for n in notes["notes"])


def test_aggregate_isfinite_flags_one_and_many() -> None:
    # Single bool (must not be passed through all() as a non-iterable mistake).
    assert aggregate_isfinite_flags(True) is True
    assert aggregate_isfinite_flags(False) is False

    # Iterable of per-scalar bools (one current + loss + multiple blocks).
    finite_flags = [True, True, True, True, True]
    assert aggregate_isfinite_flags(finite_flags) is True
    assert aggregate_isfinite_flags([True, False, True]) is False

    # Mix: single bools + iterable of block scalar reports.
    block_reports = [
        {"isfinite": True, "value": -1.0},
        {"isfinite": True, "value": -2.0},
        {"isfinite": False, "value": float("nan")},
    ]
    assert (
        aggregate_isfinite_flags(True, True, block_reports) is False
    )
    assert (
        aggregate_isfinite_flags(
            True,
            True,
            [True, True, True],
        )
        is True
    )


def test_sdpa_nested_identity_hooks_keep_tensor_object() -> None:
    from rl.selective_saved_tensor_offload import (
        SelectiveSavedTensorOffload,
        patch_sdpa_nested_identity_hooks,
        selective_saved_tensor_offload_context,
    )
    import torch.nn.functional as F

    mgr = SelectiveSavedTensorOffload(
        threshold_bytes=1,
        pin_memory=False,
        protect_attn_bias=True,
        verify_unpack_values=False,
        offload_devices=("cpu", "cuda"),
    )
    q = torch.randn(1, 1, 4, 8, requires_grad=True)
    k = torch.randn(1, 1, 4, 8, requires_grad=True)
    v = torch.randn(1, 1, 4, 8, requires_grad=True)
    seen_ids = []

    def _spy_pack(t):
        if torch.is_tensor(t):
            seen_ids.append(id(t))
        return t

    with patch_sdpa_nested_identity_hooks(mgr):
        # Manually install spy as nested identity already wraps SDPA.
        out = F.scaled_dot_product_attention(q, k, v)
        loss = out.sum()
        loss.backward()
    assert mgr.stats.sdpa_identity_pack_calls > 0
    assert mgr.stats.sdpa_protected_unique_bytes >= 0
    # Outer selective context also works with nested SDPA patch.
    mgr2 = SelectiveSavedTensorOffload(
        threshold_bytes=1 << 30,
        pin_memory=False,
        offload_devices=("cpu", "cuda"),
    )
    with selective_saved_tensor_offload_context(mgr2, model=None):
        q2 = torch.randn(1, 1, 2, 4, requires_grad=True)
        k2 = torch.randn(1, 1, 2, 4, requires_grad=True)
        v2 = torch.randn(1, 1, 2, 4, requires_grad=True)
        out2 = F.scaled_dot_product_attention(q2, k2, v2)
        out2.sum().backward()
    assert mgr2.stats.sdpa_identity_pack_calls > 0


def test_hard_fail_nonfinite_grads() -> None:
    param = torch.nn.Parameter(torch.ones(3))
    model = torch.nn.Module()
    model.w = param
    param.grad = torch.tensor([1.0, float("nan"), 2.0])
    stats = analyze_grad_tensor(param.grad)
    assert stats["nan_count"] == 1
    assert stats["all_finite"] is False
    try:
        assert_trainable_grads_finite(model)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "refusing optimizer.step" in str(exc)


def test_short_rollout_trace_cache_lengths() -> None:
    prompt = list(range(120))
    trace = build_short_rollout_trace(prompt, num_scored_blocks=3, block_size=6)
    assert trace.decoder_path == "hybrid"
    assert len(trace.prompt_token_ids) == 120
    assert len(trace.generated_token_ids) == 18
    assert sum(1 for b in trace.blocks if b.scored_for_grpo) == 3
    b0, b1, b2 = trace.blocks
    assert b0.prefix_length == 120 and b0.cache_length_before == 0
    assert b0.cache_length_after == 120
    assert b1.prefix_length == 126 and b1.cache_length_before == 120
    assert b2.prefix_length == 132 and b2.cache_length_before == 126
    assert all(len(b.action_token_ids) == 6 for b in trace.blocks)


def test_cosine_similarity_clamped_never_reports_above_one() -> None:
    from rl.nan_grad_diagnostics import _cosine_similarity_clamped

    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 1e-20, -1e-20])
    clamped, raw = _cosine_similarity_clamped(a, a)
    assert clamped == 1.0
    assert raw is not None
    # Near-identical vectors can yield raw cosine slightly > 1 in float32.
    x = torch.ones(1024, dtype=torch.float32)
    y = x * (1.0 + 1e-7)
    clamped2, raw2 = _cosine_similarity_clamped(x, y)
    assert -1.0 <= clamped2 <= 1.0
    assert raw2 is not None
    # Reported fields used for acceptance must be clamped.
    cmp = compare_grad_dicts({"w": x}, {"w": y})
    assert cmp["global_cosine_finite"] is not None
    assert -1.0 <= float(cmp["global_cosine_finite"]) <= 1.0
    assert "global_cosine_raw" in cmp
    assert "min_cosine_raw" in cmp


def test_compare_grad_dicts_zero_mask_and_global_metrics() -> None:
    a = {"lora_A": torch.tensor([1.0, 0.0, 2.0]), "lora_B": torch.tensor([0.5])}
    b = {"lora_A": torch.tensor([1.0, 0.0, 2.0]), "lora_B": torch.tensor([0.5])}
    ok = compare_grad_dicts(a, b)
    assert ok["within_tol"] is True
    assert ok["zero_nonzero_mask_mismatches"] == 0
    assert ok["global_cosine_finite"] is not None
    assert ok["global_cosine_status"] == "both_nonzero"
    bad = compare_grad_dicts(
        a, {"lora_A": torch.tensor([1.0, 1.0, 2.0]), "lora_B": torch.tensor([0.5])}
    )
    assert bad["zero_nonzero_mask_mismatches"] == 1
    assert bad["within_tol"] is False

    z = {"lora_A": torch.zeros(3), "lora_B": torch.zeros(1)}
    zero_cmp = compare_grad_dicts(z, z)
    assert zero_cmp["all_gradients_zero"] is True
    assert zero_cmp["exact_zero_match"] is True
    assert zero_cmp["global_cosine_status"] == "both_zero"
    assert zero_cmp["global_cosine_finite"] is None
    assert zero_cmp["min_cosine_finite"] is None
    assert zero_cmp["within_tol"] is True  # values match; acceptance rejects zero
    assert zero_cmp["num_both_zero_tensors"] == 2


def test_oracle_block_loss_ratio_one_nonzero() -> None:
    from rl.short_ab_nan_compare import build_oracle_block_grpo_loss

    blocks = [
        torch.tensor(-2.0, requires_grad=True),
        torch.tensor(-1.5, requires_grad=True),
        torch.tensor(-3.0, requires_grad=True),
    ]
    loss, meta = build_oracle_block_grpo_loss(
        blocks, [1.0, -0.7, 1.5], clip_epsilon=0.2
    )
    assert meta["max_ratio_abs_err_from_one"] < 1e-6
    assert abs(float(loss.detach())) > 1e-8
    loss.backward()
    assert all(b.grad is not None and float(b.grad.abs()) > 0 for b in blocks)


def test_interpret_degenerate_zero_grads_note() -> None:
    runs = {
        "A_no_offload_bf16": {
            "backward_status": "ok",
            "trainable_grad_report": {"all_trainable_grads_finite": True},
        },
        "B_selective_offload_bf16": {
            "backward_status": "ok",
            "trainable_grad_report": {"all_trainable_grads_finite": True},
        },
        "_compare_A_B": {
            "grads": {
                "within_tol": True,
                "all_gradients_zero": True,
                "exact_zero_match": True,
                "global_cosine_status": "both_zero",
                "max_abs_error_finite": 0.0,
                "nan_inf_mask_mismatches": 0,
                "zero_nonzero_mask_mismatches": 0,
            }
        },
    }
    notes = interpret_ab_cd(runs)["notes"]
    assert any("degenerate all-zero gradient comparison" in n for n in notes)
    assert not any("gradients differ" in n for n in notes)


def test_decoder_layer_discovery_and_lora_sanity_skip() -> None:
    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Linear(2, 2)

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.model = torch.nn.Module()
            self.language_model.model.layers = torch.nn.ModuleList(
                [Layer(), Layer(), Layer()]
            )

    model = Toy()
    found = _iter_decoder_layer_modules(model)
    assert [i for i, _n, _m in found] == [0, 1, 2]
    sanity = run_first_mlp_lora_sanity(model, torch.device("cpu"))
    assert sanity["status"] == "skip"


def test_unique_save_id_and_full_unpack_verify() -> None:
    mgr = SelectiveSavedTensorOffload(
        threshold_bytes=8,
        pin_memory=False,
        protect_attn_bias=False,
        verify_unpack_values=False,
        full_unpack_verify=True,
        allow_storage_dedup=False,
        offload_devices=("cpu", "cuda"),
    )
    a = torch.arange(32, dtype=torch.float32)
    b = torch.arange(32, dtype=torch.float32) + 100
    pa = mgr.pack_hook(a)
    pb = mgr.pack_hook(b)
    assert pa.save_id == 1 and pb.save_id == 2
    assert pa.cpu_checksum is not None and pb.cpu_checksum is not None
    assert pa.reference_cpu is not None
    ra = mgr.unpack_hook(pa)
    rb = mgr.unpack_hook(pb)
    assert torch.equal(ra, a) and torch.equal(rb, b)
    assert mgr.stats.full_unpack_value_mismatches == 0
    assert mgr.first_full_unpack_mismatch is None
    assert len(mgr.full_verify_log) == 2


def test_normalize_ab_by_a12_baseline() -> None:
    from rl.inprocess_short_fixture_oracle import (
        interpret_inprocess_oracle,
        normalize_ab_by_a12_baseline,
    )

    a12 = {
        "within_tol": False,
        "global_rel_l2_error_finite": 0.01,
        "max_abs_error_finite": 0.1,
    }
    ab_ok = {
        "within_tol": False,
        "global_rel_l2_error_finite": 0.0102,
        "max_abs_error_finite": 0.101,
    }
    ab_bad = {
        "within_tol": False,
        "global_rel_l2_error_finite": 0.05,
        "max_abs_error_finite": 0.5,
    }
    n_ok = normalize_ab_by_a12_baseline(ab_ok, a12)
    n_bad = normalize_ab_by_a12_baseline(ab_bad, a12)
    assert n_ok["ab_no_worse_than_a12_baseline"] is True
    assert n_bad["ab_no_worse_than_a12_baseline"] is False
    assert "nondeterminism" in interpret_inprocess_oracle(
        a12_match=False, ab_match=False, ab_no_worse=True
    )
    assert "selective offload" in interpret_inprocess_oracle(
        a12_match=True, ab_match=False, ab_no_worse=False
    )
    assert "exactness passes" in interpret_inprocess_oracle(
        a12_match=True, ab_match=True, ab_no_worse=True
    )


def test_g3_global_identity_never_offloads() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.lin(x)

    model = Tiny()
    mgr = SelectiveSavedTensorOffload(
        threshold_bytes=1,
        pin_memory=False,
        protect_attn_bias=False,
        allow_storage_dedup=False,
        offload_devices=("cpu", "cuda"),
    )
    x = torch.randn(4, 8, requires_grad=True)
    with selective_saved_tensor_offload_context(
        mgr, model, identity_guard_level="G3"
    ):
        y = model(x)
        y.sum().backward()
    assert mgr.stats.offloaded_tensors == 0
    assert mgr.stats.global_identity_pack_calls > 0


def test_ab_repro_fingerprint_seed_and_compare() -> None:
    from rl.ab_repro_diagnostics import (
        apply_identical_oracle_seed,
        capture_ab_repro_fingerprint,
        compare_ab_repro_fingerprints,
    )

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(3))

        def forward(self, x):
            return x * self.w

    model = Tiny().eval()
    seed_a = apply_identical_oracle_seed(424242)
    fa = capture_ab_repro_fingerprint(
        model=model,
        device=torch.device("cpu"),
        decoder_kwargs={"x": torch.arange(4.0)},
        trace=type("T", (), {"prompt_token_ids": [1, 2], "generated_token_ids": [3]})(),
        seed_report=seed_a,
    )
    seed_b = apply_identical_oracle_seed(424242)
    fb = capture_ab_repro_fingerprint(
        model=model,
        device=torch.device("cpu"),
        decoder_kwargs={"x": torch.arange(4.0)},
        trace=type("T", (), {"prompt_token_ids": [1, 2], "generated_token_ids": [3]})(),
        seed_report=seed_b,
    )
    cmp = compare_ab_repro_fingerprints(fa, fb)
    assert cmp["seed_match"] is True
    assert cmp["field_matches"]["torch_cpu_rng_hash"] is True
    assert cmp["field_matches"]["fixture_tensor_checksums"] is True
    assert cmp["field_matches"]["trainable_param_checksums_head"] is True


def test_mlp_identity_guard_g2_covers_full_mlp() -> None:
    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 16, bias=False)
            self.up_proj = torch.nn.Linear(8, 16, bias=False)
            self.down_proj = torch.nn.Linear(16, 8, bias=False)
            self.act_fn = torch.nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = TinyMLP()

        def forward(self, x):
            return self.mlp(x)

    model = TinyBlock()
    mgr = SelectiveSavedTensorOffload(
        threshold_bytes=1,
        pin_memory=False,
        protect_attn_bias=False,
        allow_storage_dedup=False,
        offload_devices=("cpu", "cuda"),
    )
    x = torch.randn(4, 8, requires_grad=True)
    with selective_saved_tensor_offload_context(
        mgr, model, identity_guard_level="G2"
    ):
        y = model(x)
        y.sum().backward()
    assert mgr.stats.mlp_identity_pack_calls > 0


def test_lora_identity_guard_g1_keeps_branch_resident() -> None:
    class TinyLora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(8, 2, bias=False)}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(2, 8, bias=False)}
            )
            self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})

        def forward(self, x):
            h = self.lora_A["default"](x)
            h = self.lora_dropout["default"](h)
            return self.lora_B["default"](h)

    # Monkeypatch PEFT discovery to treat TinyLora as a LoRA parent.
    import rl.selective_saved_tensor_offload as offload_mod

    model = TinyLora()
    orig_iter = offload_mod._iter_lora_modules

    def _fake_iter(_model):
        return [("layers.0.mlp.down_proj", model)]

    offload_mod._iter_lora_modules = _fake_iter  # type: ignore[assignment]
    try:
        mgr = SelectiveSavedTensorOffload(
            threshold_bytes=1,
            pin_memory=False,
            protect_attn_bias=False,
            allow_storage_dedup=False,
            offload_devices=("cpu", "cuda"),
        )
        x = torch.randn(4, 8, requires_grad=True)
        with selective_saved_tensor_offload_context(
            mgr, model, identity_guard_level="G1"
        ):
            y = model(x)
            y.sum().backward()
        assert mgr.stats.lora_identity_pack_calls > 0
    finally:
        offload_mod._iter_lora_modules = orig_iter  # type: ignore[assignment]


def test_trainable_only_init_snapshot_and_copy_() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(4, 4, bias=False)
            # Mimic PEFT LoRA naming.
            self.lora_A = torch.nn.Parameter(torch.randn(2, 4))
            self.lora_B = torch.nn.Parameter(torch.randn(4, 2))
            self.mlp1 = torch.nn.Linear(4, 4, bias=False)

    model = Tiny()
    for p in model.parameters():
        p.requires_grad_(False)
    model.lora_A.requires_grad_(True)
    model.lora_B.requires_grad_(True)
    # Case B: projector frozen.
    art = snapshot_trainable_init(model, include_projector=False)
    assert art["format"] == "trainable_only_v1"
    assert art["num_tensors"] == 2
    assert art["cpu_bytes"] < 1 << 20  # << 7.2GB; tiny fixture
    assert all("lora_" in n for n in art["tensor_names"])
    assert not any(n.startswith("mlp1.") for n in art["tensor_names"])

    # Mutate live params, then restore via copy_ (storage ptr must stay).
    ptr_a = int(model.lora_A.untyped_storage().data_ptr())
    model.lora_A.data.zero_()
    model.lora_B.data.zero_()
    report = restore_trainable_init(
        model, art, device=torch.device("cpu"), clear_cpu_refs=True
    )
    assert report["num_restored"] == 2
    assert report["storage_pointer_consistent"] is True
    assert int(model.lora_A.untyped_storage().data_ptr()) == ptr_a
    assert not torch.allclose(model.lora_A, torch.zeros_like(model.lora_A))

    try:
        snapshot_state_dict(model)
        raise AssertionError("full-state snapshot must raise")
    except RuntimeError as exc:
        assert "forbidden" in str(exc)

    try:
        restore_trainable_init(
            model, {"format": "full"}, device=torch.device("cpu")
        )
        raise AssertionError("full-state restore must raise")
    except RuntimeError as exc:
        assert "trainable_only_v1" in str(exc)


def main() -> None:
    test_cuda_device_metadata_to_storage_cuda_index()
    test_stride1_alone_does_not_protect()
    test_attention_context_qk_bias_protects()
    test_kv_activation_not_protected_outside_identity()
    test_argument_identity_protects()
    test_shared_storage_view_restore_preserves_stride()
    test_dry_run_stride_only_reclassification_and_threshold_predict()
    test_allocation_free_sample_and_unpack_verify()
    test_truncate_trace_keeps_at_least_three_scored_blocks()
    test_interpret_ab_cd_notes()
    test_aggregate_isfinite_flags_one_and_many()
    test_sdpa_nested_identity_hooks_keep_tensor_object()
    test_hard_fail_nonfinite_grads()
    test_short_rollout_trace_cache_lengths()
    test_cosine_similarity_clamped_never_reports_above_one()
    test_compare_grad_dicts_zero_mask_and_global_metrics()
    test_oracle_block_loss_ratio_one_nonzero()
    test_interpret_degenerate_zero_grads_note()
    test_decoder_layer_discovery_and_lora_sanity_skip()
    test_trainable_only_init_snapshot_and_copy_()
    test_unique_save_id_and_full_unpack_verify()
    test_lora_identity_guard_g1_keeps_branch_resident()
    test_mlp_identity_guard_g2_covers_full_mlp()
    test_g3_global_identity_never_offloads()
    test_ab_repro_fingerprint_seed_and_compare()
    test_normalize_ab_by_a12_baseline()
    print("ALL_SELECTIVE_SAVED_TENSOR_OFFLOAD_TESTS_PASSED")


if __name__ == "__main__":
    main()
