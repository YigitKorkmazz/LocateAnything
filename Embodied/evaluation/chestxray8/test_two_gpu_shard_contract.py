"""CPU contract test for the LocateAnything visual-feature decoder wrapper."""

from __future__ import annotations

import torch

from rl.two_gpu_shard import (
    DecoderShardLayout,
    _ExactBF16PeerCopy,
    _bf16_values_via_fp32,
    _copy_across_shards,
    _copy_initialization_value_exact,
    _install_sharded_decoder_forward,
    materialize_shard_position_ids,
)


# These names deliberately mirror helpers read from the original decoder
# forward's globals.  The test verifies that the wrapper copies that contract.
def find_prefix_seq_length_by_pe(position_ids):
    return torch.zeros(position_ids.size(0), dtype=torch.long, device=position_ids.device)


def _prepare_4d_causal_attention_mask(*_args, **_kwargs):
    return None


def update_causal_mask_for_one_gen_window_2d(ids, old, **_kwargs):
    return old


def update_causal_mask_with_pad_non_visible_2d(ids, old, **_kwargs):
    return old


def create_block_diff_mask_by_pe_4d(**_kwargs):
    return None, None


def apply_rotary_pos_emb(q, k, _cos, _sin, _position_ids, _unsqueeze_dim=1):
    return q, k


class _Config:
    output_attentions = False
    output_hidden_states = False
    use_cache = False
    use_return_dict = True
    sliding_window = None


class _SelfAttention(torch.nn.Module):
    def forward(self, hidden_states, **_kwargs):
        return hidden_states


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttention()

    def forward(self, hidden_states, **_kwargs):
        return (hidden_states + 1, None)


class _Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.gradient_checkpointing = False
        self._attn_implementation = "sdpa"
        self.embed_tokens = torch.nn.Embedding(16, 4)
        self.layers = torch.nn.ModuleList([_Layer()])
        self.norm = torch.nn.Identity()
        self.block_size = 6
        self.causal_attn = False
        self.text_mask_token_id = 15
        self.visual_seen = None

    def image_processing(self, input_ids, visual_features, image_token_index):
        self.visual_seen = (visual_features, image_token_index)
        return self.embed_tokens(input_ids)

    # Exact pinned remote-code parameter names, including visual_features.
    def forward(
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
    ):
        raise AssertionError("the original forward must be replaced in this contract test")


def test_patched_decoder_accepts_and_consumes_visual_features_on_cpu() -> None:
    decoder = _Decoder()
    _install_sharded_decoder_forward(
        decoder,
        DecoderShardLayout(torch.device("cpu"), torch.device("cpu"), 1),
    )
    visual = torch.randn(1, 2, 4)
    output = decoder(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        visual_features=visual,
        image_token_index=7,
        use_cache=False,
        return_dict=True,
    )
    assert output.last_hidden_state.shape == (1, 2, 4)
    assert decoder.visual_seen[0] is visual
    assert decoder.visual_seen[1] == 7
    arguments = decoder._chestxray8_two_gpu_last_argument_report
    assert arguments["visual_features"]["shape"] == [1, 2, 4]
    assert arguments["visual_features_consumed"] == "before_layer_0_via_image_processing"


def _assert_independent_position_ownership(values: torch.Tensor) -> None:
    snapshot, early, late = materialize_shard_position_ids(values, "cpu", "cpu")
    assert torch.equal(snapshot, values)
    assert torch.equal(early, snapshot)
    assert torch.equal(late, snapshot)
    pointers = {
        snapshot.untyped_storage().data_ptr(),
        early.untyped_storage().data_ptr(),
        late.untyped_storage().data_ptr(),
    }
    assert len(pointers) == 3
    original_snapshot = snapshot.clone()
    original_late = late.clone()
    early[0, 0] = -999
    assert torch.equal(snapshot, original_snapshot)
    assert torch.equal(late, original_late)


def test_ar_position_ids_are_identical_but_independently_owned() -> None:
    _assert_independent_position_ownership(torch.arange(1478, 1490).unsqueeze(0))


def test_pbd_duplicate_position_pattern_is_preserved_without_arange_conversion() -> None:
    pattern = torch.tensor([[0, 0, 0, 4, 4, 5, 5, 5, 6, 7, 7, 8]], dtype=torch.long)
    snapshot, early, late = materialize_shard_position_ids(pattern, "cpu", "cpu")
    assert torch.equal(snapshot, pattern)
    assert torch.equal(early, pattern)
    assert torch.equal(late, pattern)
    _assert_independent_position_ownership(pattern)


def test_bf16_boundary_transport_is_value_exact_and_differentiable_on_cpu() -> None:
    source = torch.tensor([-39.0, 0.0, 29.125], dtype=torch.bfloat16, requires_grad=True)
    copied = _copy_across_shards(source, torch.device("cpu"))
    assert copied.dtype == torch.bfloat16
    assert torch.equal(copied, source)
    copied.float().sum().backward()
    assert source.grad is not None
    assert torch.equal(source.grad, torch.ones_like(source))


def test_custom_bf16_boundary_backward_is_identity_on_cpu() -> None:
    source = torch.tensor([-7.5, 0.0, 31.25], dtype=torch.bfloat16, requires_grad=True)
    copied = _ExactBF16PeerCopy.apply(source, torch.device("cpu"))
    upstream = torch.tensor([2.0, -3.0, 4.0], dtype=torch.bfloat16)
    copied.backward(upstream)
    assert source.grad is not None
    assert torch.equal(source.grad, upstream)


def test_float32_boundary_backward_is_identity_on_cpu() -> None:
    source = torch.tensor([-7.5, 0.0, 31.25], dtype=torch.float32, requires_grad=True)
    copied = _ExactBF16PeerCopy.apply(source, torch.device("cpu"))
    upstream = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float32)
    copied.backward(upstream)
    assert torch.equal(copied, source)
    assert source.grad is not None
    assert torch.equal(source.grad, upstream)


def test_safe_bf16_initialization_transport_and_lm_head_untie_are_exact() -> None:
    source = torch.tensor(
        [[-39.0, 0.0, 29.125], [float("inf"), float("-inf"), -9984.0]],
        dtype=torch.bfloat16,
    )
    copied = _bf16_values_via_fp32(source, "cpu")
    assert copied.dtype == torch.bfloat16
    assert torch.equal(copied, source)
    assert copied.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    untied_lm_head_weight = torch.nn.Parameter(copied.clone(), requires_grad=False)
    assert torch.equal(untied_lm_head_weight, source)
    assert (
        untied_lm_head_weight.untyped_storage().data_ptr()
        != source.untyped_storage().data_ptr()
    )
    checked, report = _copy_initialization_value_exact(
        source, "cpu", label="lm_head.weight"
    )
    assert torch.equal(checked, source)
    assert report["exact_value_equality"]
    assert report["source_sha256"] == report["target_sha256"]


def test_mask_transport_preserves_negative_infinity_and_large_negative_values() -> None:
    mask = torch.tensor(
        [[0.0, float("-inf"), -9984.0]], dtype=torch.bfloat16, requires_grad=True
    )
    copied = _ExactBF16PeerCopy.apply(mask, torch.device("cpu"))
    assert torch.equal(copied, mask)
    assert torch.isneginf(copied[0, 1])
    assert copied[0, 2].item() == mask[0, 2].item()
    copied.float().sum().backward()
    assert torch.equal(mask.grad, torch.ones_like(mask))
