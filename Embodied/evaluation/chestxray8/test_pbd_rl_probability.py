#!/usr/bin/env python3
"""Mandatory probability tests for stochastic native PBD-RL decoding."""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import torch

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.pbd_rl import (  # noqa: E402
    BlockTrace,
    PBDSamplingConfig,
    SlotTrace,
    build_filtered_categorical,
    sample_pbd_block,
    score_pbd_block,
)
from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.policy_state import (  # noqa: E402
    OldPolicyController,
    PolicySnapshot,
    use_policy_snapshot,
)


TOKEN_IDS = {
    "box_start_token_id": 10,
    "box_end_token_id": 11,
    "coord_start_token_id": 20,
    "coord_end_token_id": 25,
    "default_mask_token_id": 12,
    "im_end_token_id": 13,
}


def test_coordinate_support_and_normalization() -> None:
    logits = torch.linspace(-2, 2, 64)
    config = PBDSamplingConfig()
    dist = build_filtered_categorical(
        logits,
        history_ids=[],
        config=config,
        allowed_token_ids=range(20, 26),
    )
    assert dist.token_ids.tolist() == [20, 21, 22, 23, 24, 25]
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0), atol=1e-7)


def test_top_k_is_applied_after_coordinate_restriction() -> None:
    logits = torch.zeros(64)
    logits[0] = 1000  # Must never enter coordinate top-k.
    logits[22] = 3
    logits[24] = 2
    logits[25] = 1
    config = PBDSamplingConfig(top_k=2)
    dist = build_filtered_categorical(
        logits,
        history_ids=[],
        config=config,
        allowed_token_ids=range(20, 26),
    )
    assert set(dist.token_ids.tolist()) == {22, 24}
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0), atol=1e-7)


def test_top_p_keeps_boundary_token_and_normalizes() -> None:
    logits = torch.full((8,), -20.0)
    logits[:3] = torch.log(torch.tensor([0.5, 0.3, 0.2]))
    dist = build_filtered_categorical(
        logits,
        history_ids=[],
        config=PBDSamplingConfig(top_p=0.6),
    )
    assert set(dist.token_ids.tolist()) == {0, 1}
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0), atol=1e-7)


def test_parallel_slots_use_only_committed_prefix_for_repetition() -> None:
    logits = torch.zeros(6, 64)
    logits[0, 10] = 10
    logits[1:5, 20:26] = 1
    logits[5, 11] = 10
    config = PBDSamplingConfig(repetition_penalty=2.0)
    _, actions, slots, _ = sample_pbd_block(
        logits,
        history_ids=[20],
        token_ids=TOKEN_IDS,
        config=config,
        generator=torch.Generator().manual_seed(5),
    )
    block = BlockTrace(
        block_index=0,
        prefix_length=1,
        cache_length_before=0,
        cache_length_after=1,
        block_type="box",
        position_ids=list(range(6)),
        input_window_ids=[20, 12, 12, 12, 12, 12],
        action_token_ids=actions,
        slots=slots,
    )
    replay, _ = score_pbd_block(
        logits,
        block,
        history_ids=[20],
        token_ids=TOKEN_IDS,
        config=config,
    )
    assert math.isclose(float(replay), block.old_log_prob, abs_tol=1e-6)


def test_sampled_box_tokens_are_recorded_actions() -> None:
    logits = torch.full((6, 64), -20.0)
    logits[0, TOKEN_IDS["box_start_token_id"]] = 10
    expected_coords = [21, 22, 23, 24]
    for slot, token_id in enumerate(expected_coords, start=1):
        logits[slot, token_id] = 10
    logits[5, TOKEN_IDS["box_end_token_id"]] = 10
    config = PBDSamplingConfig()
    block_type, actions, slots, stopped = sample_pbd_block(
        logits,
        history_ids=[1, 2, 3],
        token_ids=TOKEN_IDS,
        config=config,
        generator=torch.Generator().manual_seed(7),
    )
    assert block_type == "box"
    assert not stopped
    assert actions == [10, 21, 22, 23, 24, 11]
    assert actions == [slot.action_token_id for slot in slots]
    assert [slot.support_kind for slot in slots] == [
        "full",
        "coordinate",
        "coordinate",
        "coordinate",
        "coordinate",
        "box_end",
    ]


def test_replay_log_probability_matches_recorded_probability() -> None:
    torch.manual_seed(11)
    logits = torch.randn(6, 64, requires_grad=True)
    config = PBDSamplingConfig()
    _, actions, slots, _ = sample_pbd_block(
        logits.detach(),
        history_ids=[5, 6],
        token_ids=TOKEN_IDS,
        config=config,
        generator=torch.Generator().manual_seed(19),
        force_box_block=True,
    )
    block = BlockTrace(
        block_index=0,
        prefix_length=2,
        cache_length_before=0,
        cache_length_after=2,
        block_type="box",
        position_ids=list(range(6)),
        input_window_ids=[6, 12, 12, 12, 12, 12],
        action_token_ids=actions,
        slots=slots,
    )
    replay, per_slot = score_pbd_block(
        logits,
        block,
        history_ids=[5, 6],
        token_ids=TOKEN_IDS,
        config=config,
    )
    assert math.isclose(
        float(replay.detach()), block.old_log_prob, rel_tol=0, abs_tol=1e-5
    )
    replay.backward()
    assert logits.grad is not None
    for slot in slots:
        assert int(slot.action_token_id) in range(64)
    assert len(per_slot) == len(actions)


def test_geometry_is_not_postprocessed() -> None:
    logits = torch.full((6, 64), -100.0)
    logits[0, 10] = 20
    # Native xyxy tokens deliberately encode x1>x2 and y1>y2.
    for slot, token in enumerate([25, 25, 20, 20], start=1):
        logits[slot, token] = 20
    logits[5, 11] = 20
    _, actions, _, _ = sample_pbd_block(
        logits,
        history_ids=[],
        token_ids=TOKEN_IDS,
        config=PBDSamplingConfig(),
        generator=torch.Generator().manual_seed(0),
    )
    assert actions == [10, 25, 25, 20, 20, 11]


def test_rl_path_has_no_deterministic_decoder_calls() -> None:
    from rl import pbd_rl

    source = inspect.getsource(pbd_rl)
    assert "decode_bbox_avg(" not in source
    assert "decode_ref(" not in source
    assert "handle_pattern(" not in source


def test_initial_policy_ratio_and_clipping() -> None:
    old = torch.tensor(-7.5)
    current = old.clone().requires_grad_(True)
    ratio = torch.exp(current - old)
    assert torch.equal(ratio, torch.tensor(1.0))
    epsilon = 0.2
    advantage = torch.tensor(2.0)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    objective = torch.minimum(ratio * advantage, clipped * advantage)
    assert torch.equal(objective, torch.tensor(2.0))


def test_grpo_loss_is_policy_only_and_initial_ratios_are_one() -> None:
    old = torch.tensor([-3.0, -4.0, -5.0, -6.0])
    current = old.clone().requires_grad_(True)
    advantages = group_relative_advantages([0.0, 1.0, 0.5, 1.0])
    loss = grpo_clipped_loss(current, old, advantages, clip_epsilon=0.2)
    assert torch.allclose(torch.exp(current - old), torch.ones(4))
    assert torch.isfinite(loss)
    loss.backward()
    assert current.grad is not None
    source = inspect.getsource(grpo_clipped_loss).lower()
    for forbidden in ("cross_entropy", "labels", "bbox_ce", "coordinate"):
        assert forbidden not in source


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.layer = torch.nn.Module()
        self.language_model.layer.lora_A = torch.nn.Parameter(torch.ones(2, 2))
        self.language_model.layer.base = torch.nn.Parameter(torch.full((2, 2), 9.0))
        self.mlp1 = torch.nn.Linear(2, 2, bias=False)


def test_old_policy_sync_copies_only_lora_and_projector() -> None:
    current = _ToyPolicy()
    old = _ToyPolicy()
    with torch.no_grad():
        current.language_model.layer.lora_A.fill_(3)
        current.language_model.layer.base.fill_(4)
        current.mlp1.weight.fill_(5)
        old.language_model.layer.lora_A.fill_(1)
        old.language_model.layer.base.fill_(9)
        old.mlp1.weight.fill_(1)
    controller = OldPolicyController(current, old, sync_interval=1)
    controller.synchronize_after_optimizer_step(1)
    assert torch.equal(
        old.language_model.layer.lora_A, current.language_model.layer.lora_A
    )
    assert torch.equal(old.mlp1.weight, current.mlp1.weight)
    assert torch.all(old.language_model.layer.base == 9)
    controller.assert_frozen()


def test_snapshot_requires_lora_and_projector() -> None:
    snapshot = PolicySnapshot.capture(_ToyPolicy(), optimizer_step=0)
    assert any("lora_" in name for name in snapshot.tensors)
    assert any(name.startswith("mlp1.") for name in snapshot.tensors)
    assert all(
        "base" not in name and not name.startswith("vision_model.")
        for name in snapshot.tensors
    )


def test_shared_base_snapshot_swap_restores_current_policy() -> None:
    model = _ToyPolicy()
    old = PolicySnapshot.capture(model, optimizer_step=0)
    with torch.no_grad():
        model.language_model.layer.lora_A.fill_(7)
        model.mlp1.weight.fill_(8)
    with use_policy_snapshot(model, old):
        assert torch.all(model.language_model.layer.lora_A == 1)
        assert not torch.all(model.mlp1.weight == 8)
    assert torch.all(model.language_model.layer.lora_A == 7)
    assert torch.all(model.mlp1.weight == 8)


def test_native_bbox_token_roundtrip() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/LocateAnything-3B",
        revision="c32291ca5e996f5a7a485845b4f57a233936bba0",
        trust_remote_code=True,
    )
    ids = [
        tokenizer.convert_tokens_to_ids("<box>"),
        tokenizer.convert_tokens_to_ids("<10>"),
        tokenizer.convert_tokens_to_ids("<20>"),
        tokenizer.convert_tokens_to_ids("<30>"),
        tokenizer.convert_tokens_to_ids("<40>"),
        tokenizer.convert_tokens_to_ids("</box>"),
    ]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert text == "<box><10><20><30><40></box>"
    assert tokenizer.encode(text, add_special_tokens=False) == ids


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PBD-RL PROBABILITY TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
