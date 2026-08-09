#!/usr/bin/env python3
"""Crafted tests: Hybrid NTP fallback gate vs production handle_pattern."""

from __future__ import annotations

import sys
from pathlib import Path

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.hybrid_rl import (  # noqa: E402
    classify_hybrid_mtp_proposal,
    forced_pbd_box_proposal_can_emit_error_box,
    sample_hybrid_mtp_block,
)
from rl.pbd_rl import PBDSamplingConfig, sample_pbd_block  # noqa: E402
import torch  # noqa: E402


TOKEN_IDS = {
    "box_start_token_id": 151668,
    "box_end_token_id": 151669,
    "coord_start_token_id": 151677,
    "coord_end_token_id": 152677,
    "none_token_id": 4064,
    "null_token_id": 152678,
    "im_end_token_id": 151645,
    "ref_end_token_id": 151673,
    "default_mask_token_id": 151643,
}


def _coord(value: int) -> int:
    return TOKEN_IDS["coord_start_token_id"] + int(value)


def _valid_coord_box(x1=110, y1=210, x2=320, y2=430):
    return [
        TOKEN_IDS["box_start_token_id"],
        _coord(x1),
        _coord(y1),
        _coord(x2),
        _coord(y2),
        TOKEN_IDS["box_end_token_id"],
    ]


def test_malformed_coordinate_block_triggers_error_box_ntp_gate() -> None:
    """Production Hybrid: incomplete/non-coord frame -> error_box -> AR/NTP."""
    # box_start + one valid coord + non-coord junk (mirrors decode_bbox_avg
    # writing token id 0 into abnormal slots, or unrestricted MTP junk).
    malformed = [
        TOKEN_IDS["box_start_token_id"],
        _coord(10),
        0,
        0,
        0,
        TOKEN_IDS["null_token_id"],
    ]
    pattern = classify_hybrid_mtp_proposal(malformed, TOKEN_IDS)
    assert pattern["type"] == "error_box"
    assert pattern["need_switch_to_ar"] is True
    assert pattern["tokens"] == [
        TOKEN_IDS["box_start_token_id"],
        _coord(10),
    ]


def test_spatially_ambiguous_or_broken_end_triggers_error_box() -> None:
    """box_start + 4 coords but missing </box> -> error_box (hybrid only)."""
    broken_end = [
        TOKEN_IDS["box_start_token_id"],
        _coord(10),
        _coord(20),
        _coord(30),
        _coord(40),
        TOKEN_IDS["null_token_id"],  # not box_end
    ]
    pattern = classify_hybrid_mtp_proposal(broken_end, TOKEN_IDS)
    assert pattern["type"] == "error_box"
    assert pattern["need_switch_to_ar"] is True
    assert pattern["tokens"] == [
        TOKEN_IDS["box_start_token_id"],
        _coord(10),
        _coord(20),
        _coord(30),
        _coord(40),
    ]


def test_second_valid_coord_box_does_not_trigger_error_box() -> None:
    """Production Hybrid does not fall back for a second valid coord_box.

    Each MTP proposal is gated independently. A structurally valid second box
    is accepted online; single-object completed_box_count!=1 is post-hoc only.
    """
    first = _valid_coord_box(10, 20, 30, 40)
    second = _valid_coord_box(50, 60, 70, 80)
    p1 = classify_hybrid_mtp_proposal(first, TOKEN_IDS)
    p2 = classify_hybrid_mtp_proposal(second, TOKEN_IDS)
    assert p1["type"] == "coord_box"
    assert p1["need_switch_to_ar"] is False
    assert p2["type"] == "coord_box"
    assert p2["need_switch_to_ar"] is False


def test_accepted_valid_pbd_coord_box_remains_pbd() -> None:
    pattern = classify_hybrid_mtp_proposal(_valid_coord_box(), TOKEN_IDS)
    assert pattern["type"] == "coord_box"
    assert pattern["need_switch_to_ar"] is False
    assert pattern["tokens"] == _valid_coord_box()
    assert len(pattern["tokens"]) == 6


def test_forced_pbd_box_supports_cannot_emit_error_box() -> None:
    """Root cause of zero NTP fallback in the saved Hybrid viability run."""
    assert forced_pbd_box_proposal_can_emit_error_box() is False
    vocab = 152700
    logits = torch.zeros(6, vocab)
    # Force <box> at slot 0 and valid coords / </box> elsewhere via huge logits.
    logits[0, TOKEN_IDS["box_start_token_id"]] = 100.0
    for i, value in enumerate((110, 210, 320, 430), start=1):
        logits[i, _coord(value)] = 100.0
    logits[5, TOKEN_IDS["box_end_token_id"]] = 100.0
    cfg = PBDSamplingConfig(temperature=1.0, top_k=0, top_p=1.0)
    gen = torch.Generator()
    gen.manual_seed(0)
    _block_type, actions, _slots, _eos = sample_pbd_block(
        logits,
        history_ids=[1, 2, 3],
        token_ids=TOKEN_IDS,
        config=cfg,
        generator=gen,
        force_box_block=True,
    )
    # Forced supports guarantee a structurally valid coord_box.
    pattern = classify_hybrid_mtp_proposal(actions, TOKEN_IDS)
    assert pattern["type"] == "coord_box"
    assert pattern["need_switch_to_ar"] is False


def test_hybrid_unrestricted_sampling_can_reach_error_box() -> None:
    """Unrestricted Hybrid MTP sampling + junk coords -> error_box gate."""
    vocab = 152700
    logits = torch.zeros(6, vocab)
    logits[0, TOKEN_IDS["box_start_token_id"]] = 100.0
    logits[1, _coord(10)] = 100.0
    # Slots 2-5 prefer null / zero-ish non-coords (token 0 has mass via default).
    logits[2, TOKEN_IDS["null_token_id"]] = 100.0
    logits[3, TOKEN_IDS["null_token_id"]] = 100.0
    logits[4, TOKEN_IDS["null_token_id"]] = 100.0
    logits[5, TOKEN_IDS["null_token_id"]] = 100.0
    cfg = PBDSamplingConfig(temperature=1.0, top_k=0, top_p=1.0)
    gen = torch.Generator()
    gen.manual_seed(0)
    actions, slots = sample_hybrid_mtp_block(
        logits,
        history_ids=[1, 2, 3],
        token_ids=TOKEN_IDS,
        config=cfg,
        generator=gen,
    )
    assert all(slot.support_kind == "full" for slot in slots)
    pattern = classify_hybrid_mtp_proposal(actions, TOKEN_IDS)
    assert pattern["type"] == "error_box"
    assert pattern["need_switch_to_ar"] is True


def test_multi_box_failure_is_post_hoc_not_online_gate() -> None:
    """The 86 saved multi-coord_box failures were not production Hybrid fallbacks.

    Detection of completed_box_count != 1 happens only when attaching the
    final committed box for reward, after the full token sequence was already
    generated by accepting each valid coord_box online.
    """
    boxes = [
        _valid_coord_box(10, 20, 30, 40),
        _valid_coord_box(50, 60, 70, 80),
    ]
    online_types = [
        classify_hybrid_mtp_proposal(b, TOKEN_IDS)["type"] for b in boxes
    ]
    assert online_types == ["coord_box", "coord_box"]
    completed_box_count = sum(1 for t in online_types if t == "coord_box")
    # Post-hoc single-object rule (reward / final-prediction), not error_box.
    assert completed_box_count != 1
    assert all(
        classify_hybrid_mtp_proposal(b, TOKEN_IDS)["need_switch_to_ar"] is False
        for b in boxes
    )


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"HYBRID FALLBACK GATE TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
