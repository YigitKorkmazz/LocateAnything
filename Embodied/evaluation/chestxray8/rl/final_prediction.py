"""Decoder-committed final bbox extraction for native GRPO rewards.

Spatial and semantic rewards consume exactly one final predicted box: the
bbox explicitly committed by the active decoder (PBD-only or Hybrid), never
an arbitrary BOX_RE match mined from malformed text and never a GT-IoU
selected candidate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

Box = Tuple[int, int, int, int]


def coord_token_to_int(token_id: int, token_ids: Dict[str, int]) -> Optional[int]:
    start = int(token_ids["coord_start_token_id"])
    end = int(token_ids["coord_end_token_id"])
    if start <= int(token_id) <= end:
        return int(token_id) - start
    return None


def box_from_coord_box_tokens(
    action_token_ids: Sequence[int],
    token_ids: Dict[str, int],
) -> Optional[Box]:
    """Decode a committed 6-token ``<box><x1><y1><x2><y2></box>`` action."""
    ids = [int(x) for x in action_token_ids]
    if len(ids) != 6:
        return None
    if ids[0] != int(token_ids["box_start_token_id"]):
        return None
    if ids[5] != int(token_ids["box_end_token_id"]):
        return None
    coords: List[int] = []
    for token_id in ids[1:5]:
        value = coord_token_to_int(token_id, token_ids)
        if value is None:
            return None
        coords.append(value)
    box = (coords[0], coords[1], coords[2], coords[3])
    if not all(0 <= value <= 1000 for value in box):
        return None
    return box


def box_from_token_span(
    action_token_ids: Sequence[int],
    token_ids: Dict[str, int],
) -> Optional[Box]:
    """Decode a committed box span that may come from PBD accept or NTP fallback."""
    ids = [int(x) for x in action_token_ids]
    if not ids:
        return None
    if ids[0] != int(token_ids["box_start_token_id"]):
        return None
    if ids[-1] != int(token_ids["box_end_token_id"]):
        return None
    middle = ids[1:-1]
    if len(middle) != 4:
        return None
    coords: List[int] = []
    for token_id in middle:
        value = coord_token_to_int(token_id, token_ids)
        if value is None:
            return None
        coords.append(value)
    box = (coords[0], coords[1], coords[2], coords[3])
    if not all(0 <= value <= 1000 for value in box):
        return None
    return box


def attach_pbd_final_prediction(
    trace: Any,
    token_ids: Dict[str, int],
) -> Any:
    """Set PBD-only committed final box fields on a ``RolloutTrace``."""
    box_blocks = [block for block in trace.blocks if block.block_type == "box"]
    trace.decoder_path = "pbd"
    trace.fallback_triggered = False
    trace.rejected_pbd_proposals = []
    if len(box_blocks) != 1:
        trace.committed_final_box_norm_1000 = None
        trace.has_unambiguous_committed_box = False
        trace.reward_branch = "none"
        return trace
    box = box_from_coord_box_tokens(box_blocks[0].action_token_ids, token_ids)
    if box is None:
        trace.committed_final_box_norm_1000 = None
        trace.has_unambiguous_committed_box = False
        trace.reward_branch = "none"
        return trace
    trace.committed_final_box_norm_1000 = box
    trace.has_unambiguous_committed_box = True
    trace.reward_branch = "pbd"
    return trace


def committed_box_payload(trace: Any) -> Dict[str, Any]:
    return {
        "decoder_path": getattr(trace, "decoder_path", "pbd"),
        "reward_branch": getattr(trace, "reward_branch", "none"),
        "has_unambiguous_committed_box": bool(
            getattr(trace, "has_unambiguous_committed_box", False)
        ),
        "committed_final_box_norm_1000": getattr(
            trace, "committed_final_box_norm_1000", None
        ),
        "fallback_triggered": bool(getattr(trace, "fallback_triggered", False)),
        "rejected_pbd_proposals": list(
            getattr(trace, "rejected_pbd_proposals", []) or []
        ),
    }
