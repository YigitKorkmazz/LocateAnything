#!/usr/bin/env python3
"""Unit tests for joint NTP+MTP packing, Figure-4 attention, and loss separation.

Run (from Embodied/):

  /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python \\
    evaluation/chestxray8/test_joint_ntp_mtp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from joint_ntp_mtp import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    STREAM_MTP,
    STREAM_NTP,
    assert_attention_mask_invariants,
    assert_packing_integrity,
    build_figure4_attention_mask,
    compute_joint_losses,
    describe_packing,
    pack_joint_ntp_mtp,
)
from sft_common import IGNORE_INDEX  # noqa: E402


def _synthetic_detection_ids(tokenizer, n_boxes: int = 1) -> torch.Tensor:
    """Build a minimal chat-like id sequence with assistant + one box answer."""
    ids_list = []
    ids_list += tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    ids_list += tokenizer.encode("Locate the Mass in this chest X-ray", add_special_tokens=False)
    ids_list += tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    ids_list += tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    # Answer: <ref>Mass</ref><box><10><20><30><40></box>
    answer = "<ref>Mass</ref><box><10><20><30><40></box>"
    if n_boxes > 1:
        answer += "<box><50><60><70><80></box>"
    ids_list += tokenizer.encode(answer, add_special_tokens=False)
    ids_list += tokenizer.encode("<|im_end|>", add_special_tokens=False)
    return torch.tensor(ids_list, dtype=torch.long)


def load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "nvidia/LocateAnything-3B",
        revision="c32291ca5e996f5a7a485845b4f57a233936bba0",
        trust_remote_code=True,
    )
    # Ensure special tokens exist (they should on the released vocab).
    for t in ("<text_mask>", "<null>", "<box>", "</box>", "<ref>", "</ref>"):
        assert tok.convert_tokens_to_ids(t) is not None
    return tok


def test_packing(tokenizer) -> dict:
    print("\n=== PACKING TEST ===")
    ids = _synthetic_detection_ids(tokenizer)
    packed = pack_joint_ntp_mtp(ids, tokenizer, block_size=DEFAULT_BLOCK_SIZE)
    dump = describe_packing(packed, tokenizer)
    print(dump)
    assert packed["ntp_token_count"] > 0
    assert packed["mtp_token_count"] > 0
    assert packed["n_mtp_blocks"] > 0
    # PE drop
    x0 = packed["x0_len"]
    assert int(packed["position_ids"][x0]) < int(packed["position_ids"][x0 - 1])
    # First token of each block preserved / rest masks
    mask_id = packed["token_ids"]["mask"]
    mtp = packed["input_ids"][x0:-1]
    B = packed["block_size"]
    for b in range(mtp.numel() // B):
        block = mtp[b * B : (b + 1) * B]
        assert torch.all(block[1:] == mask_id)
    print("PACKING TEST OK")
    return packed


def test_attention_mask(packed: dict) -> None:
    print("\n=== ATTENTION MASK TEST ===")
    report = assert_attention_mask_invariants(
        packed["position_ids"],
        block_size=packed["block_size"],
        x0_len=packed["x0_len"],
    )
    print(json.dumps(report, indent=2))
    mask = build_figure4_attention_mask(
        packed["position_ids"], block_size=packed["block_size"]
    )
    allowed = mask[0, 0] == 0
    x0 = packed["x0_len"]
    # Spot-check: last x0 token cannot see any MTP key
    assert not bool(allowed[x0 - 1, x0:].any().item())
    # First MTP token can see some x0 prefix (block_prefix)
    assert bool(allowed[x0, :x0].any().item()), "MTP should see some x0 prefix"
    print("ATTENTION MASK TEST OK")


def test_loss_separation(packed: dict) -> None:
    print("\n=== LOSS SEPARATION TEST ===")
    S = packed["input_ids"].numel()
    V = 128  # tiny fake vocab
    torch.manual_seed(0)
    logits = torch.randn(1, S, V, requires_grad=True)
    labels = packed["labels"].unsqueeze(0).clamp(min=0, max=V - 1)
    # Restore IGNORE positions after clamp
    labels = packed["labels"].unsqueeze(0).clone()
    labels[labels != IGNORE_INDEX] = labels[labels != IGNORE_INDEX].clamp(0, V - 1)
    stream = packed["stream_ids"].unsqueeze(0)

    base = compute_joint_losses(logits, labels, stream, lambda_ntp=1.0, lambda_mtp=1.0)
    assert torch.isfinite(base["loss_ntp"])
    assert torch.isfinite(base["loss_mtp"])
    # With lambdas 1.0: total == ntp + mtp
    assert torch.allclose(
        base["loss_total"], base["loss_ntp"] + base["loss_mtp"], atol=1e-5
    )

    # Change one NTP label → only loss_ntp should change
    labels_ntp = labels.clone()
    ntp_pos = (stream == STREAM_NTP).nonzero(as_tuple=False)[0]
    b, j = int(ntp_pos[0]), int(ntp_pos[1])
    labels_ntp[b, j] = (labels_ntp[b, j] + 3) % V
    if labels_ntp[b, j] == IGNORE_INDEX:
        labels_ntp[b, j] = 0
    alt_ntp = compute_joint_losses(
        logits, labels_ntp, stream, lambda_ntp=1.0, lambda_mtp=1.0
    )
    assert not torch.allclose(alt_ntp["loss_ntp"], base["loss_ntp"])
    assert torch.allclose(alt_ntp["loss_mtp"], base["loss_mtp"], atol=1e-6)

    # Change one MTP label → only loss_mtp should change
    labels_mtp = labels.clone()
    mtp_pos = (stream == STREAM_MTP).nonzero(as_tuple=False)[0]
    b, j = int(mtp_pos[0]), int(mtp_pos[1])
    labels_mtp[b, j] = (labels_mtp[b, j] + 5) % V
    if labels_mtp[b, j] == IGNORE_INDEX:
        labels_mtp[b, j] = 1
    alt_mtp = compute_joint_losses(
        logits, labels_mtp, stream, lambda_ntp=1.0, lambda_mtp=1.0
    )
    assert torch.allclose(alt_mtp["loss_ntp"], base["loss_ntp"], atol=1e-6)
    assert not torch.allclose(alt_mtp["loss_mtp"], base["loss_mtp"])

    # Lambda weighting
    w = compute_joint_losses(logits, labels, stream, lambda_ntp=2.0, lambda_mtp=0.5)
    expected = 2.0 * base["loss_ntp"] + 0.5 * base["loss_mtp"]
    assert torch.allclose(w["loss_total"], expected, atol=1e-5)
    print(
        f"loss_ntp={float(base['loss_ntp']):.4f} "
        f"loss_mtp={float(base['loss_mtp']):.4f} "
        f"loss_total={float(base['loss_total']):.4f}"
    )
    print("LOSS SEPARATION TEST OK")


def main() -> None:
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    packed = test_packing(tokenizer)
    test_attention_mask(packed)
    test_loss_separation(packed)
    print("\nALL UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
