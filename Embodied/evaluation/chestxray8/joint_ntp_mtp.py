#!/usr/bin/env python3
"""Joint NTP + MTP packing / loss helpers for LocateAnything ChestX-ray8 SFT.

Implements the paper Section 3.2 / Figure 4 dual-stream layout used by the
official LocateAnything finetune path (``get_targets_flag_with_mtp``):

    x_all = x_vis + x_q + x_ntp + x_blk

with PE drop at the first MTP token so ``create_block_diff_mask_by_pe_4d``
isolates streams.

This module is only used when ``--training-objective joint_ntp_mtp``.
The default ``ntp_only`` path must not import any of this behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sft_common import IGNORE_INDEX

# Stream ids aligned with ``labels`` (before causal shift).
STREAM_IGNORE = 0
STREAM_NTP = 1
STREAM_MTP = 2

DEFAULT_BLOCK_SIZE = 6  # LocateAnything released config / paper L=6


def resolve_special_token_ids(tokenizer) -> Dict[str, int]:
    """Resolve special token IDs from the tokenizer (no hard-coded IDs)."""
    required = {
        "mask": "<text_mask>",
        "null": "<null>",
        "box_end": "</box>",
        "ref_end": "</ref>",
        "eos": "<|im_end|>",
        "im_start": "<|im_start|>",
        "assistant": "assistant",
    }
    out: Dict[str, int] = {}
    for key, tok in required.items():
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid == tokenizer.unk_token_id:
            raise RuntimeError(f"Tokenizer missing required token {tok!r}")
        out[key] = int(tid)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer.pad_token_id is None")
    out["pad"] = int(pad_id)
    return out


def find_assistant_spans(
    input_ids: torch.Tensor,
    tokenizer,
) -> List[Tuple[int, int, int, int]]:
    """Return list of (resp_start, resp_end, label_start, label_end).

    Matches official ``get_targets_flag_with_mtp`` indexing:
      resp_start = assistant_idx + 1  (usually newline after 'assistant')
      label_start = resp_start + 1
      label_end = eot index (inclusive)
      resp_end = eot + 1
    """
    if input_ids.dim() == 2:
        assert input_ids.size(0) == 1
        ids = input_ids[0]
    else:
        ids = input_ids

    ids_tok = resolve_special_token_ids(tokenizer)
    im_start_id = ids_tok["im_start"]
    assistant_id = ids_tok["assistant"]
    eos_id = ids_tok["eos"]

    start_header_idxs = torch.where(ids == im_start_id)[0]
    assistant_idxs = torch.where(ids == assistant_id)[0]
    eot_idxs = torch.where(ids == eos_id)[0]
    header_followers = set((start_header_idxs + 1).tolist())

    spans: List[Tuple[int, int, int, int]] = []
    for assistant_idx in assistant_idxs.tolist():
        if assistant_idx not in header_followers:
            continue
        st = assistant_idx + 1
        for eot_idx in eot_idxs.tolist():
            if eot_idx > st:
                spans.append((st, eot_idx + 1, st + 1, eot_idx))
                break
    if not spans:
        raise RuntimeError("No assistant spans found for joint NTP+MTP packing")
    return spans


def build_ntp_labels(
    input_ids: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """Assistant-only NTP labels on the x0 sequence (IGNORE elsewhere)."""
    if input_ids.dim() == 2:
        assert input_ids.size(0) == 1
        ids = input_ids[0]
    else:
        ids = input_ids
    labels = ids.clone()
    flag = torch.zeros_like(ids)
    for _st, _end, label_start, label_end in find_assistant_spans(ids, tokenizer):
        flag[label_start : label_end + 1] = 1
    if int(flag.sum().item()) == 0:
        raise RuntimeError("No NTP supervision tokens")
    labels[flag == 0] = IGNORE_INDEX
    return labels


def _pack_detection_blocks(
    input_ids_np: np.ndarray,
    spans: Sequence[Tuple[int, int, int, int]],
    block_size: int,
    ids_tok: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Official box/ref-aware MTP packing (LocateAnything PBD)."""
    box_end_id = ids_tok["box_end"]
    ref_end_id = ids_tok["ref_end"]
    eos_id = ids_tok["eos"]
    null_id = ids_tok["null"]
    mask_id = ids_tok["mask"]

    all_mask_input_ids: List[np.ndarray] = []
    all_mask_targets: List[np.ndarray] = []
    all_mask_positions: List[np.ndarray] = []
    block_meta: List[Dict[str, Any]] = []

    for start, end, _ls, _le in spans:
        curr = start
        while curr < end:
            anchor_token = int(input_ids_np[curr])
            if anchor_token == eos_id:
                break
            pred_start = curr + 1
            if pred_start > end:
                break
            candidates = input_ids_np[pred_start : min(pred_start + block_size, end + 1)]
            if len(candidates) == 0:
                break

            valid_len = len(candidates)
            eos_indices = np.where(candidates == eos_id)[0]
            if len(eos_indices) > 0:
                first_eos_idx = int(eos_indices[0])
                valid_len = 1 if first_eos_idx == 0 else first_eos_idx

            if valid_len > 1 or (len(eos_indices) > 0 and int(eos_indices[0]) != 0):
                ref_indices = np.where(candidates[:valid_len] == ref_end_id)[0]
                if len(ref_indices) > 0:
                    valid_len = min(valid_len, int(ref_indices[0]) + 1)
                box_indices = np.where(candidates[:valid_len] == box_end_id)[0]
                if len(box_indices) > 0:
                    valid_len = min(valid_len, int(box_indices[0]) + 1)

            target_block = np.full(block_size, null_id, dtype=input_ids_np.dtype)
            target_block[:valid_len] = candidates[:valid_len]

            mask_input_block = np.full(block_size, mask_id, dtype=input_ids_np.dtype)
            mask_input_block[0] = anchor_token

            # Integrity: only first token preserved; rest are masks.
            if mask_input_block[0] != anchor_token:
                raise RuntimeError("MTP block anchor mismatch")
            if not np.all(mask_input_block[1:] == mask_id):
                raise RuntimeError("MTP block non-anchor tokens must be <text_mask>")
            # Masked slots must not equal their targets in the *input* stream
            # (targets live only in labels).
            for j in range(1, valid_len):
                if mask_input_block[j] == target_block[j] and target_block[j] != mask_id:
                    raise RuntimeError(
                        f"Label leakage in MTP input: position {j} has target token"
                    )

            pos_block = np.arange(curr, curr + block_size, dtype=np.int32)
            all_mask_input_ids.append(mask_input_block)
            all_mask_targets.append(target_block)
            all_mask_positions.append(pos_block)
            block_meta.append(
                {
                    "anchor_pos": int(curr),
                    "valid_len": int(valid_len),
                    "anchor_token": int(anchor_token),
                    "targets": target_block[:valid_len].tolist(),
                }
            )
            curr += valid_len

    if not all_mask_input_ids:
        raise RuntimeError("Detection packing produced zero MTP blocks")

    return (
        np.concatenate(all_mask_input_ids),
        np.concatenate(all_mask_targets),
        np.concatenate(all_mask_positions),
        block_meta,
    )


def pack_joint_ntp_mtp(
    input_ids: torch.Tensor,
    tokenizer,
    block_size: int = DEFAULT_BLOCK_SIZE,
    run_integrity_checks: bool = True,
) -> Dict[str, Any]:
    """Build ``x_vis+x_q+x_ntp+x_blk`` with NTP+MTP labels and PE drop.

    Returns tensors shaped ``[seq]`` (no batch dim):
      input_ids, labels, position_ids, attention_mask, stream_ids, x0_len, ...
    """
    if input_ids.dim() == 2:
        assert input_ids.size(0) == 1
        ids = input_ids[0]
    else:
        ids = input_ids

    ids_tok = resolve_special_token_ids(tokenizer)
    spans = find_assistant_spans(ids, tokenizer)
    ntp_labels = build_ntp_labels(ids, tokenizer)

    input_ids_np = ids.detach().cpu().numpy()
    targets_np = ntp_labels.detach().cpu().numpy()
    len_x0 = int(len(input_ids_np))

    has_box = bool((ids == ids_tok["box_end"]).any().item())
    has_ref = bool((ids == ids_tok["ref_end"]).any().item())
    if not (has_box or has_ref):
        raise RuntimeError(
            "joint_ntp_mtp packing currently requires </box> or </ref> "
            "(ChestX-ray8 detection targets)."
        )

    final_mask_ids, final_mask_targets, final_mask_positions, block_meta = (
        _pack_detection_blocks(input_ids_np, spans, block_size, ids_tok)
    )

    bridge_ignore = np.array([IGNORE_INDEX], dtype=targets_np.dtype)
    pad_token = np.array([ids_tok["pad"]], dtype=input_ids_np.dtype)

    input_ids_out = np.concatenate([input_ids_np, final_mask_ids, pad_token])
    # Bridge IGNORE sits under the first MTP token after the causal shift.
    targets_out = np.concatenate([targets_np, bridge_ignore, final_mask_targets])
    orig_pos = np.arange(len_x0, dtype=np.int32)
    pad_pos = np.array([int(final_mask_positions[-1]) + 1], dtype=np.int32)
    position_ids_out = np.concatenate([orig_pos, final_mask_positions, pad_pos])

    if len(input_ids_out) != len(targets_out) or len(input_ids_out) != len(position_ids_out):
        raise RuntimeError("Packed sequence length mismatch")

    stream = np.zeros(len(targets_out), dtype=np.int64)
    stream[:len_x0] = np.where(targets_np != IGNORE_INDEX, STREAM_NTP, STREAM_IGNORE)
    # bridge at index len_x0
    stream[len_x0] = STREAM_IGNORE
    mtp_start = len_x0 + 1
    mtp_end = mtp_start + len(final_mask_targets)
    stream[mtp_start:mtp_end] = np.where(
        final_mask_targets != IGNORE_INDEX, STREAM_MTP, STREAM_IGNORE
    )
    # trailing pad
    stream[-1] = STREAM_IGNORE

    out_ids = torch.tensor(input_ids_out, dtype=torch.long)
    out_labels = torch.tensor(targets_out, dtype=torch.long)
    out_pos = torch.tensor(position_ids_out, dtype=torch.long)
    out_stream = torch.tensor(stream, dtype=torch.long)
    out_attn = out_ids.ne(ids_tok["pad"])

    result = {
        "input_ids": out_ids,
        "labels": out_labels,
        "position_ids": out_pos,
        "attention_mask": out_attn,
        "stream_ids": out_stream,
        "x0_len": len_x0,
        "block_size": int(block_size),
        "n_mtp_blocks": len(block_meta),
        "block_meta": block_meta,
        "token_ids": ids_tok,
        "ntp_token_count": int((out_stream == STREAM_NTP).sum().item()),
        "mtp_token_count": int((out_stream == STREAM_MTP).sum().item()),
    }

    if run_integrity_checks:
        assert_packing_integrity(result, original_ntp_ids=ids, original_ntp_labels=ntp_labels)

    return result


def assert_packing_integrity(
    packed: Dict[str, Any],
    original_ntp_ids: torch.Tensor,
    original_ntp_labels: torch.Tensor,
) -> None:
    """Fail loudly if packing violates paper dual-stream invariants."""
    x0 = int(packed["x0_len"])
    ids = packed["input_ids"]
    labels = packed["labels"]
    pos = packed["position_ids"]
    stream = packed["stream_ids"]
    block_size = int(packed["block_size"])
    mask_id = packed["token_ids"]["mask"]

    if not torch.equal(ids[:x0], original_ntp_ids.cpu()):
        raise RuntimeError("x0 prefix does not match original NTP input_ids")
    if not torch.equal(labels[:x0], original_ntp_labels.cpu()):
        raise RuntimeError("x0 NTP labels do not match original assistant labels")

    # PE must be contiguous on x0 then drop at first MTP token.
    if not torch.equal(pos[:x0], torch.arange(x0)):
        raise RuntimeError("x0 position_ids must be 0..x0_len-1")
    if pos[x0] >= pos[x0 - 1]:
        raise RuntimeError(
            f"Expected PE drop at MTP start: pos[{x0-1}]={int(pos[x0-1])} "
            f"pos[{x0}]={int(pos[x0])}"
        )

    # Bridge label must be IGNORE.
    if int(labels[x0].item()) != IGNORE_INDEX:
        raise RuntimeError("Bridge label under first MTP token must be IGNORE_INDEX")

    mtp_ids = ids[x0:-1]  # exclude trailing pad
    if mtp_ids.numel() % block_size != 0:
        raise RuntimeError(
            f"MTP region length {mtp_ids.numel()} not divisible by block_size={block_size}"
        )
    n_blocks = mtp_ids.numel() // block_size
    for b in range(n_blocks):
        block = mtp_ids[b * block_size : (b + 1) * block_size]
        if not torch.all(block[1:] == mask_id):
            raise RuntimeError(f"Block {b}: non-anchor tokens must be <text_mask>")
        # Labels for this block (shifted by bridge): labels[x0+1 + ...]
        lab = labels[x0 + 1 + b * block_size : x0 + 1 + (b + 1) * block_size]
        # Masked input tokens must not contain their supervised targets.
        for j in range(1, block_size):
            if int(lab[j].item()) == IGNORE_INDEX:
                continue
            if int(block[j].item()) == int(lab[j].item()):
                raise RuntimeError(
                    f"Block {b} slot {j}: input equals label (target leaked into input)"
                )

    if int((stream == STREAM_NTP).sum().item()) == 0:
        raise RuntimeError("No NTP stream labels")
    if int((stream == STREAM_MTP).sum().item()) == 0:
        raise RuntimeError("No MTP stream labels")


def import_block_mask_fn():
    """Load official ``create_block_diff_mask_by_pe_4d`` from the HF package."""
    try:
        from transformers_modules.nvidia.LocateAnything_hyphen_3B.c32291ca5e996f5a7a485845b4f57a233936bba0.mask_sdpa_utils import (  # noqa: E501
            create_block_diff_mask_by_pe_4d,
            find_prefix_seq_length_by_pe,
        )

        return create_block_diff_mask_by_pe_4d, find_prefix_seq_length_by_pe
    except Exception:
        from eaglevl.model.locany.mask_sdpa_utils import (  # type: ignore
            create_block_diff_mask_by_pe_4d,
            find_prefix_seq_length_by_pe,
        )

        return create_block_diff_mask_by_pe_4d, find_prefix_seq_length_by_pe


def build_figure4_attention_mask(
    position_ids: torch.Tensor,
    block_size: int,
    causal_attn: bool = False,
) -> torch.Tensor:
    """Return float 4D mask ``[B,1,S,S]`` (0 allowed, -inf blocked)."""
    create_block_diff_mask_by_pe_4d, find_prefix_seq_length_by_pe = import_block_mask_fn()
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    x0_len = find_prefix_seq_length_by_pe(position_ids)
    if int(x0_len[0].item()) < 0:
        raise RuntimeError("No PE drop found; joint packing requires MTP suffix")
    mask, _ = create_block_diff_mask_by_pe_4d(
        block_size=block_size,
        x0_len_list=x0_len,
        position_ids=position_ids,
        causal_attn=causal_attn,
    )
    return mask


def assert_attention_mask_invariants(
    position_ids: torch.Tensor,
    block_size: int,
    x0_len: int,
    tol: float = 0.0,
) -> Dict[str, Any]:
    """Verify Figure-4 visibility: causal x0, NTP isolation, block rules."""
    if position_ids.dim() == 1:
        pos = position_ids.unsqueeze(0)
    else:
        pos = position_ids
    mask = build_figure4_attention_mask(pos, block_size=block_size)  # [1,1,S,S]
    allowed = mask[0, 0] == 0  # True = visible
    S = allowed.size(0)
    report: Dict[str, Any] = {"seq_len": S, "x0_len": x0_len, "block_size": block_size}

    # 1) Causal on x0
    for q in range(x0_len):
        for kv in range(x0_len):
            should = q >= kv
            if bool(allowed[q, kv].item()) != should:
                raise RuntimeError(
                    f"x0 causal fail at q={q}, kv={kv}: allowed={bool(allowed[q, kv])}"
                )

    # 2) NTP (x0) must not attend to MTP region
    for q in range(x0_len):
        if bool(allowed[q, x0_len:].any().item()):
            bad = torch.where(allowed[q, x0_len:])[0][:5].tolist()
            raise RuntimeError(
                f"NTP leakage: x0 query {q} can see MTP keys at offsets {bad}"
            )

    # 3/4) MTP blocks: future blocks invisible; within-block bidirectional;
    # previous blocks visible (via block index), prefix into x0 allowed.
    n_mtp = S - x0_len
    # Exclude trailing pad if present: still OK to test full suffix.
    n_blocks = n_mtp // block_size
    for bq in range(n_blocks):
        q0 = x0_len + bq * block_size
        q1 = q0 + block_size
        for bkv in range(n_blocks):
            k0 = x0_len + bkv * block_size
            k1 = k0 + block_size
            sub = allowed[q0:q1, k0:k1]
            if bkv > bq:
                if bool(sub.any().item()):
                    raise RuntimeError(
                        f"Future block visible: q_block={bq} sees kv_block={bkv}"
                    )
            elif bkv == bq:
                # Bidirectional within block: all True
                if not bool(sub.all().item()):
                    raise RuntimeError(
                        f"Within-block attention not fully bidirectional at block {bq}"
                    )
            else:
                # Official HF mask: different MTP blocks do NOT attend each other.
                # "Previous committed" context is exposed via block_prefix into x0
                # (PE at block start = NTP anchor index), not via cross-block MTP.
                if bool(sub.any().item()):
                    raise RuntimeError(
                        f"Unexpected cross-block MTP visibility q={bq} kv={bkv}"
                    )

    report["ntp_isolated"] = True
    report["x0_causal"] = True
    report["within_block_bidirectional"] = True
    report["future_blocks_masked"] = True
    report["note"] = (
        "Official create_block_diff_mask_by_pe_4d exposes prior NTP prefix via "
        "block_prefix (PE at block start), not cross-block attention among MTP "
        "mask tokens. Same-block attention is bidirectional."
    )
    return report


def compute_joint_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    stream_ids: torch.Tensor,
    lambda_ntp: float = 1.0,
    lambda_mtp: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Explicit L_ntp, L_mtp, L_total from shifted CE (IGNORE excluded via stream)."""
    # logits: [B, S, V]; labels/stream: [B, S]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_stream = stream_ids[..., 1:].contiguous()

    vocab = shift_logits.size(-1)
    flat_logits = shift_logits.view(-1, vocab)
    flat_labels = shift_labels.view(-1)
    flat_stream = shift_stream.view(-1)

    def _mean_ce(mask: torch.Tensor) -> torch.Tensor:
        if int(mask.sum().item()) == 0:
            # Keep graph connection with a zero that still depends on logits.
            return flat_logits.sum() * 0.0
        return F.cross_entropy(flat_logits[mask], flat_labels[mask])

    ntp_mask = flat_stream == STREAM_NTP
    mtp_mask = flat_stream == STREAM_MTP
    # Safety: never supervise IGNORE even if stream mislabeled.
    ntp_mask = ntp_mask & (flat_labels != IGNORE_INDEX)
    mtp_mask = mtp_mask & (flat_labels != IGNORE_INDEX)

    loss_ntp = _mean_ce(ntp_mask)
    loss_mtp = _mean_ce(mtp_mask)
    loss_total = float(lambda_ntp) * loss_ntp + float(lambda_mtp) * loss_mtp
    return {
        "loss_ntp": loss_ntp,
        "loss_mtp": loss_mtp,
        "loss_total": loss_total,
        "n_ntp": ntp_mask.sum().detach(),
        "n_mtp": mtp_mask.sum().detach(),
    }


def describe_packing(packed: Dict[str, Any], tokenizer, max_tokens: int = 80) -> str:
    """Human-readable packing dump for unit tests / debug."""
    ids = packed["input_ids"]
    labels = packed["labels"]
    pos = packed["position_ids"]
    stream = packed["stream_ids"]
    x0 = int(packed["x0_len"])

    def _dec(t: torch.Tensor) -> str:
        return tokenizer.decode(t.tolist(), skip_special_tokens=False)

    lines = [
        f"x0_len={x0} total_len={ids.numel()} n_mtp_blocks={packed['n_mtp_blocks']} "
        f"block_size={packed['block_size']}",
        f"ntp_tokens={packed['ntp_token_count']} mtp_tokens={packed['mtp_token_count']}",
        f"NTP input preview: {_dec(ids[: min(x0, max_tokens)])}",
        f"MTP input preview: {_dec(ids[x0 : min(ids.numel(), x0 + max_tokens)])}",
        "Blocks:",
    ]
    for i, meta in enumerate(packed["block_meta"][:20]):
        lines.append(
            f"  [{i}] anchor_pos={meta['anchor_pos']} valid_len={meta['valid_len']} "
            f"anchor={meta['anchor_token']} targets={meta['targets']}"
        )
    # Show PE drop
    lines.append(
        f"PE around drop: ... {pos[max(0, x0-3): x0+3].tolist()} ..."
    )
    lines.append(
        f"stream around drop: {stream[max(0, x0-3): x0+3].tolist()}"
    )
    lines.append(
        f"labels around drop: {labels[max(0, x0-3): x0+3].tolist()}"
    )
    return "\n".join(lines)


def param_grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Sum L2 grad norms for LoRA / mlp1 / vision / embeds groups."""
    groups = {
        "lora": 0.0,
        "mlp1": 0.0,
        "vision": 0.0,
        "embed_tokens": 0.0,
        "lm_head": 0.0,
        "other_trainable": 0.0,
    }
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().float().norm().item())
        lname = name.lower()
        if "lora_" in lname:
            groups["lora"] += g * g
        elif lname.startswith("mlp1.") or ".mlp1." in lname:
            groups["mlp1"] += g * g
        elif "vision" in lname:
            groups["vision"] += g * g
        elif "embed_tokens" in lname:
            groups["embed_tokens"] += g * g
        elif "lm_head" in lname:
            groups["lm_head"] += g * g
        elif p.requires_grad:
            groups["other_trainable"] += g * g
    return {k: float(v ** 0.5) for k, v in groups.items()}
