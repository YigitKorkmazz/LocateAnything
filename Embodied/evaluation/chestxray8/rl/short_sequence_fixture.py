"""Fit-capable short-sequence fixture for selective-offload A/B gradient oracle.

Production traces keep ~1400-token prompts (mostly image tokens). That makes
pristine no-offload A infeasible on one 24 GB GPU. This fixture keeps the same
production model/PEFT/SDPA/live-KV semantics but shrinks the image so block-0
activations fit, with >=3 scored MTP blocks for cross-block KV gradients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from rl.pbd_rl import BlockTrace, PBDSamplingConfig, RolloutTrace, SlotTrace
from rl.prompt import build_rl_user_text, resolve_prompt_mode


IMAGE_CONTEXT_TOKEN_ID = 151665

# Backend-validation-only per-block advantages (unequal signs/magnitudes).
DEFAULT_ORACLE_BLOCK_ADVANTAGES: Tuple[float, ...] = (1.0, -0.7, 1.5)

# Deterministic MTP action templates (support_kind=full). Token IDs match the
# production smoke-trace vocabulary patterns; this is a backend-gradient
# fixture only (not GT selection / reward evaluation).
DEFAULT_SCORED_ACTION_BLOCKS: Tuple[Tuple[int, ...], ...] = (
    (151672, 641, 84746, 367, 304, 419),
    (15138, 1599, 29530, 151673, 220, 198),
    (151668, 151677, 151760, 151728, 152300, 151669),
)

# Resize candidates (max side, pixels). First that yields prompt_len <= target wins.
DEFAULT_IMAGE_MAX_SIDE_CANDIDATES: Tuple[int, ...] = (168, 128, 112, 96, 64)


def _resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    scale = float(max_side) / float(max(w, h))
    if scale >= 1.0:
        return image
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    try:
        resample = Image.Resampling.BICUBIC
    except AttributeError:  # pragma: no cover
        resample = Image.BICUBIC
    return image.resize((nw, nh), resample)


def tokenize_pair_with_resized_image(
    processor,
    pair: Dict[str, Any],
    device: torch.device,
    *,
    config: Optional[Dict[str, Any]],
    image_max_side: int,
) -> Dict[str, Any]:
    """Like runtime.tokenize_rl_pair but with a downscaled RGB image."""
    mode = resolve_prompt_mode(config) if config is not None else "native_locateanything"
    image = Image.open(pair["image_path"]).convert("RGB")
    original_size = image.size
    image = _resize_max_side(image, int(image_max_side))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": build_rl_user_text(pair, prompt_mode=mode),
                },
            ],
        }
    ]
    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = processor.process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    prompt_token_ids = input_ids[0].detach().cpu().tolist()
    return {
        "input_ids": input_ids,
        "attention_mask": inputs["attention_mask"].to(device),
        "pixel_values": inputs["pixel_values"].to(device=device, dtype=torch.bfloat16),
        "image_grid_hws": torch.as_tensor(inputs["image_grid_hws"], device=device),
        "rendered_prompt": text,
        "prompt_token_ids": prompt_token_ids,
        "prompt_len": len(prompt_token_ids),
        "image_token_count": sum(
            1 for t in prompt_token_ids if int(t) == IMAGE_CONTEXT_TOKEN_ID
        ),
        "image_original_size": list(original_size),
        "image_resized_size": list(image.size),
        "image_max_side": int(image_max_side),
    }


def build_short_rollout_trace(
    prompt_token_ids: Sequence[int],
    *,
    num_scored_blocks: int = 3,
    block_size: int = 6,
    action_blocks: Optional[Sequence[Sequence[int]]] = None,
    sampling: Optional[PBDSamplingConfig] = None,
) -> RolloutTrace:
    """Build a hybrid RolloutTrace with live-cache-consistent MTP blocks."""
    if int(num_scored_blocks) < 3:
        raise RuntimeError("short-sequence oracle requires num_scored_blocks >= 3")
    if int(block_size) != 6:
        raise RuntimeError("LocateAnything hybrid fixture requires block_size=6")
    templates = list(action_blocks or DEFAULT_SCORED_ACTION_BLOCKS)
    if len(templates) < int(num_scored_blocks):
        # Repeat last template deterministically if more blocks requested.
        while len(templates) < int(num_scored_blocks):
            templates.append(templates[-1])
    sampling = sampling or PBDSamplingConfig()
    sampling.validate()

    prompt = [int(t) for t in prompt_token_ids]
    prefix = len(prompt)
    cache_before = 0
    blocks: List[BlockTrace] = []
    generated: List[int] = []

    for bi in range(int(num_scored_blocks)):
        actions = [int(x) for x in templates[bi][:block_size]]
        if len(actions) != block_size:
            raise RuntimeError(
                f"action block {bi} must have exactly {block_size} tokens; got {len(actions)}"
            )
        slots = [
            SlotTrace(
                slot_index=si,
                action_token_id=actions[si],
                support_kind="full",
                log_prob_old=0.0,
                support_size=0,
                top_k=int(sampling.top_k),
                top_p=float(sampling.top_p),
                temperature=float(sampling.temperature),
            )
            for si in range(block_size)
        ]
        # Approximate window ids for trace completeness (score() recomputes).
        window_ids = list(prompt + generated) + [actions[0]] * 1 + [0] * (block_size - 1)
        blocks.append(
            BlockTrace(
                block_index=bi,
                prefix_length=int(prefix),
                cache_length_before=int(cache_before),
                cache_length_after=int(prefix),
                block_type="short_fixture_mtp",
                position_ids=list(range(prefix + block_size)),
                input_window_ids=window_ids[: prefix + block_size],
                action_token_ids=list(actions),
                slots=slots,
                scored_for_grpo=True,
                source="pbd",
                rejected_proposal_token_ids=None,
            )
        )
        generated.extend(actions)
        cache_before = prefix
        prefix = prefix + block_size

    return RolloutTrace(
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        blocks=blocks,
        sampling=sampling,
        stopped_on_eos=False,
        truncated=True,
        decoded_text=None,
        decoder_path="hybrid",
        reward_branch="none",
        committed_final_box_norm_1000=None,
        has_unambiguous_committed_box=False,
        fallback_triggered=False,
        rejected_pbd_proposals=[],
    )


def build_short_sequence_gradient_oracle_fixture(
    *,
    pair: Dict[str, Any],
    processor,
    device: torch.device,
    config: Dict[str, Any],
    num_scored_blocks: int = 3,
    target_max_prompt_tokens: int = 192,
    image_max_side_candidates: Sequence[int] = DEFAULT_IMAGE_MAX_SIDE_CANDIDATES,
    action_blocks: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    """Construct fit-capable decoder inputs + RolloutTrace for A/B oracle."""
    attempts: List[Dict[str, Any]] = []
    chosen = None
    for max_side in image_max_side_candidates:
        tok = tokenize_pair_with_resized_image(
            processor,
            pair,
            device,
            config=config,
            image_max_side=int(max_side),
        )
        attempt = {
            "image_max_side": int(max_side),
            "prompt_len": int(tok["prompt_len"]),
            "image_token_count": int(tok["image_token_count"]),
            "image_resized_size": tok["image_resized_size"],
        }
        attempts.append(attempt)
        if int(tok["prompt_len"]) <= int(target_max_prompt_tokens):
            chosen = tok
            break
        chosen = tok  # keep shortest attempted if none meet target
    assert chosen is not None
    if int(chosen["prompt_len"]) > int(target_max_prompt_tokens):
        # Still proceed with the smallest candidate, but mark the miss.
        fit_ok = False
    else:
        fit_ok = True

    sampling = PBDSamplingConfig(
        temperature=float(config.get("rollout", {}).get("temperature", 1.0)),
        top_k=int(config.get("rollout", {}).get("top_k", 0)),
        top_p=float(config.get("rollout", {}).get("top_p", 1.0)),
        repetition_penalty=float(
            config.get("rollout", {}).get("repetition_penalty", 1.0)
        ),
        block_size=int(config.get("rollout", {}).get("block_size", 6)),
    )
    trace = build_short_rollout_trace(
        chosen["prompt_token_ids"],
        num_scored_blocks=int(num_scored_blocks),
        block_size=int(sampling.block_size),
        action_blocks=action_blocks,
        sampling=sampling,
    )
    decoder_kwargs = {
        "input_ids": chosen["input_ids"],
        "pixel_values": chosen["pixel_values"],
        "image_grid_hws": chosen["image_grid_hws"],
    }
    block0_window = int(trace.blocks[0].prefix_length) + int(sampling.block_size)
    meta = {
        "format": "short_sequence_gradient_oracle_v1",
        "purpose": "backend_gradient_validation_fixture",
        "not_gt_selection": True,
        "hybrid_semantics_unchanged": True,
        "carry_cache": True,
        "num_scored_blocks": int(num_scored_blocks),
        "block_size": int(sampling.block_size),
        "prompt_len": int(chosen["prompt_len"]),
        "image_token_count": int(chosen["image_token_count"]),
        "text_token_count": int(chosen["prompt_len"]) - int(chosen["image_token_count"]),
        "generated_len": len(trace.generated_token_ids),
        "block0_prefix_length": int(trace.blocks[0].prefix_length),
        "block0_window_length": block0_window,
        "image_original_size": chosen["image_original_size"],
        "image_resized_size": chosen["image_resized_size"],
        "image_max_side": int(chosen["image_max_side"]),
        "target_max_prompt_tokens": int(target_max_prompt_tokens),
        "prompt_len_fits_target": bool(fit_ok),
        "resize_attempts": attempts,
        "action_blocks": [list(b.action_token_ids) for b in trace.blocks],
        "executed_generated_len": len(trace.generated_token_ids),
        "executed_scored_blocks": int(num_scored_blocks),
        "oracle_block_advantages": [
            float(a) for a in DEFAULT_ORACLE_BLOCK_ADVANTAGES[: int(num_scored_blocks)]
        ],
        "oracle_loss_note": (
            "Fixed per-block advantages for backend-gradient validation only; "
            "not GT selection. Replay uses grpo_clipped_loss with "
            "old=current.detach() so PPO ratio initializes to 1."
        ),
        "production_prompt_len_reference": 1406,
        "production_image_token_count_reference": 1369,
    }
    return {
        "format": "short_sequence_gradient_oracle_v1",
        "decoder_kwargs": decoder_kwargs,
        "rollout_trace": trace,
        "meta": meta,
    }


def save_short_sequence_fixture(path: Path, fixture: Dict[str, Any]) -> Dict[str, Any]:
    """Persist CPU tensors + trace for isolated child processes."""
    path = Path(path)
    dk = fixture["decoder_kwargs"]
    payload = {
        "format": "short_sequence_gradient_oracle_v1",
        "meta": dict(fixture["meta"]),
        "rollout_trace": fixture["rollout_trace"].to_dict(),
        "decoder_kwargs_cpu": {
            "input_ids": dk["input_ids"].detach().to("cpu").contiguous(),
            "pixel_values": dk["pixel_values"].detach().to("cpu").contiguous(),
            "image_grid_hws": dk["image_grid_hws"].detach().to("cpu").contiguous(),
        },
    }
    torch.save(payload, path)
    meta = dict(payload["meta"])
    meta["fixture_path"] = str(path)
    meta["fixture_file_bytes"] = int(path.stat().st_size)
    return meta


def load_short_sequence_fixture(
    path: Path, device: torch.device
) -> Tuple[Dict[str, Any], RolloutTrace, Dict[str, Any]]:
    """Load fixture → (decoder_kwargs on device, RolloutTrace, meta)."""
    from verify_exact_replay_equivalence import _trace_from_record

    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("format") != "short_sequence_gradient_oracle_v1":
        raise RuntimeError(
            f"expected short_sequence_gradient_oracle_v1 fixture, got {payload.get('format')}"
        )
    cpu = payload["decoder_kwargs_cpu"]
    decoder_kwargs = {
        "input_ids": cpu["input_ids"].to(device=device),
        "pixel_values": cpu["pixel_values"].to(device=device, dtype=torch.bfloat16),
        "image_grid_hws": cpu["image_grid_hws"].to(device=device),
    }
    trace = _trace_from_record(payload["rollout_trace"])
    return decoder_kwargs, trace, dict(payload.get("meta") or {})
