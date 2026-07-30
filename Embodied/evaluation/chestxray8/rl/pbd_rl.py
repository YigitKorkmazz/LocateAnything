"""Probability-consistent stochastic PBD decoding for LocateAnything GRPO.

This module is intentionally separate from LocateAnything's ordinary
``decode_bbox_avg`` inference path.  Every token emitted here is sampled from
the same filtered categorical distribution whose log probability is recorded
in the rollout trace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class PBDSamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    block_size: int = 6

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("stochastic PBD-RL requires temperature > 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.block_size != 6:
            raise ValueError("LocateAnything bbox PBD requires block_size=6")


@dataclass
class SlotTrace:
    slot_index: int
    action_token_id: int
    support_kind: str
    log_prob_old: float
    support_size: int
    top_k: int
    top_p: float
    temperature: float


@dataclass
class BlockTrace:
    block_index: int
    prefix_length: int
    cache_length_before: int
    cache_length_after: int
    block_type: str
    position_ids: List[int]
    input_window_ids: List[int]
    action_token_ids: List[int]
    slots: List[SlotTrace] = field(default_factory=list)

    @property
    def old_log_prob(self) -> float:
        return float(sum(slot.log_prob_old for slot in self.slots))


@dataclass
class RolloutTrace:
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    blocks: List[BlockTrace]
    sampling: PBDSamplingConfig
    stopped_on_eos: bool
    truncated: bool
    decoded_text: Optional[str] = None

    @property
    def old_log_prob(self) -> float:
        return float(sum(block.old_log_prob for block in self.blocks))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FilteredCategorical:
    token_ids: torch.Tensor
    log_probs: torch.Tensor

    @property
    def probs(self) -> torch.Tensor:
        return self.log_probs.exp()

    def sample(self, generator: Optional[torch.Generator] = None) -> Tuple[int, torch.Tensor]:
        local_index = torch.multinomial(
            self.probs.detach(), num_samples=1, generator=generator
        ).squeeze(0)
        action = self.token_ids[local_index]
        return int(action.item()), self.log_probs[local_index]

    def log_prob(self, action_token_id: int) -> torch.Tensor:
        matches = self.token_ids.eq(int(action_token_id))
        if not bool(matches.any().item()):
            return self.log_probs.sum() * 0.0 + torch.tensor(
                float("-inf"), device=self.log_probs.device, dtype=self.log_probs.dtype
            )
        index = matches.nonzero(as_tuple=False)[0, 0]
        return self.log_probs[index]


def _apply_repetition_penalty(
    logits: torch.Tensor,
    history_ids: Sequence[int],
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0 or not history_ids:
        return logits
    out = logits.clone()
    ids = torch.tensor(
        sorted(set(int(x) for x in history_ids)),
        device=logits.device,
        dtype=torch.long,
    )
    ids = ids[(ids >= 0) & (ids < logits.numel())]
    selected = out[ids]
    out[ids] = torch.where(selected > 0, selected / penalty, selected * penalty)
    return out


def build_filtered_categorical(
    logits: torch.Tensor,
    *,
    history_ids: Sequence[int],
    config: PBDSamplingConfig,
    allowed_token_ids: Optional[Iterable[int]] = None,
) -> FilteredCategorical:
    """Build the exact categorical used for both sampling and replay.

    Restriction is applied before top-k/top-p, so coordinate top-k is computed
    only among native coordinate tokens.
    """
    config.validate()
    if logits.dim() != 1:
        raise ValueError(f"expected 1D logits, got shape={tuple(logits.shape)}")

    processed = _apply_repetition_penalty(
        logits, history_ids, config.repetition_penalty
    )
    if allowed_token_ids is None:
        token_ids = torch.arange(
            processed.numel(), device=processed.device, dtype=torch.long
        )
        selected = processed
    else:
        token_ids = torch.tensor(
            list(allowed_token_ids), device=processed.device, dtype=torch.long
        )
        if token_ids.numel() == 0:
            raise ValueError("allowed_token_ids cannot be empty")
        if int(token_ids.min()) < 0 or int(token_ids.max()) >= processed.numel():
            raise ValueError("allowed token ID outside vocabulary")
        selected = processed.index_select(0, token_ids)

    selected = selected / config.temperature

    if config.top_k > 0 and config.top_k < selected.numel():
        _, keep_local = torch.topk(selected, k=config.top_k, dim=-1)
        token_ids = token_ids.index_select(0, keep_local)
        selected = selected.index_select(0, keep_local)

    if config.top_p < 1.0 and selected.numel() > 1:
        sorted_logits, sorted_local = torch.sort(selected, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        keep = ~remove
        sorted_local = sorted_local[keep]
        token_ids = token_ids.index_select(0, sorted_local)
        selected = selected.index_select(0, sorted_local)

    return FilteredCategorical(
        token_ids=token_ids,
        log_probs=torch.log_softmax(selected.float(), dim=-1),
    )


def _support_ids(
    support_kind: str,
    token_ids: Dict[str, int],
) -> Optional[range | List[int]]:
    if support_kind == "full":
        return None
    if support_kind == "coordinate":
        return range(
            int(token_ids["coord_start_token_id"]),
            int(token_ids["coord_end_token_id"]) + 1,
        )
    if support_kind == "box_start":
        return [int(token_ids["box_start_token_id"])]
    if support_kind == "box_end":
        return [int(token_ids["box_end_token_id"])]
    raise ValueError(f"unknown support_kind={support_kind!r}")


def _sample_slot(
    logits: torch.Tensor,
    *,
    slot_index: int,
    support_kind: str,
    history_ids: Sequence[int],
    token_ids: Dict[str, int],
    config: PBDSamplingConfig,
    generator: Optional[torch.Generator],
) -> SlotTrace:
    distribution = build_filtered_categorical(
        logits,
        history_ids=history_ids,
        config=config,
        allowed_token_ids=_support_ids(support_kind, token_ids),
    )
    action, log_prob = distribution.sample(generator=generator)
    return SlotTrace(
        slot_index=slot_index,
        action_token_id=action,
        support_kind=support_kind,
        log_prob_old=float(log_prob.detach().cpu().item()),
        support_size=int(distribution.token_ids.numel()),
        top_k=config.top_k,
        top_p=config.top_p,
        temperature=config.temperature,
    )


def sample_pbd_block(
    block_logits: torch.Tensor,
    *,
    history_ids: Sequence[int],
    token_ids: Dict[str, int],
    config: PBDSamplingConfig,
    generator: Optional[torch.Generator] = None,
    force_box_block: bool = False,
) -> Tuple[str, List[int], List[SlotTrace], bool]:
    """Sample one variable-length action block directly from PBD logits."""
    if block_logits.dim() != 2 or block_logits.size(0) != config.block_size:
        raise ValueError(
            f"expected [{config.block_size}, vocab] logits, "
            f"got {tuple(block_logits.shape)}"
        )

    first_kind = "box_start" if force_box_block else "full"
    slots = [
        _sample_slot(
            block_logits[0],
            slot_index=0,
            support_kind=first_kind,
            history_ids=history_ids,
            token_ids=token_ids,
            config=config,
            generator=generator,
        )
    ]
    first = slots[0].action_token_id
    actions = [first]

    if first == int(token_ids["box_start_token_id"]):
        for slot_index in range(1, 5):
            slot = _sample_slot(
                block_logits[slot_index],
                slot_index=slot_index,
                support_kind="coordinate",
                # PBD predicts all six slots in parallel. Repetition processing
                # therefore sees only the committed prefix, never earlier actions
                # sampled from this same block.
                history_ids=history_ids,
                token_ids=token_ids,
                config=config,
                generator=generator,
            )
            slots.append(slot)
            actions.append(slot.action_token_id)
        end = _sample_slot(
            block_logits[5],
            slot_index=5,
            support_kind="box_end",
            history_ids=history_ids,
            token_ids=token_ids,
            config=config,
            generator=generator,
        )
        slots.append(end)
        actions.append(end.action_token_id)
        return "box", actions, slots, False

    eos_id = int(token_ids["im_end_token_id"])
    if first == eos_id:
        return "eos", actions, slots, True

    for slot_index in range(1, config.block_size):
        slot = _sample_slot(
            block_logits[slot_index],
            slot_index=slot_index,
            support_kind="full",
            history_ids=history_ids,
            token_ids=token_ids,
            config=config,
            generator=generator,
        )
        slots.append(slot)
        actions.append(slot.action_token_id)
        if slot.action_token_id == eos_id:
            return "text", actions, slots, True
    return "text", actions, slots, False


def score_pbd_block(
    block_logits: torch.Tensor,
    block: BlockTrace,
    *,
    history_ids: Sequence[int],
    token_ids: Dict[str, int],
    config: PBDSamplingConfig,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Replay one recorded block under current, old, or reference parameters."""
    if block.action_token_ids != [slot.action_token_id for slot in block.slots]:
        raise RuntimeError("rollout trace action/slot mismatch")
    per_slot: List[torch.Tensor] = []
    for slot in block.slots:
        if (
            slot.top_k != config.top_k
            or slot.top_p != config.top_p
            or slot.temperature != config.temperature
        ):
            raise RuntimeError("rollout slot sampling config differs from replay config")
        distribution = build_filtered_categorical(
            block_logits[slot.slot_index],
            history_ids=history_ids,
            config=config,
            allowed_token_ids=_support_ids(slot.support_kind, token_ids),
        )
        if int(distribution.token_ids.numel()) != slot.support_size:
            raise RuntimeError("replay categorical support differs from rollout trace")
        value = distribution.log_prob(slot.action_token_id)
        per_slot.append(value)
    return torch.stack(per_slot).sum(), per_slot


def resolve_token_ids(model) -> Dict[str, int]:
    if hasattr(model, "token_ids"):
        out = {k: int(v) for k, v in model.token_ids.items()}
    else:
        config = model.config
        text_config = config.text_config
        out = {
            "box_start_token_id": int(config.box_start_token_id),
            "box_end_token_id": int(config.box_end_token_id),
            "coord_start_token_id": int(config.coord_start_token_id),
            "coord_end_token_id": int(config.coord_end_token_id),
            "default_mask_token_id": int(text_config.text_mask_token_id),
            "im_end_token_id": int(text_config.eos_token_id),
        }
    required = {
        "box_start_token_id",
        "box_end_token_id",
        "coord_start_token_id",
        "coord_end_token_id",
        "default_mask_token_id",
        "im_end_token_id",
    }
    missing = required - set(out)
    if missing:
        raise RuntimeError(f"model token map missing {sorted(missing)}")
    return out


def _unwrap_lm_output(outputs):
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs


def _cache_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    return int(past_key_values[0][0].size(2))


def _truncate_legacy_cache(past_key_values, length: int):
    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError(
            "LocateAnything PBD-RL expects the pinned model's legacy tuple KV cache"
        )
    return tuple(
        (
            kv[0][:, :, :length, :],
            kv[1][:, :, :length, :],
        )
        for kv in past_key_values
    )


def _ensure_safe_image_processing(model) -> None:
    """Patch only image embedding injection; retain input_ids for PBD masking."""
    causal_lm = model.language_model
    if hasattr(causal_lm, "get_base_model"):
        causal_lm = causal_lm.get_base_model()
    decoder = causal_lm.model
    if getattr(decoder, "_pbd_rl_safe_image_processing", False):
        return

    def safe_image_processing(self, input_ids, visual_features, image_token_index):
        input_embeds = self.embed_tokens(input_ids)
        if visual_features is None:
            return input_embeds
        flat_ids = input_ids.reshape(-1)
        indices = flat_ids.eq(int(image_token_index)).nonzero(
            as_tuple=False
        ).squeeze(-1)
        features = visual_features.reshape(-1, input_embeds.size(-1)).to(
            device=input_embeds.device, dtype=input_embeds.dtype
        )
        if indices.numel() != features.size(0):
            raise RuntimeError(
                "image token/feature count mismatch during PBD-RL replay: "
                f"{indices.numel()} tokens vs {features.size(0)} features"
            )
        flat_embeds = input_embeds.reshape(-1, input_embeds.size(-1))
        return torch.index_copy(flat_embeds, 0, indices, features).reshape_as(
            input_embeds
        )

    decoder.image_processing = MethodType(safe_image_processing, decoder)
    decoder._pbd_rl_safe_image_processing = True


def _forward_language_model(model, prepared, visual_features=None):
    if visual_features is None:
        return model.language_model(**prepared)
    _ensure_safe_image_processing(model)
    model_inputs = dict(prepared)
    model_inputs["visual_features"] = visual_features
    model_inputs["image_token_index"] = model.config.image_token_index
    return model.language_model(**model_inputs)


class StochasticPBDRLDecoder:
    """Native PBD rollout decoder with action/log-prob identity."""

    def __init__(
        self,
        model,
        tokenizer,
        sampling: Optional[PBDSamplingConfig] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sampling = sampling or PBDSamplingConfig()
        self.sampling.validate()
        self.token_ids = resolve_token_ids(model)

    def _visual_features(self, pixel_values, image_grid_hws):
        pixel_values = pixel_values.to(self.model.language_model.dtype)
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(
                pixel_values.device, dtype=torch.int32
            )
        features = self.model.extract_feature(pixel_values, image_grid_hws)
        if image_grid_hws is not None:
            features = self.model.mlp1(torch.cat(features, dim=0))
        return features, image_grid_hws

    def generate(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
        max_new_tokens: int,
        seed: int,
        force_first_box_block: bool = False,
    ) -> RolloutTrace:
        if input_ids.size(0) != 1:
            raise ValueError("stochastic PBD-RL currently requires batch size 1")
        self.model.eval()
        device = input_ids.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        block_size = self.sampling.block_size
        mask_tail = torch.full(
            (1, block_size - 1),
            int(self.token_ids["default_mask_token_id"]),
            dtype=input_ids.dtype,
            device=device,
        )
        generated = input_ids.clone()
        prompt_length = int(input_ids.size(1))
        total_length = min(
            int(self.tokenizer.model_max_length), prompt_length + int(max_new_tokens)
        )
        full_positions = torch.arange(
            total_length + block_size, device=device
        ).unsqueeze(0)
        visual_features, _ = self._visual_features(pixel_values, image_grid_hws)
        past_key_values = None
        blocks: List[BlockTrace] = []
        stopped = False
        budget_exhausted = False

        with torch.no_grad():
            while generated.size(1) < total_length and not stopped:
                prefix_length = int(generated.size(1))
                cache_before = _cache_length(past_key_values)
                window = torch.cat(
                    (generated, generated[:, -1:].clone(), mask_tail), dim=1
                )
                start = cache_before
                position_ids = full_positions[:, start : window.size(1)].clone()
                position_ids[0, -block_size:] -= 1
                prepared = self.model.language_model.prepare_inputs_for_generation(
                    window,
                    past_key_values,
                    None,
                    inputs_embeds=None,
                    use_cache=True,
                    position_ids=position_ids,
                )
                outputs = _unwrap_lm_output(
                    _forward_language_model(
                        self.model,
                        prepared,
                        visual_features=visual_features if not blocks else None,
                    )
                )
                block_logits = outputs.logits[0, -block_size:, :]
                block_type, actions, slots, stopped = sample_pbd_block(
                    block_logits,
                    history_ids=generated[0].tolist(),
                    token_ids=self.token_ids,
                    config=self.sampling,
                    generator=generator,
                    force_box_block=force_first_box_block and not blocks,
                )
                remaining = total_length - int(generated.size(1))
                if len(actions) > remaining:
                    budget_exhausted = True
                    break
                past_key_values = _truncate_legacy_cache(
                    outputs.past_key_values, prefix_length
                )
                cache_after = _cache_length(past_key_values)
                generated = torch.cat(
                    [
                        generated,
                        torch.tensor(actions, device=device, dtype=generated.dtype)
                        .unsqueeze(0),
                    ],
                    dim=1,
                )
                blocks.append(
                    BlockTrace(
                        block_index=len(blocks),
                        prefix_length=prefix_length,
                        cache_length_before=cache_before,
                        cache_length_after=cache_after,
                        block_type=block_type,
                        position_ids=position_ids[0].detach().cpu().tolist(),
                        input_window_ids=prepared["input_ids"][0]
                        .detach()
                        .cpu()
                        .tolist(),
                        action_token_ids=list(actions),
                        slots=slots,
                    )
                )

        generated_ids = generated[0, prompt_length:].detach().cpu().tolist()
        truncated = not stopped and (
            budget_exhausted or int(generated.size(1)) >= total_length
        )
        return RolloutTrace(
            prompt_token_ids=input_ids[0].detach().cpu().tolist(),
            generated_token_ids=generated_ids,
            blocks=blocks,
            sampling=self.sampling,
            stopped_on_eos=stopped,
            truncated=truncated,
            decoded_text=self.tokenizer.decode(
                generated_ids, skip_special_tokens=False
            ),
        )


class PBDRolloutReplayer:
    """Recompute a rollout's exact PBD action probability under a model."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.token_ids = resolve_token_ids(model)

    def score(
        self,
        trace: RolloutTrace,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if input_ids[0].tolist() != trace.prompt_token_ids:
            raise RuntimeError("replay prompt does not match rollout trace")
        self.model.eval()
        block_size = trace.sampling.block_size
        device = input_ids.device
        generated = input_ids.clone()
        mask_tail = torch.full(
            (1, block_size - 1),
            int(self.token_ids["default_mask_token_id"]),
            dtype=input_ids.dtype,
            device=device,
        )
        max_length = len(trace.prompt_token_ids) + len(trace.generated_token_ids)
        full_positions = torch.arange(
            max_length + block_size, device=device
        ).unsqueeze(0)
        pixel_values = pixel_values.to(self.model.language_model.dtype)
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(
                pixel_values.device, dtype=torch.int32
            )
        visual_features = self.model.extract_feature(pixel_values, image_grid_hws)
        if image_grid_hws is not None:
            visual_features = self.model.mlp1(torch.cat(visual_features, dim=0))
        past_key_values = None
        block_logps: List[torch.Tensor] = []

        for block in trace.blocks:
            if int(generated.size(1)) != block.prefix_length:
                raise RuntimeError("replay prefix length differs from trace")
            cache_before = _cache_length(past_key_values)
            if cache_before != block.cache_length_before:
                raise RuntimeError("replay cache length differs from rollout trace")
            window = torch.cat(
                (generated, generated[:, -1:].clone(), mask_tail), dim=1
            )
            position_ids = full_positions[:, cache_before : window.size(1)].clone()
            position_ids[0, -block_size:] -= 1
            prepared = self.model.language_model.prepare_inputs_for_generation(
                window,
                past_key_values,
                None,
                inputs_embeds=None,
                use_cache=True,
                position_ids=position_ids,
            )
            if position_ids[0].tolist() != block.position_ids:
                raise RuntimeError("replay position IDs differ from rollout trace")
            if prepared["input_ids"][0].tolist() != block.input_window_ids:
                raise RuntimeError("replay input window differs from rollout trace")
            outputs = _unwrap_lm_output(
                _forward_language_model(
                    self.model,
                    prepared,
                    visual_features=visual_features if not block_logps else None,
                )
            )
            block_logits = outputs.logits[0, -block_size:, :]
            block_logp, _ = score_pbd_block(
                block_logits,
                block,
                history_ids=generated[0].tolist(),
                token_ids=self.token_ids,
                config=trace.sampling,
            )
            block_logps.append(block_logp)
            prefix_length = int(generated.size(1))
            past_key_values = _truncate_legacy_cache(
                outputs.past_key_values, prefix_length
            )
            if _cache_length(past_key_values) != block.cache_length_after:
                raise RuntimeError("replay output cache length differs from rollout trace")
            generated = torch.cat(
                [
                    generated,
                    torch.tensor(
                        block.action_token_ids,
                        device=device,
                        dtype=generated.dtype,
                    ).unsqueeze(0),
                ],
                dim=1,
            )

        if generated[0, len(trace.prompt_token_ids) :].tolist() != trace.generated_token_ids:
            raise RuntimeError("replayed actions differ from emitted rollout tokens")
        if not block_logps:
            return torch.zeros((), device=device, requires_grad=True), []
        return torch.stack(block_logps).sum(), block_logps

    def score_one_block_group(
        self,
        traces: Sequence[RolloutTrace],
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        image_grid_hws,
    ) -> torch.Tensor:
        """Score G one-block rollouts with one shared model forward."""
        if not traces or any(len(trace.blocks) != 1 for trace in traces):
            raise ValueError("group replay requires nonempty one-block traces")
        first = traces[0]
        if any(trace.prompt_token_ids != first.prompt_token_ids for trace in traces):
            raise RuntimeError("group replay traces have different prompts")
        if input_ids[0].tolist() != first.prompt_token_ids:
            raise RuntimeError("group replay prompt does not match rollout traces")
        self.model.eval()
        block_size = first.sampling.block_size
        device = input_ids.device
        mask_tail = torch.full(
            (1, block_size - 1),
            int(self.token_ids["default_mask_token_id"]),
            dtype=input_ids.dtype,
            device=device,
        )
        window = torch.cat((input_ids, input_ids[:, -1:].clone(), mask_tail), dim=1)
        position_ids = torch.arange(
            window.size(1) + block_size, device=device
        ).unsqueeze(0)[:, : window.size(1)]
        position_ids[0, -block_size:] -= 1
        prepared = self.model.language_model.prepare_inputs_for_generation(
            window,
            None,
            None,
            inputs_embeds=None,
            use_cache=True,
            position_ids=position_ids,
        )
        for trace in traces:
            block = trace.blocks[0]
            if trace.sampling != first.sampling:
                raise RuntimeError("group replay sampling configs differ")
            if block.prefix_length != input_ids.size(1):
                raise RuntimeError("group replay prefix length differs from trace")
            if block.cache_length_before != 0:
                raise RuntimeError("first rollout block unexpectedly has a cache")
            if position_ids[0].tolist() != block.position_ids:
                raise RuntimeError("group replay position IDs differ from trace")
            if prepared["input_ids"][0].tolist() != block.input_window_ids:
                raise RuntimeError("group replay input window differs from trace")

        visual_features, _ = StochasticPBDRLDecoder(
            self.model, self.tokenizer, first.sampling
        )._visual_features(pixel_values, image_grid_hws)
        outputs = _unwrap_lm_output(
            _forward_language_model(self.model, prepared, visual_features)
        )
        block_logits = outputs.logits[0, -block_size:, :]
        values = [
            score_pbd_block(
                block_logits,
                trace.blocks[0],
                history_ids=input_ids[0].tolist(),
                token_ids=self.token_ids,
                config=trace.sampling,
            )[0]
            for trace in traces
        ]
        return torch.stack(values)
