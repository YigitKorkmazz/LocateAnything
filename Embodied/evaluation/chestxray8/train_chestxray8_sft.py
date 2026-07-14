#!/usr/bin/env python3
"""
ChestX-ray8 supervised fine-tuning for LocateAnything-3B.

Prompt (default): ``Locate the {Disease} in this chest X-ray``
Target: ``<ref>{Disease}</ref><box><x1><y1><x2><y2></box>...``

Experiment types:
  - full_sft       : all intended model parameters trainable
  - lora           : freeze base (vision/mlp1/embeds); LoRA on LLM attention + MLP
  - lora_projector : same LoRA + fully fine-tuned mlp1 (dual LR)

Designed for single NVIDIA RTX A4000 (~16 GB). Does not require DeepSpeed or
MagiAttention (uses sdpa). Reuses LocateAnything model/processor/tokenization
and assistant-only label masking (loss on assistant tokens only).

Default LRs (shown explicitly; override via CLI):
  full_sft       : 2e-5  (matches official LocateAnything continual-SFT default)
  lora           : 2e-5  (matches official LocateAnything LoRA scripts; 1e-4 collapsed)
  lora_projector : LoRA 2e-5 + fully unfrozen mlp1 at 1e-5 (vision / embeds frozen)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CHEST_DIR))

from eaglevl.train.constants import (  # noqa: E402
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    NULL_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    TEXT_MASK_TOKEN,
    number_tokens_list,
    special_tokens_list,
)
from sft_common import (  # noqa: E402
    DEFAULT_FULL_SFT_LR,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_LR,
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MODEL_NAME,
    DEFAULT_PROJECTOR_LR,
    DEFAULT_SEED,
    IGNORE_INDEX,
    LLM_LORA_TARGET_MODULES,
    PROMPT_STRATEGY,
    collect_reproducibility_info,
    gpu_mem_mb,
    process_image_sft_sample,
    read_jsonl,
    refresh_pair_prompt_fields,
    set_global_seed,
)


logger = logging.getLogger("chestxray8_sft")


# ---------------------------------------------------------------------------
# Model / processor loading
# ---------------------------------------------------------------------------

def load_tokenizer_and_processor(
    model_path: str,
    revision: Optional[str] = None,
    max_seq_length: int = 4096,
):
    from transformers import AutoProcessor, AutoTokenizer

    if revision is None:
        revision = PINNED_MODEL_REVISION

    tok_kwargs = dict(
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
        revision=revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    tokenizer.model_max_length = max_seq_length
    tokenizer.add_tokens(special_tokens_list + number_tokens_list, special_tokens=True)
    if len(tokenizer.encode("assistant")) > 1:
        tokenizer.add_tokens(["assistant"], special_tokens=False)

    proc_kwargs = dict(trust_remote_code=True, use_fast=True, revision=revision)
    try:
        processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)
        processor.tokenizer = tokenizer
    except Exception as e:  # noqa: BLE001
        logger.warning("AutoProcessor failed (%s); building from local configs", e)
        from eaglevl.utils.locany.image_processing_locateanything import (
            LocateAnythingImageProcessor,
        )
        from eaglevl.utils.locany.processing_locateanything import LocateAnythingProcessor

        utils_dir = REPO_ROOT / "eaglevl" / "utils" / "locany"

        def _load_json(path: Path) -> Dict[str, Any]:
            return json.loads(path.read_text(encoding="utf-8"))

        chat_template_data = _load_json(utils_dir / "chat_template.json")
        processor_config = _load_json(utils_dir / "processor_config.json")
        preprocessor_config = _load_json(utils_dir / "preprocessor_config.json")
        image_processor = LocateAnythingImageProcessor(**preprocessor_config)
        processor_config["chat_template"] = chat_template_data["chat_template"]
        processor = LocateAnythingProcessor(
            tokenizer=tokenizer, image_processor=image_processor, **processor_config
        )
    return tokenizer, processor


def load_locateanything_model(
    model_path: str,
    tokenizer,
    revision: Optional[str] = None,
    attn_implementation: str = "sdpa",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load LocateAnything via HF remote code (has flash/magi -> sdpa fallbacks).

    The local ``eaglevl`` training copy expects flash_attn/magi/liger and is not
    suitable for RTX A4000 without those packages. The published HF model
    revision used by ``LocateAnythingWorker`` is the correct inference/train base.
    """
    from transformers import AutoModel

    # Pin the revision that is already cached / used by zero-shot eval when possible.
    if revision is None:
        revision = PINNED_MODEL_REVISION

    logger.info(
        "Loading AutoModel %s (revision=%s, requested_attn=%s, HF_HOME=%s)",
        model_path,
        revision,
        attn_implementation,
        os.environ.get("HF_HOME", "<unset>"),
    )
    model = AutoModel.from_pretrained(
        model_path,
        revision=revision,
        dtype=torch_dtype,
        trust_remote_code=True,
    )

    # Align special-token / image-token indices with the training tokenizer.
    image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.image_token_index = image_token_index
    if hasattr(model, "config"):
        model.config.image_token_index = image_token_index
        model.config.use_cache = False
    if hasattr(model, "language_model"):
        # Only resize when vocab actually grew. Calling resize with the same size
        # still marks embeddings as "resized" for PEFT, which then dumps the full
        # embed_tokens + lm_head (~1.3GB) into every adapter checkpoint.
        current_vocab = int(model.language_model.get_input_embeddings().weight.size(0))
        target_vocab = len(tokenizer)
        if current_vocab != target_vocab:
            logger.info(
                "Resizing language_model embeddings %d -> %d", current_vocab, target_vocab
            )
            model.language_model.resize_token_embeddings(target_vocab)
        else:
            logger.info(
                "Skipping resize_token_embeddings (vocab already %d)", current_vocab
            )
        if hasattr(model.language_model, "config"):
            model.language_model.config.use_cache = False
            model.language_model.config.vocab_size = target_vocab
        if hasattr(model.config, "text_config"):
            model.config.text_config.vocab_size = target_vocab

    # Ensure LoRA flag starts off; we apply adapters explicitly for the lora experiment.
    model.use_llm_lora = False

    # Runtime fix for HF Qwen2 training return when labels is None.
    pos_loss_patch = patch_qwen2_pos_loss_list_bug(model)
    logger.info("pos_loss_list patch: %s", pos_loss_patch)
    return model, revision


PINNED_MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"


def resolve_qwen2_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying Qwen2ForCausalLM module (unwrap PEFT if needed)."""
    lm = getattr(model, "language_model", model)
    # PeftModel.get_base_model() -> Qwen2ForCausalLM
    if hasattr(lm, "get_base_model"):
        try:
            return lm.get_base_model()
        except Exception:
            pass
    # PeftModel.base_model is LoraModel; LoraModel.model is Qwen2ForCausalLM
    if hasattr(lm, "base_model") and hasattr(lm.base_model, "model"):
        inner = lm.base_model.model
        if inner.__class__.__name__.endswith("ForCausalLM"):
            return inner
    return lm


def patch_qwen2_pos_loss_list_bug(model: torch.nn.Module) -> Dict[str, Any]:
    """Monkey-patch Qwen2ForCausalLM.forward to always define ``pos_loss_list``.

    Bug analysis (revision c32291ca5e996f5a7a485845b4f57a233936bba0):
      - ``pos_loss_list`` is assigned ONLY inside ``if labels is not None:``
        as ``torch.zeros(max_n_future_tokens, device=...)``.
      - When ``self.training`` is True the method ALWAYS returns
        ``(CausalLMOutputWithPast(...), pos_loss_list)``.
      - LocateAnything.forward (and our training forward) call
        ``language_model(..., labels=None)`` — labels stay on the outer model.
      - Training path: ``training=True`` + ``labels=None`` → UnboundLocalError.
      - Inference path: ``model.eval()`` → ``training=False`` → returns only
        ``CausalLMOutputWithPast`` and never touches ``pos_loss_list`` → OK.

    Safe default: ``None`` (no auxiliary MTP position losses when labels were
    not provided). When labels ARE provided the original tensor assignment
    still overwrites this default. Downstream LocateAnything callers only use
    ``outputs.logits`` / unwrap tuples; they do not consume ``pos_loss_list``.
    """
    import inspect
    import re
    import textwrap

    qwen = resolve_qwen2_causal_lm(model)
    cls = qwen.__class__
    report: Dict[str, Any] = {
        "qwen_class": f"{cls.__module__}.{cls.__name__}",
        "patched": False,
        "already_patched": False,
        "method": None,
    }
    if getattr(cls, "_chestxray8_pos_loss_patched", False):
        report["already_patched"] = True
        report["patched"] = True
        report["method"] = "idempotent_skip"
        return report

    original_forward = cls.forward

    # Preferred: rewrite source from the on-disk modeling_qwen2.py (HF remote-code
    # modules are unreliable with inspect.getsource).
    try:
        module = inspect.getmodule(cls) or sys.modules.get(cls.__module__)
        mod_file = getattr(module, "__file__", None) if module is not None else None
        if not mod_file:
            raise RuntimeError(f"module file not found for {cls.__module__}")
        file_text = Path(mod_file).read_text(encoding="utf-8")
        marker = "class Qwen2ForCausalLM"
        start = file_text.find(marker)
        if start < 0:
            raise RuntimeError(f"Qwen2ForCausalLM not found in {mod_file}")
        fwd_start = file_text.find("\n    def forward(", start)
        if fwd_start < 0:
            raise RuntimeError(f"forward() not found in Qwen2ForCausalLM in {mod_file}")
        end_marker = "\n    def prepare_inputs_for_generation"
        end = file_text.find(end_marker, fwd_start)
        if end < 0:
            raise RuntimeError("Could not bound forward() method in modeling_qwen2.py")
        raw_src = file_text[fwd_start + 1 : end]
        src = textwrap.dedent(raw_src)
        logger.info("Loaded Qwen2.forward source from file %s (%d chars)", mod_file, len(src))

        # Already safe?
        m_init = re.search(
            r"pos_loss_list\s*=\s*None\s*\n\s*if labels is not None:",
            src,
        )
        if m_init:
            report["already_patched"] = True
            report["patched"] = True
            report["method"] = "source_already_safe"
            cls._chestxray8_pos_loss_patched = True
            return report

        pattern = re.compile(
            r"(?P<indent>[ \t]*)loss = None\n(?P=indent)if labels is not None:"
        )
        match = pattern.search(src)
        if not match:
            raise RuntimeError(
                "Could not locate 'loss = None / if labels is not None' anchor in Qwen2.forward"
            )
        indent = match.group("indent")
        replacement = (
            f"{indent}loss = None\n"
            f"{indent}# ChestX-ray8 SFT monkey-patch: always define pos_loss_list.\n"
            f"{indent}# None == no aux MTP position losses when labels were not passed.\n"
            f"{indent}pos_loss_list = None\n"
            f"{indent}if labels is not None:"
        )
        new_src, nsub = pattern.subn(replacement, src, count=1)
        if nsub != 1:
            raise RuntimeError(f"Expected 1 substitution, got {nsub}")

        module = inspect.getmodule(cls)
        if module is None:
            raise RuntimeError("Could not resolve Qwen2 module for exec patch")
        ns: Dict[str, Any] = dict(module.__dict__)
        exec(
            compile(
                new_src,
                filename=f"<chestxray8_patch:{cls.__name__}.forward>",
                mode="exec",
            ),
            ns,
        )
        if "forward" not in ns:
            raise RuntimeError("Patched forward not found after exec")
        cls.forward = ns["forward"]
        cls._chestxray8_pos_loss_original_forward = original_forward
        cls._chestxray8_pos_loss_patched = True
        report["patched"] = True
        report["method"] = "source_rewrite_exec"
        logger.info(
            "Patched %s.forward: initialize pos_loss_list=None before labels branch",
            report["qwen_class"],
        )
        return report
    except Exception as e:  # noqa: BLE001
        logger.warning("Source-rewrite pos_loss_list patch failed (%s); using wrapper", e)

    # Fallback: wrap forward and normalize the training return without disabling
    # dropout for the whole forward body. We re-implement only the return glue by
    # catching UnboundLocalError is too late; instead call original under a
    # local that pre-binds via an injected wrapper around the training flag for
    # the RETURN only — not possible without source rewrite.
    #
    # Correct fallback that preserves train-mode dropout: call the inner
    # transformer + lm_head ourselves when labels is None.
    def safe_forward(self, *args, **kwargs):
        labels = kwargs.get("labels", None)
        if self.training and labels is None:
            # Mirror Qwen2ForCausalLM.forward up to logits, then return a plain
            # CausalLMOutput (no pos_loss_list). Keep self.training True so
            # dropout in LoRA layers remains active.
            from transformers.modeling_outputs import CausalLMOutputWithPast

            output_attentions = kwargs.get("output_attentions", None)
            output_hidden_states = kwargs.get("output_hidden_states", None)
            return_dict = kwargs.get("return_dict", None)
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            output_hidden_states = (
                output_hidden_states
                if output_hidden_states is not None
                else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.model(
                input_ids=kwargs.get("input_ids", args[0] if args else None),
                attention_mask=kwargs.get("attention_mask"),
                position_ids=kwargs.get("position_ids"),
                past_key_values=kwargs.get("past_key_values"),
                inputs_embeds=kwargs.get("inputs_embeds"),
                use_cache=kwargs.get("use_cache"),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                visual_features=kwargs.get("visual_features"),
                image_token_index=kwargs.get("image_token_index"),
            )
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states).float()
            if not return_dict:
                return (logits,) + outputs[1:]
            # Match the training return signature with a safe pos_loss_list=None.
            return (
                CausalLMOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                ),
                None,
            )
        return original_forward(self, *args, **kwargs)

    cls.forward = safe_forward
    cls._chestxray8_pos_loss_original_forward = original_forward
    cls._chestxray8_pos_loss_patched = True
    report["patched"] = True
    report["method"] = "labels_none_inner_forward_wrapper"
    logger.info(
        "Patched %s.forward via labels=None inner-forward wrapper (fallback)",
        report["qwen_class"],
    )
    return report


# Keep old name as alias for callers.
_patch_qwen2_pos_loss_list_bug = patch_qwen2_pos_loss_list_bug


# ---------------------------------------------------------------------------
# Tokenization / labels
# ---------------------------------------------------------------------------

def messages_from_pair(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Always rebuild prompt/target from the active strategy so stale split
    # jsonl text cannot silently train the wrong query format.
    pair = refresh_pair_prompt_fields(pair, PROMPT_STRATEGY)
    sample = {
        "conversations": [
            {"from": "human", "value": f"<image-1>{pair['user_query']}"},
            {"from": "gpt", "value": pair["assistant_target"]},
        ],
        "image": pair["image_path"],
    }
    return process_image_sft_sample(sample, media_root="")


def tokenize_messages(processor, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    message_text = processor.py_apply_chat_template(messages, tokenize=False)
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=message_text,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    return inputs


def build_assistant_only_labels(
    input_ids: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """Mask all tokens except assistant response (loss on assistant only).

    Mirrors the assistant-span logic in the official training dataset, without
    MTP/packing blocks (appropriate for single-GPU AR fine-tuning / inference).
    """
    if input_ids.dim() == 2:
        assert input_ids.size(0) == 1
        ids = input_ids[0]
    else:
        ids = input_ids

    labels = ids.clone()
    targets_flag = torch.zeros_like(ids)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    start_header_idxs = torch.where(ids == im_start_id)[0]
    assistant_idxs = torch.where(ids == assistant_id)[0]
    eot_idxs = torch.where(ids == eos_id)[0]

    header_followers = set((start_header_idxs + 1).tolist())
    for assistant_idx in assistant_idxs.tolist():
        if assistant_idx not in header_followers:
            continue
        st = assistant_idx + 1  # usually newline after 'assistant'
        for eot_idx in eot_idxs.tolist():
            if eot_idx > st:
                # Supervise tokens from after the assistant header through <|im_end|>.
                targets_flag[st + 1 : eot_idx + 1] = 1
                break

    if int(targets_flag.sum().item()) == 0:
        raise RuntimeError("No assistant tokens found for supervision")

    labels[targets_flag == 0] = IGNORE_INDEX
    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChestXray8SFTDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Dict[str, Any]],
        processor,
        tokenizer,
        max_seq_length: int = 4096,
    ):
        self.pairs = list(pairs)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        messages = messages_from_pair(pair)
        inputs = tokenize_messages(self.processor, messages)
        input_ids = inputs["input_ids"][0]
        if input_ids.numel() > self.max_seq_length:
            raise RuntimeError(
                f"Sequence length {input_ids.numel()} exceeds max_seq_length={self.max_seq_length}"
            )
        labels = build_assistant_only_labels(input_ids, self.tokenizer)
        pixel_values = inputs["pixel_values"]
        image_grid_hws = inputs["image_grid_hws"]
        image_flags = torch.tensor([len(image_grid_hws)], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
            "image_flags": image_flags,
            "meta": {
                "image_index": pair["image_index"],
                "disease": pair["disease"],
                "patient_id": pair["patient_id"],
            },
        }


def collate_batch(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batch size is expected to be 1 on A4000; keep logic general for padding."""
    assert len(features) >= 1
    if len(features) == 1:
        f = features[0]
        return {
            "input_ids": f["input_ids"].unsqueeze(0),
            "labels": f["labels"].unsqueeze(0),
            "attention_mask": f["attention_mask"].unsqueeze(0),
            "pixel_values": f["pixel_values"],
            "image_grid_hws": f["image_grid_hws"],
            "image_flags": f["image_flags"],
            "meta": [f["meta"]],
        }

    max_len = max(f["input_ids"].numel() for f in features)
    pad_id = 0
    input_ids, labels, attn = [], [], []
    pixel_values, grids, flags, metas = [], [], [], []
    for f in features:
        L = f["input_ids"].numel()
        pad = max_len - L
        input_ids.append(
            torch.nn.functional.pad(f["input_ids"], (0, pad), value=pad_id)
        )
        labels.append(
            torch.nn.functional.pad(f["labels"], (0, pad), value=IGNORE_INDEX)
        )
        attn.append(
            torch.nn.functional.pad(f["attention_mask"], (0, pad), value=0)
        )
        pixel_values.append(f["pixel_values"])
        grids.append(torch.as_tensor(f["image_grid_hws"]))
        flags.append(f["image_flags"])
        metas.append(f["meta"])
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attn),
        "pixel_values": torch.cat(pixel_values, dim=0),
        "image_grid_hws": torch.cat(grids, dim=0),
        "image_flags": torch.cat(flags, dim=0),
        "meta": metas,
    }


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------

def discover_linear_module_names(model: torch.nn.Module) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names.append(name)
    return names


def match_lora_targets(
    model: torch.nn.Module,
    target_modules: Sequence[str],
    scope_prefix: str = "language_model",
) -> Tuple[List[str], List[str]]:
    """Return (matched_leaf_names, excluded_linear_names) under scope_prefix."""
    matched: List[str] = []
    excluded: List[str] = []
    targets = set(target_modules)
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if scope_prefix and scope_prefix not in name:
            excluded.append(name)
            continue
        leaf = name.split(".")[-1]
        # Match either full suffix (self_attn.q_proj) or leaf (q_proj) within attn/mlp.
        suffix_ok = any(name.endswith(t) or name.endswith("." + t) for t in targets)
        # Also accept dotted target patterns like self_attn.q_proj
        pattern_ok = any(
            (f".{t}" in name or name.endswith(t)) for t in targets
        )
        if suffix_ok or pattern_ok:
            # Restrict to attention / mlp blocks only.
            if ".self_attn." in name or ".mlp." in name:
                matched.append(name)
            else:
                excluded.append(name)
        else:
            excluded.append(name)
    return matched, excluded


def patch_qwen_force_causal_attn_for_sft(model: torch.nn.Module) -> Dict[str, Any]:
    """Force standard causal attention for AR SFT (disable MTP block masks).

    LocateAnything's Qwen2 training path builds an MTP block-diffusion mask
    whenever ``Qwen2Model.training`` is True. Empirically that mask yields
    **exactly zero** gradients into image-token embeddings, so ``mlp1`` cannot
    learn. ChestX-ray SFT is ordinary autoregressive supervision, so we replace
    ``create_block_diff_mask_by_pe_4d`` with a standard 4D causal mask builder.
    """
    import inspect as _inspect

    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

    qwen_causal = resolve_qwen2_causal_lm(model)
    qwen_model = qwen_causal.model  # Qwen2Model
    if getattr(qwen_model, "_chestxray8_causal_sft_mask_patched", False):
        return {"patched": True, "already_patched": True}

    # modeling_qwen2 imports create_block_diff_mask_by_pe_4d into its module dict.
    qwen_mod = _inspect.getmodule(qwen_model.forward)
    if qwen_mod is None or not hasattr(qwen_mod, "create_block_diff_mask_by_pe_4d"):
        # Fallback: resolve via transformers_modules path used by this revision.
        import transformers_modules.nvidia.LocateAnything_hyphen_3B.c32291ca5e996f5a7a485845b4f57a233936bba0.modeling_qwen2 as qwen_mod

    original_create = qwen_mod.create_block_diff_mask_by_pe_4d

    def create_causal_mask_by_pe_4d(
        block_size=None,
        x0_len_list=None,
        position_ids=None,
        causal_attn=None,
        **kwargs,
    ):
        # position_ids: (batch, seq)
        if position_ids is None:
            raise RuntimeError("causal SFT mask patch requires position_ids")
        batch_size, seq_length = position_ids.shape
        device = position_ids.device
        # Dummy embeds tensor for dtype/device in mask helper.
        dummy = torch.zeros(batch_size, seq_length, 1, device=device, dtype=torch.bfloat16)
        causal = _prepare_4d_causal_attention_mask(
            None,
            (batch_size, seq_length),
            dummy,
            0,
            sliding_window=getattr(qwen_model.config, "sliding_window", None),
        )
        return causal, None

    qwen_mod.create_block_diff_mask_by_pe_4d = create_causal_mask_by_pe_4d
    qwen_model._chestxray8_causal_sft_mask_patched = True
    qwen_model._chestxray8_original_block_mask_fn = original_create
    return {
        "patched": True,
        "already_patched": False,
        "qwen_module": getattr(qwen_mod, "__name__", str(qwen_mod)),
        "method": "replace_create_block_diff_mask_by_pe_4d_with_causal",
    }


def patch_lora_embedding_inplace_fix(model: torch.nn.Module) -> None:
    """Make vision-token injection safe under LoRA input-grad hooks.

    Prefer non-inplace scatter so autograd accepts embedding outputs that require
    grad, and keep ``mlp1`` on the loss graph when embed_tokens are frozen.
    """
    cls = model.__class__
    if getattr(cls, "_chestxray8_embed_clone_patched", False):
        return

    import inspect
    import textwrap

    original_forward = cls.forward

    def forward(self, pixel_values, input_ids=None, attention_mask=None, position_ids=None,
                image_grid_hws=None, image_flags=None, past_key_values=None, labels=None,
                use_cache=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        has_images = image_flags is not None and image_flags.sum() > 0
        vit_embeds = self.extract_feature(pixel_values, image_grid_hws)
        B, N, C = input_embeds.shape

        vit_proj = None
        if has_images:
            filtered_vit_embeds = []
            idx = 0
            for flag in image_flags:
                flag_val = int(flag.item()) if hasattr(flag, "item") else int(flag)
                if flag_val != 0:
                    filtered_vit_embeds.extend(vit_embeds[idx:idx + flag_val])
                    idx += flag_val
                else:
                    idx += 1
            if filtered_vit_embeds:
                vit_cat = torch.cat(filtered_vit_embeds, dim=0)
                vit_proj = self.mlp1(vit_cat)
        elif vit_embeds:
            vit_cat = torch.cat(vit_embeds, dim=0)
            vit_proj = self.mlp1(vit_cat)

        if vit_proj is not None:
            flat_ids = input_ids.reshape(B * N)
            selected = flat_ids == self.image_token_index
            n_img = int(selected.sum().item())
            if n_img > 0:
                vit_proj = vit_proj[:n_img].to(dtype=input_embeds.dtype)
                flat = input_embeds.reshape(B * N, C)
                idx = torch.nonzero(selected, as_tuple=False).squeeze(-1)
                mask_f = selected.unsqueeze(-1).to(dtype=flat.dtype)
                # Scatter projector outputs into image-token rows; keep text rows
                # detached (frozen embed_tokens). torch.scatter keeps vit_proj in graph
                # when mlp1 is trainable.
                padded = flat.new_zeros(flat.shape)
                padded = padded.scatter(
                    0, idx.unsqueeze(-1).expand(-1, C), vit_proj
                )
                merged = flat.detach() * (1.0 - mask_f) + padded
                input_embeds = merged.reshape(B, N, C)
                # mlp1 training requires image-token embeds on the autograd graph.
                # LoRA-only intentionally freezes mlp1/vision, so merged embeds may
                # (and should) have requires_grad=False; LoRA adapters still receive
                # grads inside the language model.
                mlp1_trainable = any(
                    p.requires_grad for p in self.mlp1.parameters()
                )
                if mlp1_trainable and not input_embeds.requires_grad:
                    raise RuntimeError(
                        "Merged input_embeds do not require grad; mlp1 would get zero "
                        f"gradients. vit_proj.requires_grad={vit_proj.requires_grad} "
                        f"padded.requires_grad={padded.requires_grad}"
                    )

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        logits = outputs.logits
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1).to(shift_logits.device),
            )
        if not return_dict:
            output = (logits,) + (outputs.past_key_values,)
            return ((loss,) + output) if loss is not None else output
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    cls.forward = forward
    cls._chestxray8_embed_clone_patched = True
    _ = original_forward, inspect, textwrap

def apply_llm_lora(
    model: torch.nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
) -> Dict[str, Any]:
    """Apply LoRA to Qwen2 attention + MLP only (vision / mlp1 / embeds frozen).

    Uses an explicit PEFT config so ``target_modules`` is honored. The stock
    ``wrap_llm_lora`` helper hard-codes the same module names but ignores any
    caller-provided target list, which made the LoRA-only setup easy to
    silently drift from the intended attn+MLP contract.
    """
    from peft import LoraConfig, get_peft_model

    # Freeze everything first (including mlp1 / vision / embeds / lm_head).
    for p in model.parameters():
        p.requires_grad = False

    matched, excluded = match_lora_targets(
        model, target_modules, scope_prefix="language_model"
    )
    if not matched:
        raise RuntimeError(
            f"No LoRA target modules matched. Targets={list(target_modules)}"
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()
    model.use_llm_lora = True

    patch_lora_embedding_inplace_fix(model)

    report = verify_lora_trainability(model, matched)
    report["matched_modules"] = matched
    report["excluded_linear_modules"] = excluded
    report["target_module_patterns"] = list(target_modules)
    report["lora_rank"] = rank
    report["lora_alpha"] = alpha
    report["lora_dropout"] = dropout
    return report


def unfreeze_mlp1(model: torch.nn.Module) -> Dict[str, Any]:
    """Fully unfreeze the multimodal projector (mlp1)."""
    if not hasattr(model, "mlp1"):
        raise RuntimeError("Model has no mlp1 projector module")
    n = 0
    n_params = 0
    for p in model.mlp1.parameters():
        p.requires_grad = True
        n += 1
        n_params += p.numel()
    # Ensure projector stays in train mode for dropout/norm updates if any.
    model.mlp1.train()
    return {"n_tensors": n, "n_parameters": int(n_params)}


def _is_embed_or_lm_head(name: str) -> bool:
    n = name.lower()
    return (
        "embed_tokens" in n
        or n.endswith("lm_head.weight")
        or ".lm_head." in n
        or n.endswith("lm_head.bias")
    )


def verify_lora_projector_trainability(model: torch.nn.Module) -> Dict[str, Any]:
    """Assert LoRA + mlp1 trainability contract for lora_projector experiments."""
    lora_trainable = []
    mlp1_trainable = []
    other_trainable = []
    vision_trainable = []
    embed_trainable = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name.lower():
            lora_trainable.append(name)
        elif name.startswith("mlp1.") or name == "mlp1":
            mlp1_trainable.append(name)
        elif name.startswith("vision_model"):
            vision_trainable.append(name)
        elif _is_embed_or_lm_head(name):
            embed_trainable.append(name)
        else:
            other_trainable.append(name)

    n_lora = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n.lower()
    )
    n_mlp1 = sum(
        p.numel()
        for n, p in model.named_parameters()
        if p.requires_grad and (n.startswith("mlp1.") or n == "mlp1")
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    # All mlp1 params must be trainable.
    mlp1_all = list(model.mlp1.parameters()) if hasattr(model, "mlp1") else []
    mlp1_fully = bool(mlp1_all) and all(p.requires_grad for p in mlp1_all)

    # Vision must be fully frozen.
    vision_frozen = not any(
        p.requires_grad for n, p in model.named_parameters() if n.startswith("vision_model")
    )

    # Embeds / lm_head frozen.
    embeds_frozen = not any(
        p.requires_grad for n, p in model.named_parameters() if _is_embed_or_lm_head(n)
    )

    # Qwen2 base (non-LoRA language_model) frozen except we already counted LoRA.
    qwen_base_trainable = [
        n
        for n, p in model.named_parameters()
        if p.requires_grad
        and ("language_model" in n or n.startswith("base_model"))
        and "lora_" not in n.lower()
        and not (n.startswith("mlp1.") or n == "mlp1")
    ]

    ok = (
        len(lora_trainable) > 0
        and mlp1_fully
        and vision_frozen
        and embeds_frozen
        and len(qwen_base_trainable) == 0
        and len(other_trainable) == 0
        and len(vision_trainable) == 0
        and len(embed_trainable) == 0
    )

    return {
        "ok": ok,
        "total_parameters": int(n_total),
        "trainable_parameters": int(n_train),
        "trainable_percent": float(100.0 * n_train / n_total) if n_total else 0.0,
        "llm_lora_parameters": int(n_lora),
        "mlp1_trainable_parameters": int(n_mlp1),
        "n_lora_param_tensors": len(lora_trainable),
        "n_mlp1_param_tensors": len(mlp1_trainable),
        "mlp1_fully_trainable": mlp1_fully,
        "vision_frozen": vision_frozen,
        "embed_tokens_and_lm_head_frozen": embeds_frozen,
        "qwen2_base_frozen_except_lora": len(qwen_base_trainable) == 0,
        "qwen_base_trainable_names": qwen_base_trainable[:50],
        "other_trainable_names": other_trainable[:50],
        "vision_trainable_names": vision_trainable[:20],
        "embed_trainable_names": embed_trainable[:20],
        "lora_param_names_preview": lora_trainable[:20],
        "mlp1_param_names": mlp1_trainable,
    }


def verify_lora_trainability(
    model: torch.nn.Module,
    matched_modules: Sequence[str],
) -> Dict[str, Any]:
    trainable = []
    frozen = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    lora_trainable = [n for n in trainable if "lora_" in n.lower()]
    non_lora_trainable = [n for n in trainable if "lora_" not in n.lower()]
    # Base params should be frozen (except possibly modules_to_save, none here).
    base_still_trainable = [
        n for n in non_lora_trainable if "lora_" not in n.lower()
    ]

    mlp1_trainable = any(
        n.startswith("mlp1") and p.requires_grad for n, p in model.named_parameters()
    )
    vision_trainable = any(
        n.startswith("vision_model") and p.requires_grad
        for n, p in model.named_parameters()
    )
    embed_or_head_trainable = any(
        p.requires_grad and _is_embed_or_lm_head(n) for n, p in model.named_parameters()
    )
    # LoRA-only contract: attn+MLP adapters only.
    ok = (
        len(lora_trainable) > 0
        and len(base_still_trainable) == 0
        and not mlp1_trainable
        and not vision_trainable
        and not embed_or_head_trainable
    )

    total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "ok": ok,
        "total_parameters": int(total),
        "trainable_parameters": int(n_train),
        "trainable_percent": float(100.0 * n_train / total) if total else 0.0,
        "n_lora_param_tensors": len(lora_trainable),
        "n_non_lora_trainable_tensors": len(base_still_trainable),
        "non_lora_trainable_names": base_still_trainable[:50],
        "lora_param_names_preview": lora_trainable[:50],
        "all_base_frozen_except_lora": len(base_still_trainable) == 0,
        "mlp1_frozen": not mlp1_trainable,
        "vision_frozen": not vision_trainable,
        "embed_tokens_and_lm_head_frozen": not embed_or_head_trainable,
        "mlp1_trainable": mlp1_trainable,
        "vision_trainable": vision_trainable,
        "n_matched_target_modules": len(matched_modules),
    }


def count_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": float(100.0 * trainable / total) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Training step / loop
# ---------------------------------------------------------------------------

def move_batch_to_device(batch: Dict[str, Any], device: torch.device, dtype: torch.dtype):
    out = {}
    for k, v in batch.items():
        if k == "meta":
            out[k] = v
            continue
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                out[k] = v.to(device=device, dtype=dtype)
            elif v.dtype in (torch.float16, torch.bfloat16, torch.float32):
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            # image_grid_hws may be numpy
            out[k] = v
    if not isinstance(out.get("image_grid_hws"), torch.Tensor):
        out["image_grid_hws"] = torch.as_tensor(out["image_grid_hws"], device=device)
    else:
        out["image_grid_hws"] = out["image_grid_hws"].to(device)
    return out


def forward_loss(model, batch: Dict[str, Any]) -> torch.Tensor:
    outputs = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        image_grid_hws=batch["image_grid_hws"],
        image_flags=batch["image_flags"],
        labels=batch["labels"],
        use_cache=False,
    )
    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
    return loss


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    use_8bit_adam: bool,
    projector_lr: Optional[float] = None,
):
    """Build AdamW, optionally with a separate LR group for mlp1."""
    if projector_lr is None:
        params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"params": params, "lr": lr}]
        group_report = {
            "n_groups": 1,
            "lora_or_default_lr": lr,
            "n_default_params": sum(p.numel() for p in params),
        }
    else:
        lora_params = []
        mlp1_params = []
        other = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("mlp1.") or name == "mlp1":
                mlp1_params.append(p)
            elif "lora_" in name.lower():
                lora_params.append(p)
            else:
                other.append(p)
        if other:
            raise RuntimeError(
                f"Unexpected trainable params outside LoRA/mlp1: "
                f"{[n for n,p in model.named_parameters() if p.requires_grad and 'lora_' not in n.lower() and not n.startswith('mlp1')][:20]}"
            )
        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr})
        if mlp1_params:
            param_groups.append({"params": mlp1_params, "lr": projector_lr})
        if not param_groups:
            raise RuntimeError("No trainable parameters for optimizer")
        group_report = {
            "n_groups": len(param_groups),
            "lora_lr": lr,
            "projector_lr": projector_lr,
            "n_lora_params": sum(p.numel() for p in lora_params),
            "n_mlp1_params": sum(p.numel() for p in mlp1_params),
        }

    if use_8bit_adam:
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(param_groups, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    opt._chestxray8_group_report = group_report  # type: ignore[attr-defined]
    return opt


def save_checkpoint(
    output_dir: Path,
    model,
    tokenizer,
    processor,
    args_dict: Dict[str, Any],
    step: int,
    is_lora: bool,
    extra: Optional[Dict[str, Any]] = None,
    save_projector: bool = False,
) -> Path:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if is_lora:
        # Save PEFT adapter from language_model
        lm = model.language_model
        if hasattr(lm, "save_pretrained"):
            # Avoid dumping frozen embed_tokens/lm_head (~1.3GB) when PEFT
            # thinks embeddings were resized.
            try:
                lm.save_pretrained(
                    str(ckpt_dir / "adapter"), save_embedding_layers=False
                )
            except TypeError:
                lm.save_pretrained(str(ckpt_dir / "adapter"))
        else:
            torch.save(lm.state_dict(), ckpt_dir / "adapter_state.pt")
        (ckpt_dir / "adapter_marker.json").write_text(
            json.dumps({"type": "lora_adapter", "step": step}, indent=2) + "\n"
        )
    else:
        model.save_pretrained(str(ckpt_dir))

    if save_projector and hasattr(model, "mlp1"):
        torch.save(model.mlp1.state_dict(), ckpt_dir / "mlp1.pt")
        (ckpt_dir / "projector_marker.json").write_text(
            json.dumps({"type": "mlp1_projector", "step": step}, indent=2) + "\n"
        )

    tokenizer.save_pretrained(str(ckpt_dir))
    try:
        processor.save_pretrained(str(ckpt_dir))
    except Exception:
        pass
    meta = {"step": step, "args": args_dict, "save_projector": save_projector}
    if extra:
        meta.update(extra)
    (ckpt_dir / "train_state.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Pointer to latest
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt_dir) + "\n")
    return ckpt_dir


@torch.no_grad()
def run_single_inference_check(
    model,
    processor,
    tokenizer,
    pair: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Greedy generate on one held-out example and parse boxes."""
    from sft_common import parse_normalized_boxes, BOX_RE

    model.eval()
    pair = refresh_pair_prompt_fields(pair, PROMPT_STRATEGY)
    # Prefer the same high-level path as LocateAnythingWorker for reliable parsing.
    try:
        from locateanything_worker import LocateAnythingWorker
        from eval_locateanything_bbox import run_inference

        # Build a lightweight worker shell around the already-loaded model.
        worker = LocateAnythingWorker.__new__(LocateAnythingWorker)
        worker.device = str(device)
        worker.dtype = dtype
        worker.use_batch_runtime = False
        worker.tokenizer = tokenizer
        worker.processor = processor
        # Ensure processor has tokenizer for generate()
        if getattr(processor, "tokenizer", None) is None:
            processor.tokenizer = tokenizer
        worker.model = model
        image = Image.open(pair["image_path"]).convert("RGB")
        answer, _elapsed = run_inference(
            worker,
            image,
            pair["prompt_phrase"],
            prompt_strategy=PROMPT_STRATEGY,
            final_query=pair["user_query"],
        )
        answer = str(answer)
    except Exception as e:  # noqa: BLE001
        answer = f"<generation_failed: {e}>"

    boxes = parse_normalized_boxes(answer)
    model.train()
    return {
        "image_index": pair["image_index"],
        "disease": pair["disease"],
        "user_query": pair["user_query"],
        "raw_answer": answer,
        "parsed_norm_boxes": boxes,
        "parser_ok": True,
        "n_parsed_boxes": len(boxes),
        "gt_norm_boxes": pair.get("gt_boxes_norm_1000"),
        "parser_matched_box_syntax": BOX_RE.search(answer) is not None,
    }



def one_training_step(
    model,
    batch: Dict[str, Any],
    optimizer,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    batch = move_batch_to_device(batch, device, dtype)
    optimizer.zero_grad(set_to_none=True)
    try:
        loss = forward_loss(model, batch)
        if not torch.isfinite(loss):
            return {
                "ok": False,
                "error": f"non-finite loss: {loss.item()}",
                "loss": float(loss.detach().cpu()) if loss.numel() == 1 else None,
                "peak_mem": gpu_mem_mb(),
            }
        loss.backward()

        # Gradient diagnostics (especially important for LoRA / projector).
        lora_grad_norms = []
        mlp1_grad_norms = []
        frozen_with_grad = []
        trainable_without_grad = []
        vision_with_grad = []
        embed_with_grad = []
        qwen_base_with_grad = []
        mlp1_with_grad_frozen_or_train = []
        for name, p in model.named_parameters():
            has_nonzero_grad = (
                p.grad is not None
                and torch.isfinite(p.grad.detach().float().norm())
                and float(p.grad.detach().float().norm().cpu()) > 0
            )
            if name.startswith("mlp1.") or name == "mlp1":
                if has_nonzero_grad:
                    mlp1_with_grad_frozen_or_train.append(name)
            if p.requires_grad:
                if p.grad is None:
                    trainable_without_grad.append(name)
                else:
                    gnorm = float(p.grad.detach().float().norm().cpu())
                    finite = bool(torch.isfinite(p.grad.detach().float().norm()))
                    if "lora_" in name.lower():
                        if finite:
                            lora_grad_norms.append((name, gnorm))
                    elif name.startswith("mlp1.") or name == "mlp1":
                        if finite:
                            mlp1_grad_norms.append((name, gnorm))
            else:
                if has_nonzero_grad:
                    frozen_with_grad.append(name)
                    if name.startswith("vision_model"):
                        vision_with_grad.append(name)
                    elif _is_embed_or_lm_head(name):
                        embed_with_grad.append(name)
                    elif "language_model" in name and "lora_" not in name.lower():
                        qwen_base_with_grad.append(name)

        nonzero_lora = [(n, g) for n, g in lora_grad_norms if g > 0]
        nonzero_mlp1 = [(n, g) for n, g in mlp1_grad_norms if g > 0]
        mlp1_is_trainable = any(
            n.startswith("mlp1") and p.requires_grad for n, p in model.named_parameters()
        )
        optimizer.step()
        peak = gpu_mem_mb()
        return {
            "ok": True,
            "loss": float(loss.detach().float().cpu()),
            "loss_finite": True,
            "peak_mem": peak,
            "meta": batch.get("meta"),
            "grad_checks": {
                "n_lora_tensors_with_grad": len(lora_grad_norms),
                "n_lora_tensors_with_nonzero_grad": len(nonzero_lora),
                "lora_grad_norm_sum": float(sum(g for _, g in lora_grad_norms)),
                "lora_nonzero_ok": len(nonzero_lora) > 0,
                "n_mlp1_tensors_with_grad": len(mlp1_grad_norms),
                "n_mlp1_tensors_with_nonzero_grad": len(nonzero_mlp1),
                "mlp1_grad_norm_sum": float(sum(g for _, g in mlp1_grad_norms)),
                "mlp1_is_trainable": mlp1_is_trainable,
                # Projector mode: mlp1 must get grads. LoRA-only: mlp1 must get none.
                "mlp1_nonzero_ok": len(nonzero_mlp1) > 0 if mlp1_is_trainable else None,
                "mlp1_zero_grad_ok": len(mlp1_with_grad_frozen_or_train) == 0,
                "mlp1_with_grad_names": mlp1_with_grad_frozen_or_train[:20],
                "frozen_params_with_grad": frozen_with_grad[:20],
                "frozen_base_no_grad_ok": len(frozen_with_grad) == 0,
                "vision_zero_grad_ok": len(vision_with_grad) == 0,
                "embed_lm_head_zero_grad_ok": len(embed_with_grad) == 0,
                "qwen_base_zero_grad_ok": len(qwen_base_with_grad) == 0,
                "trainable_without_grad_preview": trainable_without_grad[:20],
            },
        }
    except torch.cuda.OutOfMemoryError as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "ok": False,
            "error": "CUDA_OOM",
            "exception": str(e),
            "peak_mem": gpu_mem_mb(),
            "traceback": traceback.format_exc(),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": type(e).__name__,
            "exception": str(e),
            "peak_mem": gpu_mem_mb(),
            "traceback": traceback.format_exc(),
        }


@torch.no_grad()
def evaluate_overfit_generations(
    model,
    processor,
    tokenizer,
    pairs: Sequence[Dict[str, Any]],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Generate on overfit samples; count valid structured box outputs."""
    rows = []
    n_valid = 0
    for pair in pairs:
        info = run_single_inference_check(
            model, processor, tokenizer, pair, device, dtype
        )
        valid = bool(info.get("parser_matched_box_syntax")) and int(
            info.get("n_parsed_boxes") or 0
        ) > 0
        if valid:
            n_valid += 1
        rows.append(
            {
                "image_index": pair["image_index"],
                "disease": pair["disease"],
                "raw": info.get("raw_answer"),
                "n_parsed_boxes": info.get("n_parsed_boxes"),
                "valid": valid,
            }
        )
    return {
        "n": len(pairs),
        "n_valid": n_valid,
        "valid_rate": n_valid / len(pairs) if pairs else 0.0,
        "samples": rows,
    }


def overfit_train_loop(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    output_dir: Path,
    is_lora: bool,
    tokenizer,
    processor,
    overfit_pairs: List[Dict[str, Any]],
    save_projector: bool = False,
) -> Dict[str, Any]:
    """Short overfit: train on N samples; eval generation every K steps; stop early."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    history = []
    gen_history = []
    global_step = 0
    accum = max(1, args.gradient_accumulation_steps)
    max_grad_norm = float(getattr(args, "max_grad_norm", DEFAULT_MAX_GRAD_NORM))
    eval_every = max(1, int(args.overfit_eval_every))
    max_steps = args.max_steps if args.max_steps is not None else 80

    # Baseline generations before any updates
    before = evaluate_overfit_generations(
        model, processor, tokenizer, overfit_pairs, device, dtype
    )
    gen_history.append({"step": 0, **before})
    logger.info(
        "Overfit BEFORE: valid=%d/%d", before["n_valid"], before["n"]
    )
    for s in before["samples"]:
        logger.info("  before %s: %r", s["image_index"], (s["raw"] or "")[:160])

    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    success = False

    while global_step < max_steps and not success:
        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps or success:
                break
            batch = move_batch_to_device(batch, device, dtype)
            loss = forward_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at overfit step {global_step}: {loss}")
            (loss / accum).backward()
            if (batch_idx + 1) % accum == 0:
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                loss_f = float(loss.detach().float().cpu())
                history.append({"step": global_step, "loss": loss_f})
                logger.info("overfit step=%d loss=%.4f", global_step, loss_f)

                if global_step % eval_every == 0 or global_step >= max_steps:
                    after = evaluate_overfit_generations(
                        model, processor, tokenizer, overfit_pairs, device, dtype
                    )
                    gen_history.append({"step": global_step, **after})
                    logger.info(
                        "Overfit @step %d: valid=%d/%d",
                        global_step,
                        after["n_valid"],
                        after["n"],
                    )
                    for s in after["samples"]:
                        logger.info(
                            "  %s valid=%s: %r",
                            s["image_index"],
                            s["valid"],
                            (s["raw"] or "")[:160],
                        )
                    if after["n_valid"] == after["n"] and after["n"] > 0:
                        success = True
                        logger.info("Overfit SUCCESS at step %d", global_step)
                        break

    ckpt = save_checkpoint(
        output_dir,
        model,
        tokenizer,
        processor,
        vars(args),
        global_step,
        is_lora=is_lora,
        extra={"overfit": True, "success": success},
        save_projector=save_projector,
    )
    report = {
        "success": success,
        "steps": global_step,
        "history": history,
        "generation_history": gen_history,
        "final_checkpoint": str(ckpt),
        "elapsed_sec": time.time() - start_time,
        "before": before,
        "after": gen_history[-1] if gen_history else None,
        "checkpoint_selection": {
            "label": "final_checkpoint",
            "criterion": "overfit_run",
            "validation_evaluated": False,
        },
    }
    (output_dir / "overfit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    (output_dir / "loss_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output_dir / "final_checkpoint.txt").write_text(str(ckpt) + "\n")
    return report


def train_loop(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    args: argparse.Namespace,
    output_dir: Path,
    is_lora: bool,
    tokenizer,
    processor,
    val_pairs: List[Dict[str, Any]],
    save_projector: bool = False,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    history = []
    global_step = 0
    best_train_loss_at_save = float("inf")
    final_ckpt = None
    selected_ckpt = None
    selection_criterion = "none"
    accum = max(1, args.gradient_accumulation_steps)
    max_grad_norm = float(getattr(args, "max_grad_norm", DEFAULT_MAX_GRAD_NORM))
    model.train()
    optimizer.zero_grad(set_to_none=True)

    epochs = args.num_epochs
    max_steps = args.max_steps
    start_time = time.time()

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            if max_steps is not None and global_step >= max_steps:
                break
            batch = move_batch_to_device(batch, device, dtype)
            try:
                loss = forward_loss(model, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {global_step}: {loss}")
                (loss / accum).backward()
            except torch.cuda.OutOfMemoryError as e:
                report = {
                    "ok": False,
                    "error": "CUDA_OOM",
                    "exception": str(e),
                    "step": global_step,
                    "peak_mem": gpu_mem_mb(),
                    "config": vars(args),
                }
                (output_dir / "oom_report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                logger.error("CUDA OOM at step %d: %s", global_step, e)
                return {
                    "history": history,
                    "oom": report,
                    "final_checkpoint": str(final_ckpt) if final_ckpt else None,
                    "checkpoint_selection": {
                        "label": "interrupted_oom",
                        "criterion": selection_criterion,
                        "path": str(selected_ckpt) if selected_ckpt else None,
                    },
                }

            if (batch_idx + 1) % accum == 0:
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                loss_f = float(loss.detach().float().cpu())
                history.append({"step": global_step, "epoch": epoch, "loss": loss_f})
                if global_step % args.logging_steps == 0:
                    logger.info(
                        "step=%d epoch=%d loss=%.4f mem=%.1fMB",
                        global_step,
                        epoch,
                        loss_f,
                        gpu_mem_mb()["allocated_mb"],
                    )
                if global_step % args.save_steps == 0:
                    ckpt = save_checkpoint(
                        output_dir,
                        model,
                        tokenizer,
                        processor,
                        vars(args),
                        global_step,
                        is_lora=is_lora,
                        extra={"loss": loss_f, "selection_note": "save_step"},
                        save_projector=save_projector,
                    )
                    # Track lowest train loss among save points only — NOT validation.
                    if loss_f < best_train_loss_at_save:
                        best_train_loss_at_save = loss_f
                        selected_ckpt = ckpt
                        selection_criterion = (
                            "lowest_train_loss_at_save_steps (NO validation)"
                        )
                        (output_dir / "lowest_train_loss_checkpoint.txt").write_text(
                            str(ckpt) + "\n"
                        )

        if max_steps is not None and global_step >= max_steps:
            break

    # Final checkpoint
    final_ckpt = save_checkpoint(
        output_dir,
        model,
        tokenizer,
        processor,
        vars(args),
        global_step if global_step > 0 else 0,
        is_lora=is_lora,
        extra={"final": True},
        save_projector=save_projector,
    )
    (output_dir / "loss_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )
    (output_dir / "final_checkpoint.txt").write_text(str(final_ckpt) + "\n")

    # Do not label a train-loss save as "best" unless validation was used.
    if val_pairs:
        logger.warning(
            "Validation split present (%d pairs) but online val selection is not "
            "implemented; labeling final checkpoint only.",
            len(val_pairs),
        )
    checkpoint_selection = {
        "label": "final_checkpoint",
        "criterion": "end_of_training",
        "path": str(final_ckpt),
        "lowest_train_loss_at_save_steps": str(selected_ckpt)
        if selected_ckpt
        else None,
        "lowest_train_loss_value": best_train_loss_at_save
        if selected_ckpt
        else None,
        "validation_evaluated": False,
    }
    (output_dir / "checkpoint_selection.json").write_text(
        json.dumps(checkpoint_selection, indent=2) + "\n"
    )
    # Keep legacy filename but make content unambiguous.
    (output_dir / "best_checkpoint.txt").write_text(
        f"{final_ckpt}\n# NOTE: this is the FINAL checkpoint, not validation-selected best.\n"
    )

    return {
        "history": history,
        "final_checkpoint": str(final_ckpt),
        "checkpoint_selection": checkpoint_selection,
        "elapsed_sec": time.time() - start_time,
        "oom": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiment-type",
        choices=["full_sft", "lora", "lora_projector"],
        required=True,
        help="full_sft | lora (LLM LoRA only) | lora_projector (LLM LoRA + full mlp1 FT)",
    )
    p.add_argument(
        "--dataset-path",
        type=str,
        default="/auto/data2/ykorkmaz/nih-chest-xrays/data/versions/3",
    )
    p.add_argument("--model-path", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision (commit hash / tag) to pin.",
    )
    p.add_argument(
        "--split-dir",
        type=str,
        default=str(CHEST_DIR / "splits"),
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Match the successful lora_direct_v1 / lora_projector_v1 protocol (~450 steps).
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=None, help="Optional hard stop (debug).")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument(
        "--projector-learning-rate",
        type=float,
        default=None,
        help="LR for fully unfrozen mlp1 (lora_projector only). Default: 1e-5.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=DEFAULT_MAX_GRAD_NORM,
        help="Gradient clipping norm (official LocateAnything default=1.0).",
    )
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument(
        "--save-steps",
        type=int,
        default=25,
        help="Checkpoint every N optimizer steps (default 25; yields ~18 ckpts over 5 epochs).",
    )
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    p.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    p.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    p.add_argument("--resume-from-checkpoint", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training pairs for debugging.",
    )
    p.add_argument(
        "--overfit-samples",
        type=int,
        default=None,
        help="If set, train only on the first N train pairs and periodically "
        "evaluate generation on those same N samples until all produce valid "
        "<box>...</box> outputs (or max-steps is reached).",
    )
    p.add_argument(
        "--overfit-eval-every",
        type=int,
        default=5,
        help="Optimizer steps between overfit generation checks.",
    )
    p.add_argument(
        "--debug-one-step",
        action="store_true",
        help="Load model, run one forward/backward/optimizer step, save debug report, exit.",
    )
    p.add_argument(
        "--use-8bit-adam",
        action="store_true",
        default=True,
        help="Use bitsandbytes AdamW8bit (default True; important for full SFT on 16GB).",
    )
    p.add_argument("--no-8bit-adam", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--attn-implementation", type=str, default="sdpa")
    return p


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


def main() -> None:
    args = build_argparser().parse_args()
    if args.no_8bit_adam:
        args.use_8bit_adam = False
    if args.no_grad_checkpoint:
        args.grad_checkpoint = False

    if args.learning_rate is None:
        args.learning_rate = (
            DEFAULT_FULL_SFT_LR if args.experiment_type == "full_sft" else DEFAULT_LORA_LR
        )
    if args.experiment_type == "lora_projector" and args.projector_learning_rate is None:
        args.projector_learning_rate = DEFAULT_PROJECTOR_LR

    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    set_global_seed(args.seed)

    uses_lora = args.experiment_type in ("lora", "lora_projector")
    train_projector = args.experiment_type == "lora_projector"

    logger.info("Experiment type: %s", args.experiment_type)
    logger.info("Prompt strategy: %s", PROMPT_STRATEGY)
    logger.info(
        "LoRA LR: %s (full_sft default=%s, lora default=%s); projector LR: %s",
        args.learning_rate,
        DEFAULT_FULL_SFT_LR,
        DEFAULT_LORA_LR,
        args.projector_learning_rate if train_projector else "n/a",
    )
    if train_projector:
        logger.info(
            "lora_projector contract: LLM LoRA on attn+MLP (lr=%s) + full mlp1 FT "
            "(lr=%s); vision / embed_tokens / lm_head frozen; causal-attn SFT mask on; "
            "checkpoints save adapter/ + mlp1.pt",
            args.learning_rate,
            args.projector_learning_rate,
        )
    logger.info("Args: %s", json.dumps(vars(args), indent=2))

    split_dir = Path(args.split_dir)
    train_pairs = read_jsonl(split_dir / f"train_pairs_seed{args.seed}.jsonl")
    val_pairs = []
    val_path = split_dir / f"val_pairs_seed{args.seed}.jsonl"
    if val_path.exists():
        val_pairs = read_jsonl(val_path)
    test_pairs = read_jsonl(split_dir / f"test_pairs_seed{args.seed}.jsonl")

    if args.max_train_samples is not None:
        train_pairs = train_pairs[: args.max_train_samples]
        logger.info("Using max_train_samples=%d", len(train_pairs))
    if args.overfit_samples is not None:
        train_pairs = train_pairs[: args.overfit_samples]
        logger.info("OVERFIT mode: using %d training samples", len(train_pairs))
        if args.max_steps is None:
            args.max_steps = 80
            logger.info("OVERFIT default max_steps=%d", args.max_steps)
        # Tight loop: one sample per optimizer step for faster feedback.
        args.gradient_accumulation_steps = 1
        args.save_steps = max(args.save_steps, args.max_steps + 1)
        args.num_epochs = max(args.num_epochs, 50)

    # Reproducibility dump
    repro = collect_reproducibility_info(args.model_path, args.model_revision)
    repro["training_args"] = vars(args)
    repro["prompt_strategy"] = PROMPT_STRATEGY
    repro["user_query_example"] = (
        "Locate the Atelectasis in this chest X-ray"
        if PROMPT_STRATEGY == "direct_disease"
        else None
    )
    repro["target_format"] = "<ref>{phrase}</ref><box><x1><y1><x2><y2></box>..."
    if PROMPT_STRATEGY == "direct_disease":
        repro["target_format"] = (
            "<ref>{Disease}</ref><box><x1><y1><x2><y2></box>... "
            "(user query: Locate the {Disease} in this chest X-ray)"
        )
    repro["n_train"] = len(train_pairs)
    repro["n_val"] = len(val_pairs)
    repro["n_test"] = len(test_pairs)
    (output_dir / "reproducibility.json").write_text(json.dumps(repro, indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    mem_before = gpu_mem_mb()
    logger.info("GPU memory before model load: %s", mem_before)

    tokenizer, processor = load_tokenizer_and_processor(
        args.model_path,
        revision=args.model_revision,
        max_seq_length=args.max_seq_length,
    )
    model, resolved_revision = load_locateanything_model(
        args.model_path,
        tokenizer,
        revision=args.model_revision,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    )
    args.model_revision = resolved_revision
    logger.info("Pinned/resolved Hugging Face revision: %s", resolved_revision)
    model.to(device)
    mem_after_load = gpu_mem_mb()
    logger.info("GPU memory after model load: %s", mem_after_load)

    # Re-apply after .to(device) in case class was not yet fully resolved.
    pos_loss_patch = patch_qwen2_pos_loss_list_bug(model)
    logger.info("pos_loss_list patch (post-load): %s", pos_loss_patch)

    lora_report = None
    projector_report = None
    if args.experiment_type == "full_sft":
        for p in model.parameters():
            p.requires_grad = True
        param_report = count_parameters(model)
        logger.info("Full SFT parameter report: %s", param_report)
    else:
        # Inspect architecture before LoRA
        linear_names = discover_linear_module_names(model)
        (output_dir / "all_linear_modules.json").write_text(
            json.dumps(linear_names, indent=2) + "\n"
        )
        lora_report = apply_llm_lora(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=LLM_LORA_TARGET_MODULES,
        )
        # Patch again after PEFT wrap (idempotent on the Qwen2 class).
        pos_loss_patch = patch_qwen2_pos_loss_list_bug(model)
        logger.info("pos_loss_list patch (post-lora): %s", pos_loss_patch)

        if train_projector:
            unfreeze_info = unfreeze_mlp1(model)
            logger.info("Unfroze mlp1 projector: %s", unfreeze_info)
            projector_report = verify_lora_projector_trainability(model)
            (output_dir / "trainable_parameter_report.json").write_text(
                json.dumps(projector_report, indent=2) + "\n"
            )
            logger.info(
                "lora_projector params: llm_lora=%d mlp1=%d total_trainable=%d / %d (%.4f%%)",
                projector_report["llm_lora_parameters"],
                projector_report["mlp1_trainable_parameters"],
                projector_report["trainable_parameters"],
                projector_report["total_parameters"],
                projector_report["trainable_percent"],
            )
            logger.info(
                "Freeze checks: vision=%s embeds=%s qwen_base=%s mlp1_full=%s ok=%s",
                projector_report["vision_frozen"],
                projector_report["embed_tokens_and_lm_head_frozen"],
                projector_report["qwen2_base_frozen_except_lora"],
                projector_report["mlp1_fully_trainable"],
                projector_report["ok"],
            )
            if not projector_report["ok"]:
                raise RuntimeError(
                    f"lora_projector trainability contract failed: {projector_report}"
                )
            param_report = {
                "total_parameters": projector_report["total_parameters"],
                "trainable_parameters": projector_report["trainable_parameters"],
                "trainable_percent": projector_report["trainable_percent"],
                "llm_lora_parameters": projector_report["llm_lora_parameters"],
                "mlp1_trainable_parameters": projector_report["mlp1_trainable_parameters"],
            }
            lora_report = {**lora_report, **projector_report}
        else:
            param_report = {
                "total_parameters": lora_report["total_parameters"],
                "trainable_parameters": lora_report["trainable_parameters"],
                "trainable_percent": lora_report["trainable_percent"],
                "llm_lora_parameters": lora_report["trainable_parameters"],
                "mlp1_trainable_parameters": 0,
            }
            (output_dir / "trainable_parameter_report.json").write_text(
                json.dumps(lora_report, indent=2) + "\n"
            )
            logger.info(
                "LoRA (attn+MLP) trainable=%d / total=%d (%.4f%%); "
                "mlp1_frozen=%s vision_frozen=%s embeds_frozen=%s ok=%s",
                lora_report["trainable_parameters"],
                lora_report["total_parameters"],
                lora_report["trainable_percent"],
                lora_report.get("mlp1_frozen"),
                lora_report.get("vision_frozen"),
                lora_report.get("embed_tokens_and_lm_head_frozen"),
                lora_report.get("ok"),
            )
            if not lora_report.get("ok", False):
                raise RuntimeError(
                    "LoRA-only trainability contract failed "
                    "(expected attn+MLP adapters only; mlp1/vision/embeds frozen): "
                    f"{lora_report}"
                )

        (output_dir / "lora_target_modules.json").write_text(
            json.dumps(
                {
                    "matched": lora_report["matched_modules"],
                    "excluded_preview": lora_report["excluded_linear_modules"][:200],
                    "n_excluded": len(lora_report["excluded_linear_modules"]),
                    "patterns": LLM_LORA_TARGET_MODULES,
                    "train_projector": train_projector,
                },
                indent=2,
            )
            + "\n"
        )
        logger.info("LoRA matched modules (%d):", len(lora_report["matched_modules"]))
        for m in lora_report["matched_modules"][:20]:
            logger.info("  MATCH %s", m)
        if len(lora_report["matched_modules"]) > 20:
            logger.info("  ... (%d more)", len(lora_report["matched_modules"]) - 20)

    (output_dir / "parameter_summary.json").write_text(
        json.dumps(param_report, indent=2) + "\n"
    )
    (output_dir / "training_arguments.json").write_text(
        json.dumps(vars(args), indent=2) + "\n"
    )

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable({"use_reentrant": False})
        logger.info("Gradient checkpointing enabled")

    # Always use the training-safe forward (ignore_index CE + tuple unwrap +
    # LoRA-safe vision token injection). Required for both full_sft and lora
    # because HF Qwen2 returns (output, pos_loss_list) while training.
    patch_lora_embedding_inplace_fix(model)
    # AR SFT must use causal attention (not MTP block masks) so image tokens
    # receive gradients and mlp1 can train.
    causal_mask_patch = patch_qwen_force_causal_attn_for_sft(model)
    logger.info("causal SFT attention mask patch: %s", causal_mask_patch)

    # Ensure use_cache disabled
    model.language_model.config.use_cache = False

    dataset = ChestXray8SFTDataset(
        train_pairs, processor, tokenizer, max_seq_length=args.max_seq_length
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    optimizer = build_optimizer(
        model,
        args.learning_rate,
        args.weight_decay,
        use_8bit_adam=args.use_8bit_adam,
        projector_lr=args.projector_learning_rate if train_projector else None,
    )
    group_report = getattr(optimizer, "_chestxray8_group_report", None)
    if group_report:
        logger.info("Optimizer param groups: %s", group_report)
        (output_dir / "optimizer_param_groups.json").write_text(
            json.dumps(group_report, indent=2) + "\n"
        )

    # Debug one-step mode
    if args.debug_one_step:
        batch = next(iter(loader))
        result = one_training_step(model, batch, optimizer, device, dtype)
        debug_report = {
            "experiment_type": args.experiment_type,
            "model_revision": resolved_revision,
            "hf_home": os.environ.get("HF_HOME"),
            "pos_loss_list_patch": pos_loss_patch,
            "mem_before_load": mem_before,
            "mem_after_load": mem_after_load,
            "parameter_summary": param_report,
            "lora_report": lora_report,
            "one_step_result": result,
            "config": vars(args),
            "bug_notes": {
                "pos_loss_list_assigned_when": "labels is not None",
                "training_return_always_includes_pos_loss_list": True,
                "locateanything_calls_language_model_with_labels": False,
                "inference_uses_training_false_so_skips_bug": True,
                "safe_default": None,
            },
        }
        debug_path = output_dir / "debug_one_step_report.json"
        debug_path.write_text(json.dumps(debug_report, indent=2, default=str) + "\n")
        logger.info("One-step result: %s", result)
        if not result.get("ok"):
            logger.error("One-step training FAILED. See %s", debug_path)
            if result.get("error") == "CUDA_OOM" and args.experiment_type == "full_sft":
                logger.error(
                    "Full SFT appears infeasible on this GPU with the attempted "
                    "memory-saving settings. LoRA remains a separate experiment."
                )
            sys.exit(2)

        grads = result.get("grad_checks") or {}
        if uses_lora:
            if not grads.get("lora_nonzero_ok", False):
                logger.error("LoRA gradients are all zero / non-finite after backward")
                sys.exit(3)
            if not grads.get("vision_zero_grad_ok", True):
                logger.error("MoonViT received gradients despite freeze")
                sys.exit(3)
            if not grads.get("embed_lm_head_zero_grad_ok", True):
                logger.error("embed_tokens/lm_head received gradients despite freeze")
                sys.exit(3)
            if not grads.get("qwen_base_zero_grad_ok", True):
                logger.error(
                    "Frozen Qwen2 base params received gradients: %s",
                    grads.get("frozen_params_with_grad"),
                )
                sys.exit(3)
            if train_projector:
                if not grads.get("mlp1_nonzero_ok", False):
                    logger.error("mlp1 gradients are all zero after backward")
                    sys.exit(3)
                logger.info(
                    "Grad checks OK (lora_projector): nonzero_lora=%s nonzero_mlp1=%s "
                    "vision_clean=%s embeds_clean=%s qwen_base_clean=%s peak_mem=%s",
                    grads.get("n_lora_tensors_with_nonzero_grad"),
                    grads.get("n_mlp1_tensors_with_nonzero_grad"),
                    grads.get("vision_zero_grad_ok"),
                    grads.get("embed_lm_head_zero_grad_ok"),
                    grads.get("qwen_base_zero_grad_ok"),
                    result.get("peak_mem"),
                )
            else:
                # LoRA-only: mlp1 must stay frozen and receive no gradients.
                if grads.get("mlp1_is_trainable"):
                    logger.error("mlp1 is trainable in LoRA-only mode; refusing to continue")
                    sys.exit(3)
                if not grads.get("mlp1_zero_grad_ok", True):
                    logger.error(
                        "mlp1 received gradients in LoRA-only mode: %s",
                        grads.get("mlp1_with_grad_names"),
                    )
                    sys.exit(3)
                if not grads.get("frozen_base_no_grad_ok", True):
                    logger.error(
                        "Frozen base params received gradients: %s",
                        grads.get("frozen_params_with_grad"),
                    )
                    sys.exit(3)
                logger.info(
                    "Grad checks OK (lora-only): nonzero_lora=%s mlp1_zero_grad=%s "
                    "vision_clean=%s embeds_clean=%s qwen_base_clean=%s peak_mem=%s",
                    grads.get("n_lora_tensors_with_nonzero_grad"),
                    grads.get("mlp1_zero_grad_ok"),
                    grads.get("vision_zero_grad_ok"),
                    grads.get("embed_lm_head_zero_grad_ok"),
                    grads.get("qwen_base_zero_grad_ok"),
                    result.get("peak_mem"),
                )

        # Save a tiny checkpoint and verify adapter reload + inference
        ckpt = save_checkpoint(
            output_dir,
            model,
            tokenizer,
            processor,
            vars(args),
            step=1,
            is_lora=uses_lora,
            extra={"debug_one_step": True, "loss": result.get("loss")},
            save_projector=train_projector,
        )
        reload_ok = False
        reload_error = None
        if uses_lora and (ckpt / "adapter").is_dir():
            try:
                from peft import PeftModel

                # Prefer verifying files exist + config loadable (avoid double-wrapping live model).
                adapter_cfg = ckpt / "adapter" / "adapter_config.json"
                reload_ok = adapter_cfg.is_file()
                if not reload_ok:
                    reload_error = "adapter_config.json missing"
                else:
                    if train_projector and not (ckpt / "mlp1.pt").is_file():
                        reload_ok = False
                        reload_error = "mlp1.pt missing"
                    else:
                        reload_ok = True
                        logger.info("Adapter checkpoint reloadable at %s", ckpt / "adapter")
                        if train_projector:
                            logger.info("Projector weights saved at %s", ckpt / "mlp1.pt")
            except Exception as e:  # noqa: BLE001
                reload_error = str(e)
                reload_ok = (ckpt / "adapter" / "adapter_config.json").is_file()
        elif uses_lora:
            reload_error = "adapter directory missing"
        else:
            reload_ok = True

        infer_pair = test_pairs[0] if test_pairs else train_pairs[0]
        infer = run_single_inference_check(
            model, processor, tokenizer, infer_pair, device, dtype
        )
        debug_report["checkpoint"] = str(ckpt)
        debug_report["adapter_reload"] = {
            "ok": reload_ok,
            "path": str(ckpt / "adapter") if uses_lora else None,
            "projector_path": str(ckpt / "mlp1.pt") if train_projector else None,
            "error": reload_error,
        }
        debug_report["inference_check"] = infer
        debug_report["parameter_summary"] = param_report
        debug_path.write_text(json.dumps(debug_report, indent=2, default=str) + "\n")
        logger.info(
            "Inference check: n_parsed_boxes=%s answer_preview=%r",
            infer.get("n_parsed_boxes"),
            (infer.get("raw_answer") or "")[:200],
        )
        logger.info("Debug one-step complete. Report: %s", debug_path)
        return

    # Scheduler
    steps_per_epoch = math.ceil(len(loader) / max(1, args.gradient_accumulation_steps))
    total_steps = steps_per_epoch * args.num_epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(1, total_steps)
    )

    if args.resume_from_checkpoint:
        logger.info("Resume requested from %s (adapter/weights load)", args.resume_from_checkpoint)
        ckpt = Path(args.resume_from_checkpoint)
        if uses_lora and (ckpt / "adapter").exists():
            from peft import PeftModel

            model.language_model = PeftModel.from_pretrained(
                model.language_model, str(ckpt / "adapter")
            )
            if train_projector and (ckpt / "mlp1.pt").is_file():
                model.mlp1.load_state_dict(
                    torch.load(ckpt / "mlp1.pt", map_location=device)
                )
                unfreeze_mlp1(model)
        elif (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists():
            model = type(model).from_pretrained(str(ckpt), torch_dtype=dtype).to(device)

    if args.overfit_samples is not None:
        # Minimal scheduler for short overfit
        total_steps = args.max_steps or 80
        warmup_steps = max(1, int(total_steps * args.warmup_ratio))
        from transformers import get_cosine_schedule_with_warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, total_steps),
        )
        result = overfit_train_loop(
            model,
            loader,
            optimizer,
            scheduler,
            args,
            output_dir,
            is_lora=uses_lora,
            tokenizer=tokenizer,
            processor=processor,
            overfit_pairs=train_pairs,
            save_projector=train_projector,
        )
        (output_dir / "train_result.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        logger.info(
            "Overfit finished success=%s steps=%s checkpoint=%s",
            result.get("success"),
            result.get("steps"),
            result.get("final_checkpoint"),
        )
        return

    result = train_loop(
        model,
        loader,
        optimizer,
        scheduler,
        args,
        output_dir,
        is_lora=uses_lora,
        tokenizer=tokenizer,
        processor=processor,
        val_pairs=val_pairs,
        save_projector=train_projector,
    )
    (output_dir / "train_result.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    if result.get("oom"):
        logger.error("Training stopped due to OOM. See oom_report.json")
        sys.exit(3)
    sel = result.get("checkpoint_selection") or {}
    logger.info(
        "Training finished. Final checkpoint: %s (selection=%s)",
        result.get("final_checkpoint"),
        sel.get("label"),
    )


if __name__ == "__main__":
    main()
