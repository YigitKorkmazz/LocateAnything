"""Model / tokenizer / LoRA / optimizer construction for production_grpo_fast.

Loading goes through the existing, validated `train_chestxray8_sft.py`
helpers (not a from-scratch `AutoModel.from_pretrained` call) so this backend
gets the same tokenizer special-token setup, vocab-resize guard, and Qwen2
`pos_loss_list` patch that the rest of the repository's RL/SFT trainers rely
on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch

CHEST_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (str(CHEST_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from sft_common import LLM_LORA_TARGET_MODULES  # noqa: E402
from train_chestxray8_sft import (  # noqa: E402
    apply_llm_lora,
    build_optimizer as _sft_build_optimizer,
    load_locateanything_model,
    load_tokenizer_and_processor,
)

MODEL_NAME = "nvidia/LocateAnything-3B"
PINNED_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
EXPECTED_LORA_TENSORS = 504
EXPECTED_LORA_PARAMETERS = 14_966_784


class TrainabilityError(RuntimeError):
    pass


def build_model_and_tokenizer(
    device: torch.device,
    *,
    revision: str = PINNED_REVISION,
    max_sequence_length: int = 4096,
    attn_implementation: str = "sdpa",
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Load base model + tokenizer/processor, apply LoRA, freeze everything else.

    The projector (`mlp1`) and vision encoder are left frozen deliberately:
    this backend's baseline is LoRA-only (`model.projector_trainable=false`
    in spec terms). Do not call `unfreeze_mlp1` here.
    """
    tokenizer, processor = load_tokenizer_and_processor(
        MODEL_NAME, revision=revision, max_seq_length=max_sequence_length
    )
    model, resolved_revision = load_locateanything_model(
        MODEL_NAME,
        tokenizer,
        revision=revision,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    lora_report = apply_llm_lora(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        dropout=LORA_DROPOUT,
        target_modules=LLM_LORA_TARGET_MODULES,
    )
    model.eval()
    model.vision_model.eval()
    model.mlp1.eval()
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    if hasattr(model.language_model, "gradient_checkpointing_disable"):
        model.language_model.gradient_checkpointing_disable()
    model.language_model.config.use_cache = True
    return model, tokenizer, processor, {"revision": resolved_revision, **lora_report}


def trainability_audit(model: Any) -> Dict[str, Any]:
    """Strict contract check: exactly 504 LoRA tensors / 14,966,784 params."""
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lora = [(n, p) for n, p in trainable if "lora_" in n.lower()]
    non_lora = [(n, p) for n, p in trainable if "lora_" not in n.lower()]
    vision_trainable = [n for n, p in model.named_parameters() if p.requires_grad and n.startswith("vision_model")]
    projector_trainable = [n for n, p in model.named_parameters() if p.requires_grad and (n == "mlp1" or n.startswith("mlp1."))]
    result = {
        "trainable_tensors": len(trainable),
        "trainable_parameters": int(sum(p.numel() for _, p in trainable)),
        "lora_tensors": len(lora),
        "lora_parameters": int(sum(p.numel() for _, p in lora)),
        "non_lora_trainable_tensors": len(non_lora),
        "non_lora_trainable_names": [n for n, _ in non_lora][:20],
        "projector_trainable_tensors": len(projector_trainable),
        "vision_trainable_tensors": len(vision_trainable),
    }
    ok = (
        result["trainable_tensors"] == EXPECTED_LORA_TENSORS
        and result["lora_tensors"] == EXPECTED_LORA_TENSORS
        and result["trainable_parameters"] == EXPECTED_LORA_PARAMETERS
        and result["non_lora_trainable_tensors"] == 0
        and result["projector_trainable_tensors"] == 0
        and result["vision_trainable_tensors"] == 0
    )
    result["ok"] = ok
    if not ok:
        raise TrainabilityError(f"trainability audit failed: {result}")
    return result


def build_optimizer(
    model: Any,
    *,
    lr: float = 1e-6,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """AdamW over LoRA params only (projector frozen -> no second LR group)."""
    return _sft_build_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
        use_8bit_adam=False,
        projector_lr=None,
    )


def build_linear_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        remaining = total_steps - step
        return max(0.0, float(remaining) / float(max(1, total_steps - warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def gradient_audit(model: Any) -> Dict[str, int]:
    lora = [(n, p) for n, p in model.named_parameters() if p.requires_grad and "lora_" in n.lower()]
    finite = sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in lora)
    nonzero = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for _, p in lora)
    projector_grad = sum(
        p.grad is not None for n, p in model.named_parameters() if n == "mlp1" or n.startswith("mlp1.")
    )
    vision_grad = sum(p.grad is not None for n, p in model.named_parameters() if n.startswith("vision_model"))
    frozen_non_lora_grad = sum(
        p.grad is not None for n, p in model.named_parameters() if not p.requires_grad and "lora_" not in n.lower()
    )
    return {
        "lora_tensors_with_finite_gradients": finite,
        "lora_tensors_with_nonzero_gradients": nonzero,
        "n_lora_tensors": len(lora),
        "projector_tensors_with_gradients": projector_grad,
        "vision_tensors_with_gradients": vision_grad,
        "frozen_non_lora_tensors_with_gradients": frozen_non_lora_grad,
    }


def assert_gradient_audit_ok(audit: Dict[str, int]) -> None:
    if audit["lora_tensors_with_finite_gradients"] != EXPECTED_LORA_TENSORS:
        raise RuntimeError(f"only {audit['lora_tensors_with_finite_gradients']}/504 LoRA tensors have finite gradients")
    if audit["projector_tensors_with_gradients"] != 0:
        raise RuntimeError("projector received gradients")
    if audit["vision_tensors_with_gradients"] != 0:
        raise RuntimeError("vision encoder received gradients")
    if audit["frozen_non_lora_tensors_with_gradients"] != 0:
        raise RuntimeError("a frozen non-LoRA parameter received gradients")
    if audit["lora_tensors_with_nonzero_gradients"] == 0:
        raise RuntimeError("all LoRA gradients are identically zero")
