#!/usr/bin/env python3
"""One-sample, one-backward real-model integration test for PBD GRPO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.pbd_rl import (  # noqa: E402
    PBDSamplingConfig,
    PBDRolloutReplayer,
    StochasticPBDRLDecoder,
)
from rl.policy_state import (  # noqa: E402
    PolicySnapshot,
    assert_approved_trainable_parameters,
)
from sft_common import LLM_LORA_TARGET_MODULES, read_jsonl  # noqa: E402
from train_chestxray8_sft import (  # noqa: E402
    PINNED_MODEL_REVISION,
    apply_llm_lora,
    load_locateanything_model,
    load_tokenizer_and_processor,
    unfreeze_mlp1,
)

MODEL = "nvidia/LocateAnything-3B"
G = 4


def _prompt_inputs(processor, pair: dict, device: torch.device) -> dict:
    image = Image.open(pair["image_path"]).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": pair["user_query"]},
            ],
        }
    ]
    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = processor.process_vision_info(messages)
    inputs = processor(
        text=[text], images=images, videos=videos, return_tensors="pt"
    )
    return {
        "input_ids": inputs["input_ids"].to(device),
        "pixel_values": inputs["pixel_values"].to(device=device, dtype=torch.bfloat16),
        "image_grid_hws": torch.as_tensor(inputs["image_grid_hws"], device=device),
    }


def _localization_rewards(traces, gt_coords: list[int]) -> torch.Tensor:
    """Continuous reward used only to ensure this smoke test has GRPO signal."""
    rewards = []
    for trace in traces:
        coords = [
            token - 151677 for token in trace.blocks[0].action_token_ids[1:5]
        ]
        rewards.append(
            -sum(abs(pred - target) for pred, target in zip(coords, gt_coords))
            / 4000.0
        )
    return torch.tensor(rewards, dtype=torch.float32)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("real-model PBD-RL integration requires CUDA")
    device = torch.device("cuda")
    tokenizer, processor = load_tokenizer_and_processor(
        MODEL, revision=PINNED_MODEL_REVISION, max_seq_length=2048
    )
    model, revision = load_locateanything_model(
        MODEL,
        tokenizer,
        revision=PINNED_MODEL_REVISION,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    apply_llm_lora(
        model,
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules=LLM_LORA_TARGET_MODULES,
    )
    unfreeze_mlp1(model)
    trainability = assert_approved_trainable_parameters(model)
    initial_old = PolicySnapshot.capture(model, optimizer_step=0)

    pair = read_jsonl(CHEST_DIR / "splits/train80_pairs_seed42.jsonl")[0]
    inputs = _prompt_inputs(processor, pair, device)
    sampling = PBDSamplingConfig(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    decoder = StochasticPBDRLDecoder(model, tokenizer, sampling)
    traces = [
        decoder.generate(
            **inputs,
            max_new_tokens=6,
            seed=1701 + index,
            force_first_box_block=True,
        )
        for index in range(G)
    ]
    for trace in traces:
        emitted = trace.generated_token_ids
        actions = [token for block in trace.blocks for token in block.action_token_ids]
        assert emitted == actions
        assert len(trace.blocks) == 1
        assert trace.blocks[0].block_type == "box"
        assert len(actions) == 6

    replayer = PBDRolloutReplayer(model, tokenizer)
    with torch.no_grad():
        old_logps = replayer.score_one_block_group(traces, **inputs)
    assert torch.allclose(
        old_logps,
        torch.tensor([trace.old_log_prob for trace in traces], device=device),
        atol=2e-4,
        rtol=0,
    )
    # The reference is the raw base checkpoint: fresh LoRA is disabled and the
    # projector is still at its checkpoint initialization.
    with torch.no_grad(), model.language_model.disable_adapter():
        reference_logps = replayer.score_one_block_group(traces, **inputs)

    initial_old.load_into(model)
    current_logps = replayer.score_one_block_group(traces, **inputs)
    ratios = torch.exp(current_logps.detach() - old_logps)
    if not torch.allclose(ratios, torch.ones_like(ratios), atol=2e-4, rtol=0):
        raise AssertionError(f"initial PPO ratios are not one: {ratios.tolist()}")
    if not torch.allclose(current_logps.detach(), reference_logps, atol=2e-4, rtol=0):
        raise AssertionError("fresh policy and raw reference probabilities differ")

    rewards = _localization_rewards(traces, pair["gt_boxes_norm_1000"][0])
    advantages = group_relative_advantages(rewards).to(device)
    loss = grpo_clipped_loss(
        current_logps, old_logps, advantages, clip_epsilon=0.2
    )
    loss.backward()

    grad_names = []
    frozen_with_grad = []
    lora_grad_sq = 0.0
    projector_grad_sq = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not parameter.requires_grad:
            frozen_with_grad.append(name)
            continue
        grad_names.append(name)
        value = float(parameter.grad.detach().float().pow(2).sum().item())
        if "lora_" in name.lower():
            lora_grad_sq += value
        elif name.startswith("mlp1."):
            projector_grad_sq += value
    assert not frozen_with_grad
    assert lora_grad_sq > 0.0
    assert projector_grad_sq > 0.0

    report = {
        "model": MODEL,
        "revision": revision,
        "sample": {
            "image_index": pair["image_index"],
            "disease": pair["disease"],
            "gt_box_norm_1000": pair["gt_boxes_norm_1000"][0],
        },
        "G": G,
        "sampling": sampling.__dict__,
        "actions": [trace.generated_token_ids for trace in traces],
        "old_log_probs": old_logps.cpu().tolist(),
        "current_log_probs": current_logps.detach().cpu().tolist(),
        "reference_log_probs": reference_logps.cpu().tolist(),
        "ratios": ratios.cpu().tolist(),
        "rewards": rewards.tolist(),
        "advantages": advantages.cpu().tolist(),
        "loss_total_equals_loss_grpo": float(loss.detach().cpu()),
        "supervised_loss": None,
        "trainable_tensor_count": len(trainability["trainable_names"]),
        "gradient_tensor_count": len(grad_names),
        "lora_grad_norm": lora_grad_sq**0.5,
        "projector_grad_norm": projector_grad_sq**0.5,
        "frozen_parameters_with_grad": frozen_with_grad,
        "max_cuda_memory_mb": torch.cuda.max_memory_allocated() / (1024**2),
    }
    print(json.dumps(report, indent=2))
    print("REAL MODEL PBD-RL INTEGRATION PASSED")


if __name__ == "__main__":
    main()
