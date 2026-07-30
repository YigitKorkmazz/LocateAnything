"""Shared runtime helpers for ChestX-ray8 PBD-RL entrypoints."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import yaml

from rl.pbd_rl import PBDSamplingConfig, RolloutTrace, StochasticPBDRLDecoder
from rl.prompt import build_rl_messages
from sft_common import LLM_LORA_TARGET_MODULES, read_jsonl
from train_chestxray8_sft import (
    apply_llm_lora,
    load_locateanything_model,
    load_tokenizer_and_processor,
    unfreeze_mlp1,
)

CHEST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "chestxray8_grpo_pbd.yaml"


def load_resolved_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["objective"]["loss_total"] != "L_GRPO":
        raise RuntimeError("resolved config must use L_total=L_GRPO")
    if config["objective"].get("supervised_losses") != []:
        raise RuntimeError("resolved config must not contain supervised losses")
    if int(config["objective"]["group_size"]) != 4:
        raise RuntimeError("this experiment requires G=4")
    config["_config_path"] = str(config_path)
    return config


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else CHEST_DIR / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_pairs(config: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    key = f"{split}_split"
    hash_key = f"{split}_sha256"
    path = resolve_data_path(config["data"][key])
    expected = config["data"][hash_key]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{split} split hash mismatch: {actual}")
    return read_jsonl(path)


def fixed_sample_indices(population_size: int, count: int, seed: int) -> List[int]:
    if count != 100:
        raise ValueError("viability requires exactly 100 samples")
    if population_size < count:
        raise ValueError("training split has fewer than 100 samples")
    return sorted(random.Random(int(seed)).sample(range(population_size), count))


def assert_new_output_dir(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sampling_from_config(config: Dict[str, Any]) -> PBDSamplingConfig:
    rollout = config["rollout"]
    return PBDSamplingConfig(
        temperature=float(rollout["temperature"]),
        top_k=int(rollout["top_k"]),
        top_p=float(rollout["top_p"]),
        repetition_penalty=float(rollout["repetition_penalty"]),
        block_size=int(rollout["block_size"]),
    )


def build_policy(config: Dict[str, Any], device: torch.device):
    model_cfg = config["model"]
    tokenizer, processor = load_tokenizer_and_processor(
        model_cfg["name_or_path"],
        revision=model_cfg["revision"],
        max_seq_length=int(config["rollout"]["max_sequence_length"]),
    )
    model, revision = load_locateanything_model(
        model_cfg["name_or_path"],
        tokenizer,
        revision=model_cfg["revision"],
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    lora = model_cfg["lora"]
    apply_llm_lora(
        model,
        rank=int(lora["r"]),
        alpha=int(lora["alpha"]),
        dropout=float(lora["dropout"]),
        target_modules=LLM_LORA_TARGET_MODULES,
    )
    unfreeze_mlp1(model)
    return model, tokenizer, processor, revision


def tokenize_rl_pair(
    processor,
    pair: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    messages = build_rl_messages(pair)
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
    return {
        "input_ids": inputs["input_ids"].to(device),
        "attention_mask": inputs["attention_mask"].to(device),
        "pixel_values": inputs["pixel_values"].to(
            device=device, dtype=torch.bfloat16
        ),
        "image_grid_hws": torch.as_tensor(
            inputs["image_grid_hws"], device=device
        ),
        "rendered_prompt": text,
    }


def decoder_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_ids": inputs["input_ids"],
        "pixel_values": inputs["pixel_values"],
        "image_grid_hws": inputs["image_grid_hws"],
    }


def generate_rollout_group(
    model,
    tokenizer,
    inputs: Dict[str, Any],
    config: Dict[str, Any],
    *,
    sample_seed: int,
) -> List[RolloutTrace]:
    group_size = int(config["objective"]["group_size"])
    decoder = StochasticPBDRLDecoder(
        model, tokenizer, sampling=sampling_from_config(config)
    )
    return [
        decoder.generate(
            **decoder_inputs(inputs),
            max_new_tokens=int(config["rollout"]["max_new_tokens"]),
            seed=int(sample_seed) * group_size + group_index,
            force_first_box_block=False,
        )
        for group_index in range(group_size)
    ]


def trace_record(
    trace: RolloutTrace,
    *,
    sample_index: int,
    group_index: int,
    pair: Dict[str, Any],
    reward: Dict[str, Any],
    advantage: float,
) -> Dict[str, Any]:
    return {
        "sample_index": sample_index,
        "group_index": group_index,
        "image_index": pair["image_index"],
        "patient_id": pair["patient_id"],
        "disease": pair["disease"],
        "query": pair["user_query"],
        "completion": trace.decoded_text,
        "reward": reward,
        "advantage": float(advantage),
        "rollout_trace": trace.to_dict(),
    }


def summarize_rewards(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n_rollouts": 0}
    rewards = np_array([record["reward"]["total_reward"] for record in records])
    return {
        "n_rollouts": len(records),
        "mean_total_reward": float(rewards.mean()),
        "median_total_reward": float(np_median(rewards)),
        "format_valid_rate": float(
            sum(record["reward"]["format_valid"] for record in records)
            / len(records)
        ),
        "spatial_success_rate": float(
            sum(record["reward"]["spatial_reward"] for record in records)
            / len(records)
        ),
        "mean_semantic_reward": float(
            sum(record["reward"]["semantic_reward"] for record in records)
            / len(records)
        ),
    }


def np_array(values):
    import numpy as np

    return np.asarray(values, dtype=np.float64)


def np_median(values):
    import numpy as np

    return np.median(values)


def projector_state_cpu(model) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.mlp1.state_dict().items()
    }


def save_rl_checkpoint(
    output_dir: Path,
    model,
    optimizer,
    *,
    step: int,
    config: Dict[str, Any],
    reference_projector: Dict[str, torch.Tensor],
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint}")
    checkpoint.mkdir(parents=True)
    model.language_model.save_pretrained(
        checkpoint / "adapter", save_embedding_layers=False
    )
    torch.save(model.mlp1.state_dict(), checkpoint / "mlp1.pt")
    torch.save(reference_projector, checkpoint / "reference_mlp1.pt")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    write_json(
        checkpoint / "trainer_state.json",
        {
            "optimizer_step": step,
            "loss_total": "L_GRPO",
            "supervised_losses": [],
            "config": config,
        },
    )
    return checkpoint
