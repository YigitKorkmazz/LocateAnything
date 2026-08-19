"""Shared runtime helpers for ChestX-ray8 PBD-RL entrypoints."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
import yaml

from rl.hybrid_rl import (
    DEFAULT_LOGPROB_OBJECTIVE,
    LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED,
    LOGPROB_OBJECTIVE_FULL_TRAJECTORY,
    VALID_LOGPROB_OBJECTIVES,
    HybridRolloutReplayer,
    StochasticHybridRLDecoder,
    resolve_logprob_objective,
)
from rl.pbd_rl import (
    PBDRolloutReplayer,
    PBDSamplingConfig,
    RolloutTrace,
    StochasticPBDRLDecoder,
)
from rl.ntp_rl import NTPRolloutReplayer, StochasticNTPRLDecoder
from rl.prompt import VALID_PROMPT_MODES, build_rl_messages, resolve_prompt_mode
from rl.rewards import VALID_PARSERS, resolve_parser_name
from sft_common import LLM_LORA_TARGET_MODULES, read_jsonl
from train_chestxray8_sft import (
    apply_llm_lora,
    load_locateanything_model,
    load_tokenizer_and_processor,
    unfreeze_mlp1,
)

CHEST_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "chestxray8_grpo_pbd.yaml"
DEFAULT_NATIVE_CONFIG = (
    Path(__file__).resolve().parent / "chestxray8_grpo_pbd_native.yaml"
)
DEFAULT_HYBRID_NATIVE_CONFIG = (
    Path(__file__).resolve().parent / "chestxray8_grpo_hybrid_native.yaml"
)
DEFAULT_HYBRID_MEDGROUND_KL_CONFIG = (
    Path(__file__).resolve().parent
    / "chestxray8_grpo_hybrid_native_medground_kl.yaml"
)

VALID_LOSS_TOTALS = ("L_GRPO", "L_GRPO_PLUS_MEDGROUND_KL")

ROLLOUT_PATH_PBD = "stochastic_native_pbd_rl"
ROLLOUT_PATH_HYBRID = "stochastic_native_hybrid_rl"
ROLLOUT_PATH_NTP_ONLY = "stochastic_native_ntp_rl"
# Legacy CoB YAML may omit path or use an older alias.
VALID_ROLLOUT_PATHS = (
    ROLLOUT_PATH_PBD,
    ROLLOUT_PATH_HYBRID,
    ROLLOUT_PATH_NTP_ONLY,
    "stochastic_pbd_rl",
)


def load_resolved_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["objective"]["loss_total"] not in VALID_LOSS_TOTALS:
        raise RuntimeError(
            "resolved config must use L_GRPO or L_GRPO_PLUS_MEDGROUND_KL"
        )
    if config["objective"].get("supervised_losses") != []:
        raise RuntimeError("resolved config must not contain supervised losses")
    if int(config["objective"]["group_size"]) != 4:
        raise RuntimeError("this experiment requires G=4")
    # Default missing mode keys for the legacy Chain-of-Box YAML.
    prompt_cfg = config.setdefault("prompt", {})
    prompt_cfg.setdefault("mode", "chain_of_box")
    reward_cfg = config.setdefault("rewards", {})
    reward_cfg.setdefault("parser", "chain_of_box")
    mode = resolve_prompt_mode(config)
    parser = resolve_parser_name(config)
    if mode not in VALID_PROMPT_MODES:
        raise RuntimeError(f"unsupported prompt.mode={mode!r}")
    if parser not in VALID_PARSERS:
        raise RuntimeError(f"unsupported rewards.parser={parser!r}")
    if mode == "native_locateanything" and parser != "native_locateanything":
        raise RuntimeError(
            "native_locateanything prompt mode requires rewards.parser=native_locateanything"
        )
    if mode == "chain_of_box" and parser != "chain_of_box":
        raise RuntimeError(
            "chain_of_box prompt mode requires rewards.parser=chain_of_box"
        )
    if mode == "native_locateanything" and bool(config["rollout"].get("chain_of_box")):
        raise RuntimeError("native mode requires rollout.chain_of_box=false")
    rollout_path = config["rollout"].get("path", ROLLOUT_PATH_PBD)
    if rollout_path not in VALID_ROLLOUT_PATHS and mode == "native_locateanything":
        raise RuntimeError(f"unsupported rollout.path={rollout_path!r}")
    if rollout_path == ROLLOUT_PATH_HYBRID:
        hybrid_cfg = config["rollout"].get("hybrid") or {}
        objective = resolve_logprob_objective(
            hybrid_cfg.get("logprob_objective", DEFAULT_LOGPROB_OBJECTIVE)
        )
        score_rejected = bool(hybrid_cfg.get("score_rejected_pbd_proposals", False))
        if objective == LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED and score_rejected:
            raise RuntimeError(
                "conditional_committed_output is a surrogate that excludes "
                "rejected PBD proposal tokens; set score_rejected_pbd_proposals=false "
                "or switch hybrid.logprob_objective to full_trajectory"
            )
        if objective == LOGPROB_OBJECTIVE_FULL_TRAJECTORY and not score_rejected:
            raise RuntimeError(
                "full_trajectory must include the sampled PBD proposal tokens "
                "that determine the fallback gate; set "
                "score_rejected_pbd_proposals=true"
            )
        hybrid_cfg["logprob_objective"] = objective
        config["rollout"]["hybrid"] = hybrid_cfg
        if bool(config["rollout"].get("reconstruct_actions_from_text")):
            raise RuntimeError(
                "Hybrid GRPO forbids reconstruct_actions_from_text"
            )
    if rollout_path == ROLLOUT_PATH_NTP_ONLY:
        ntp_cfg = config["rollout"].get("ntp_only") or {}
        expected = {
            "enabled": True,
            "generation_mode": "slow",
            "pbd_enabled": False,
            "mtp_enabled": False,
            "hybrid_fallback_enabled": False,
            "rejected_proposal_trajectory_enabled": False,
        }
        if ntp_cfg != expected:
            raise RuntimeError(
                "NTP-only rollout contract mismatch: "
                + json.dumps({"expected": expected, "actual": ntp_cfg}, sort_keys=True)
            )
        hybrid_cfg = config["rollout"].get("hybrid") or {}
        if bool(hybrid_cfg.get("enabled", False)):
            raise RuntimeError("NTP-only rollout requires Hybrid decoding disabled")
        if bool(hybrid_cfg.get("score_rejected_pbd_proposals", False)):
            raise RuntimeError("NTP-only rollout forbids rejected PBD proposal scoring")
        if bool(config["rollout"].get("reconstruct_actions_from_text")):
            raise RuntimeError("NTP-only GRPO replays decoder-native token actions")
    config["_config_path"] = str(config_path)
    return config


def resolve_rollout_path(config: Dict[str, Any]) -> str:
    path = config.get("rollout", {}).get("path", ROLLOUT_PATH_PBD)
    if path == "stochastic_pbd_rl":
        return ROLLOUT_PATH_PBD
    return str(path)


def is_hybrid_rollout(config: Dict[str, Any]) -> bool:
    return resolve_rollout_path(config) == ROLLOUT_PATH_HYBRID


def is_ntp_only_rollout(config: Dict[str, Any]) -> bool:
    return resolve_rollout_path(config) == ROLLOUT_PATH_NTP_ONLY


def hybrid_logprob_objective(config: Dict[str, Any]) -> str:
    hybrid_cfg = config.get("rollout", {}).get("hybrid") or {}
    return resolve_logprob_objective(
        hybrid_cfg.get("logprob_objective", DEFAULT_LOGPROB_OBJECTIVE)
    )


def build_rollout_replayer(model, tokenizer, config: Dict[str, Any]):
    if is_ntp_only_rollout(config):
        return NTPRolloutReplayer(model, tokenizer)
    if is_hybrid_rollout(config):
        return HybridRolloutReplayer(
            model,
            tokenizer,
            logprob_objective=hybrid_logprob_objective(config),
        )
    return PBDRolloutReplayer(model, tokenizer)


def build_rollout_decoder(
    model,
    tokenizer,
    config: Dict[str, Any],
    *,
    diagnostic_observer: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    sampling = sampling_from_config(config)
    if is_ntp_only_rollout(config):
        return StochasticNTPRLDecoder(
            model,
            tokenizer,
            sampling=sampling,
            diagnostic_observer=diagnostic_observer,
        )
    if is_hybrid_rollout(config):
        return StochasticHybridRLDecoder(
            model,
            tokenizer,
            sampling=sampling,
            logprob_objective=hybrid_logprob_objective(config),
            diagnostic_observer=diagnostic_observer,
        )
    return StochasticPBDRLDecoder(model, tokenizer, sampling=sampling)


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
    # GRPO historically used Case-B (LoRA + projector).  Memory probes and
    # LoRA-only experiments keep the identical base/prompt/rollout path while
    # leaving mlp1 frozen, as requested by model.projector_trainable.
    if bool(model_cfg.get("projector_trainable", True)):
        unfreeze_mlp1(model)
    return model, tokenizer, processor, revision


def build_policy_two_gpu_live_cache(
    config: Dict[str, Any],
    *,
    first_device: str | torch.device = "cuda:0",
    second_device: str | torch.device = "cuda:1",
    on_pre_forward_layout=None,
):
    """Build Case-B LoRA policy, then install the exact 18/18 decoder shard.

    The initial full ``cuda:0`` placement is intentional: PEFT injects LoRA
    there first, then the sharder moves every LoRA module recursively with its
    owning decoder layer.  Do not call ``model.to(...)`` after this function.
    """
    from rl.two_gpu_shard import shard_locateanything_decoder_two_gpu

    primary = torch.device(first_device)
    model, tokenizer, processor, revision = build_policy(config, primary)
    shard_report = shard_locateanything_decoder_two_gpu(
        model,
        first_device=first_device,
        second_device=second_device,
        first_layer_count=18,
        on_pre_forward_layout=on_pre_forward_layout,
    )
    return model, tokenizer, processor, revision, shard_report


def tokenize_rl_pair(
    processor,
    pair: Dict[str, Any],
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    messages = build_rl_messages(pair, config=config)
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
    diagnostic_observer: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[RolloutTrace]:
    group_size = int(config["objective"]["group_size"])
    decoder = build_rollout_decoder(
        model,
        tokenizer,
        config,
        diagnostic_observer=diagnostic_observer,
    )
    traces = []
    for group_index in range(group_size):
        # The sharded decoder consumes this only for pre-RoPE failure reports.
        # It is metadata, not model input, cache, or RNG state.
        try:
            from rl.two_gpu_shard import resolve_locateanything_qwen_decoder
            root = resolve_locateanything_qwen_decoder(model).decoder
            root._chestxray8_rotary_context = {**dict(getattr(model, "_chestxray8_rotary_context_base", {}) or {}),
                "trajectory_index": group_index, "rollout_seed": int(sample_seed) * group_size + group_index}
        except Exception:
            pass
        traces.append(decoder.generate(
            **decoder_inputs(inputs),
            max_new_tokens=int(config["rollout"]["max_new_tokens"]),
            seed=int(sample_seed) * group_size + group_index,
            force_first_box_block=False,
        ))
    return traces


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
