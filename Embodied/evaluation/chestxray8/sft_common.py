#!/usr/bin/env python3
"""Shared constants and helpers for ChestX-ray8 LocateAnything SFT."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Reuse eval prompt / disease constants.
from eval_locateanything_bbox import (  # noqa: E402
    CHESTXRAY8_DISEASES,
    LABEL_ALIASES,
    RADIOLOGY_CONTEXT_PHRASE_TEMPLATE,
    BOX_RE,
    build_final_user_query,
    build_image_index,
    canonicalize_disease,
    load_bbox_annotations,
    parse_normalized_boxes,
    convert_boxes_to_pixels,
    box_iou,
    greedy_match,
    compute_pair_metrics,
)

DEFAULT_MODEL_NAME = "nvidia/LocateAnything-3B"
DEFAULT_DATASET_PATH = "/auto/data2/ykorkmaz/nih-chest-xrays/data/versions/3"
DEFAULT_BBOX_CSV = "BBox_List_2017.csv"
DEFAULT_SEED = 42
# Default for train + held-out eval (user query: "Locate the {Disease} in this chest X-ray").
PROMPT_STRATEGY = "direct_disease"

# Used by radiology_context / ground_multi path; direct_disease uses DIRECT_DISEASE_QUERY_TEMPLATE.
USER_QUERY_TEMPLATE = (
    "Locate all the instances that match the following description: {phrase}."
)

# Defaults explained in train_chestxray8_sft.py --help / README block.
DEFAULT_FULL_SFT_LR = 2e-5
# Match official LocateAnything LoRA scripts (shell/locate-anything-lora-*.sh default LR=2e-5).
# 1e-4 caused catastrophic collapse around step 44 in the first ChestX-ray8 LoRA run.
DEFAULT_LORA_LR = 2e-5
DEFAULT_PROJECTOR_LR = 1e-5
DEFAULT_LORA_RANK = 8
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05

# Official LocateAnything LLM LoRA targets (Qwen2 Attention + MLP).
LLM_LORA_TARGET_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

IGNORE_INDEX = -100


def patient_id_from_image_index(image_index: str) -> str:
    """NIH ChestX-ray filenames are ``{PatientID}_{FollowUp}.png``."""
    stem = Path(image_index).stem
    return stem.split("_", 1)[0]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def xyxy_pixels_to_norm1000(
    boxes_xyxy: Sequence[Sequence[float]],
    width: int,
    height: int,
) -> List[List[int]]:
    """Convert pixel xyxy boxes to LocateAnything [0, 1000] integer tokens."""
    out: List[List[int]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        nx1 = int(round(max(0.0, min(float(x1) / width * 1000.0, 1000.0))))
        ny1 = int(round(max(0.0, min(float(y1) / height * 1000.0, 1000.0))))
        nx2 = int(round(max(0.0, min(float(x2) / width * 1000.0, 1000.0))))
        ny2 = int(round(max(0.0, min(float(y2) / height * 1000.0, 1000.0))))
        if nx2 <= nx1:
            nx2 = min(1000, nx1 + 1)
        if ny2 <= ny1:
            ny2 = min(1000, ny1 + 1)
        out.append([nx1, ny1, nx2, ny2])
    return out


def format_box_tokens(box_norm: Sequence[int]) -> str:
    x1, y1, x2, y2 = box_norm
    return f"<box><{x1}><{y1}><{x2}><{y2}></box>"


def build_assistant_target(phrase: str, boxes_norm_1000: Sequence[Sequence[int]]) -> str:
    """Native LocateAnything multi-box phrase-grounding target."""
    if not boxes_norm_1000:
        return f"<ref>{phrase}</ref><box>none</box>"
    parts = [f"<ref>{phrase}</ref>"]
    for box in boxes_norm_1000:
        parts.append(format_box_tokens(box))
    return "".join(parts)


def refresh_pair_prompt_fields(
    pair: Dict[str, Any],
    strategy: str = PROMPT_STRATEGY,
) -> Dict[str, Any]:
    """Rebuild prompt/target fields from disease + strategy (ignore stale jsonl text)."""
    disease = pair["disease"]
    phrase, user_query = build_final_user_query(disease, strategy)
    boxes_norm = pair["gt_boxes_norm_1000"]
    out = dict(pair)
    out["prompt_strategy"] = strategy
    out["prompt_phrase"] = phrase
    out["user_query"] = user_query
    out["assistant_target"] = build_assistant_target(phrase, boxes_norm)
    return out


def build_sharegpt_sample(
    image_path: str,
    phrase: str,
    user_query: str,
    assistant_target: str,
) -> Dict[str, Any]:
    return {
        "conversations": [
            {
                "from": "human",
                "value": f"<image-1>{user_query}",
            },
            {
                "from": "gpt",
                "value": assistant_target,
            },
        ],
        "image": image_path,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def class_distribution(pairs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {d: 0 for d in CHESTXRAY8_DISEASES}
    for p in pairs:
        disease = p["disease"]
        counts[disease] = counts.get(disease, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def collect_reproducibility_info(
    model_path: str,
    model_revision: Optional[str] = None,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model_path": model_path,
        "model_revision": model_revision,
        "cwd": os.getcwd(),
    }
    try:
        import subprocess

        repo_root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_commit"] = commit
    except Exception as e:  # noqa: BLE001
        info["git_commit"] = f"unavailable: {e}"

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)

    packages = {}
    for name in (
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "numpy",
        "PIL",
        "openpyxl",
    ):
        try:
            mod = __import__(name if name != "PIL" else "PIL")
            packages[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            packages[name] = "not_installed"
    info["packages"] = packages
    return info


def gpu_mem_mb() -> Dict[str, float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }
    except Exception:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}


def process_image_sft_sample(sample: Dict[str, Any], media_root: str = "") -> List[Dict[str, Any]]:
    """Lightweight image-only ShareGPT -> LocateAnything messages converter.

    Avoids importing ``eaglevl.train.tools`` (pulls optional video deps like ``av``).
    Equivalent to the image path of ``process_multimodal_sample`` for our JSONL.
    """
    import os.path as osp

    conversations = sample.get("conversations", [])
    raw_images = sample.get("image") or sample.get("image_list")
    image_data: List[Any] = []
    if raw_images:
        image_list = raw_images if isinstance(raw_images, list) else [raw_images]
        for img in image_list:
            if isinstance(img, str):
                image_data.append(osp.join(media_root, img) if media_root else img)
            else:
                image_data.append(img)

    # Ensure numbered placeholders exist.
    all_texts = "".join(conv["value"] for conv in conversations)
    placeholders_to_add = ""
    for i in range(len(image_data)):
        if f"<image-{i + 1}>" not in all_texts:
            placeholders_to_add += f"<image-{i + 1}>"
    if placeholders_to_add and conversations:
        conversations = [dict(conversations[0])] + list(conversations[1:])
        conversations[0]["value"] = placeholders_to_add + conversations[0]["value"]

    placeholder_pattern = re.compile(r"<(image|video)-(\d+)>")
    new_messages: List[Dict[str, Any]] = []
    for conv in conversations:
        role = "user" if conv["from"] == "human" else "assistant"
        value = conv["value"]
        if role == "assistant":
            new_messages.append({"role": role, "content": value})
            continue
        content_list: List[Dict[str, Any]] = [{"type": "text", "text": value}]
        for media_type, num_str in placeholder_pattern.findall(value):
            if media_type != "image":
                continue
            index = int(num_str) - 1
            if 0 <= index < len(image_data):
                content_list.append({"type": "image", "image": image_data[index]})
        new_messages.append({"role": role, "content": content_list})
    return new_messages

