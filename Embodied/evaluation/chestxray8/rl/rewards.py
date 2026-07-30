"""Production Chain-of-Box rewards for reward-only ChestX-ray8 GRPO."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_locateanything_bbox import box_iou
from rl.medclip_loader import load_pinned_medclip

NATIVE_BOX_PATTERN = re.compile(
    r"<box><(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})></box>"
)
OUTER_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>",
    re.DOTALL,
)
# Generation may append chat/eos markers after a valid </answer>. Strip only
# trailing special tokens for reward parsing; do not rewrite box content.
_TRAILING_SPECIAL = re.compile(
    r"(?:\s|<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>)+$"
)


def normalize_completion_for_reward(text: str) -> str:
    return _TRAILING_SPECIAL.sub("", text or "").strip()


@dataclass(frozen=True)
class ParsedCompletion:
    format_valid: bool
    final_box_norm_1000: Optional[Tuple[int, int, int, int]]
    think_text: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class RewardComponents:
    format_reward: float
    spatial_reward: float
    semantic_reward: float
    total_reward: float
    final_iou: float
    format_valid: bool
    geometry_valid: bool
    final_box_norm_1000: Optional[Tuple[int, int, int, int]]
    parse_error: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticScorer(Protocol):
    def score(
        self,
        image_path: str | Path,
        box_norm_1000: Sequence[int],
        query: str,
    ) -> float: ...


def _native_boxes(text: str) -> list[Tuple[int, int, int, int]]:
    boxes = []
    for match in NATIVE_BOX_PATTERN.finditer(text):
        box = tuple(int(value) for value in match.groups())
        if all(0 <= value <= 1000 for value in box):
            boxes.append(box)
    return boxes


def parse_chain_of_box_completion(text: str) -> ParsedCompletion:
    """Extract only the final answer box; never use intermediate boxes."""
    if not isinstance(text, str):
        return ParsedCompletion(False, None, None, "completion is not text")
    text = normalize_completion_for_reward(text)
    match = OUTER_PATTERN.fullmatch(text)
    if match is None:
        return ParsedCompletion(False, None, None, "invalid outer structure")
    think = match.group("think")
    answer = match.group("answer")
    answer_match = NATIVE_BOX_PATTERN.fullmatch(answer.strip())
    if answer_match is None:
        return ParsedCompletion(False, None, think, "answer is not exactly one box")
    final_box = tuple(int(value) for value in answer_match.groups())
    if not all(0 <= value <= 1000 for value in final_box):
        return ParsedCompletion(False, None, think, "coordinate outside [0, 1000]")
    valid_think_boxes = _native_boxes(think)
    if think.count("<box>") != len(valid_think_boxes):
        return ParsedCompletion(False, None, think, "malformed intermediate box")
    if think.count("</box>") != len(valid_think_boxes):
        return ParsedCompletion(False, None, think, "malformed intermediate box")
    return ParsedCompletion(True, final_box, think, None)


def is_valid_geometry(box: Sequence[int]) -> bool:
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def format_reward(completion: str) -> float:
    return float(parse_chain_of_box_completion(completion).format_valid)


def spatial_reward(
    completion: str,
    gt_boxes_norm_1000: Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    parsed = parse_chain_of_box_completion(completion)
    if (
        not parsed.format_valid
        or parsed.final_box_norm_1000 is None
        or not is_valid_geometry(parsed.final_box_norm_1000)
    ):
        return 0.0, 0.0
    best_iou = max(
        (
            box_iou(parsed.final_box_norm_1000, gt_box)
            for gt_box in gt_boxes_norm_1000
        ),
        default=0.0,
    )
    return float(best_iou > threshold), float(best_iou)


class MedCLIPSemanticScorer:
    """Frozen MedCLIP cosine scorer over final ROI and original query."""

    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self.model, self.identity = load_pinned_medclip()
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _roi_tensor(
        image_path: str | Path, box_norm_1000: Sequence[int]
    ) -> torch.Tensor:
        image = Image.open(image_path).convert("L")
        width, height = image.size
        x1, y1, x2, y2 = box_norm_1000
        crop = image.crop(
            (
                round(x1 / 1000.0 * width),
                round(y1 / 1000.0 * height),
                round(x2 / 1000.0 * width),
                round(y2 / 1000.0 * height),
            )
        )
        side = max(224, crop.width, crop.height)
        padded = Image.new("L", (side, side), 0)
        padded.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        resized = padded.resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - 0.5862785803043838) / 0.27950088968644304
        return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)

    def score(
        self,
        image_path: str | Path,
        box_norm_1000: Sequence[int],
        query: str,
    ) -> float:
        if not is_valid_geometry(box_norm_1000):
            return 0.0
        pixels = self._roi_tensor(image_path, box_norm_1000).to(self.device)
        tokens = self.model.text_model.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        with torch.no_grad():
            image_embedding = self.model.vision_model(pixel_values=pixels)
            text_embedding = self.model.text_model(
                tokens["input_ids"].to(self.device),
                tokens["attention_mask"].to(self.device),
            )
            cosine = F.cosine_similarity(image_embedding, text_embedding).item()
        return float(cosine)


class ProductionRewardPipeline:
    def __init__(
        self,
        semantic_scorer: SemanticScorer,
        *,
        format_weight: float = 1.0,
        spatial_weight: float = 1.0,
        semantic_weight: float = 1.0,
        iou_threshold: float = 0.5,
    ) -> None:
        self.semantic_scorer = semantic_scorer
        self.format_weight = float(format_weight)
        self.spatial_weight = float(spatial_weight)
        self.semantic_weight = float(semantic_weight)
        self.iou_threshold = float(iou_threshold)

    def score(self, completion: str, pair: Dict[str, Any]) -> RewardComponents:
        parsed = parse_chain_of_box_completion(completion)
        fmt = float(parsed.format_valid)
        spatial, iou = spatial_reward(
            completion,
            pair["gt_boxes_norm_1000"],
            threshold=self.iou_threshold,
        )
        geometry_valid = bool(
            parsed.final_box_norm_1000 is not None
            and is_valid_geometry(parsed.final_box_norm_1000)
        )
        semantic = 0.0
        if parsed.format_valid and geometry_valid:
            semantic = self.semantic_scorer.score(
                pair["image_path"],
                parsed.final_box_norm_1000,
                pair["user_query"],
            )
        total = (
            self.format_weight * fmt
            + self.spatial_weight * spatial
            + self.semantic_weight * semantic
        )
        return RewardComponents(
            format_reward=fmt,
            spatial_reward=spatial,
            semantic_reward=float(semantic),
            total_reward=float(total),
            final_iou=iou,
            format_valid=parsed.format_valid,
            geometry_valid=geometry_valid,
            final_box_norm_1000=parsed.final_box_norm_1000,
            parse_error=parsed.error,
        )
