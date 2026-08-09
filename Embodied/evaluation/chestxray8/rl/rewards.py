"""Production rewards for reward-only ChestX-ray8 GRPO.

Format validity and bbox reward extraction are separate:

- ``R_format`` evaluates whether the final emitted completion satisfies the
  required grammar (Chain-of-Box or native LocateAnything).
- ``R_spatial`` / ``R_semantic`` consume the decoder's single committed final
  bbox (from ``RolloutTrace``), never an arbitrary BOX_RE match mined from a
  malformed completion and never a rejected / intermediate proposal.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eval_locateanything_bbox import BOX_RE, box_iou
from rl.medclip_loader import load_pinned_medclip

NATIVE_BOX_PATTERN = re.compile(
    r"<box><(\d{1,4})><(\d{1,4})><(\d{1,4})><(\d{1,4})></box>"
)
OUTER_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>\s*"
    r"<answer>(?P<answer>.*?)</answer>",
    re.DOTALL,
)
# Generation may append chat/eos markers after a valid completion. Strip only
# trailing special tokens for reward parsing; do not rewrite box content.
_TRAILING_SPECIAL = re.compile(
    r"(?:\s|<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>)+$"
)

PARSER_CHAIN_OF_BOX = "chain_of_box"
PARSER_NATIVE = "native_locateanything"
VALID_PARSERS = (PARSER_CHAIN_OF_BOX, PARSER_NATIVE)

Box = Tuple[int, int, int, int]


def normalize_completion_for_reward(text: str) -> str:
    return _TRAILING_SPECIAL.sub("", text or "").strip()


@dataclass(frozen=True)
class ParsedCompletion:
    format_valid: bool
    final_box_norm_1000: Optional[Box]
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
    final_box_norm_1000: Optional[Box]
    parse_error: Optional[str]
    reward_branch: Optional[str] = None
    has_unambiguous_committed_box: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SemanticScorer(Protocol):
    def score(
        self,
        image_path: str | Path,
        box_norm_1000: Sequence[int],
        query: str,
    ) -> float: ...


def _native_boxes(text: str) -> list[Box]:
    boxes = []
    for match in NATIVE_BOX_PATTERN.finditer(text):
        box = tuple(int(value) for value in match.groups())
        if all(0 <= value <= 1000 for value in box):
            boxes.append(box)  # type: ignore[arg-type]
    return boxes


def parse_chain_of_box_completion(text: str) -> ParsedCompletion:
    """Format + answer-box parse for Chain-of-Box completions."""
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
    return ParsedCompletion(True, final_box, think, None)  # type: ignore[arg-type]


def parse_native_locateanything_completion(text: str) -> ParsedCompletion:
    """Evaluate native LocateAnything *format* grammar on emitted text.

    This does **not** supply the spatial/semantic box for native PBD/Hybrid
    GRPO. Those rewards consume the decoder-committed final bbox from the
    rollout trace. The optional ``final_box_norm_1000`` here is retained only
    for format diagnostics and Chain-of-Box-style legacy callers.
    """
    if not isinstance(text, str):
        return ParsedCompletion(False, None, None, "completion is not text")
    text = normalize_completion_for_reward(text)
    matches = list(BOX_RE.finditer(text))
    if text.count("<box>") != len(matches) or text.count("</box>") != len(matches):
        return ParsedCompletion(False, None, None, "malformed native box")
    if not matches:
        return ParsedCompletion(False, None, None, "no native box")
    boxes: list[Box] = []
    for match in matches:
        box = tuple(int(value) for value in match.groups())
        if not all(0 <= value <= 1000 for value in box):
            return ParsedCompletion(
                False, None, None, "coordinate outside [0, 1000]"
            )
        boxes.append(box)  # type: ignore[arg-type]
    if len(boxes) != 1:
        return ParsedCompletion(
            False,
            None,
            None,
            "expected exactly one native box for single-object task",
        )
    final_box = boxes[0]
    if not is_valid_geometry(final_box):
        return ParsedCompletion(False, None, None, "invalid geometry")
    return ParsedCompletion(True, final_box, None, None)


COMPLETION_PARSERS: Dict[str, Callable[[str], ParsedCompletion]] = {
    PARSER_CHAIN_OF_BOX: parse_chain_of_box_completion,
    PARSER_NATIVE: parse_native_locateanything_completion,
}


def resolve_parser_name(config: Optional[Dict[str, Any]] = None) -> str:
    if config is None:
        return PARSER_CHAIN_OF_BOX
    name = config.get("rewards", {}).get("parser", PARSER_CHAIN_OF_BOX)
    if name not in COMPLETION_PARSERS:
        raise ValueError(f"unsupported rewards.parser={name!r}")
    return name


def is_valid_geometry(box: Sequence[int]) -> bool:
    x1, y1, x2, y2 = box
    return x2 > x1 and y2 > y1


def format_reward(
    completion: str,
    *,
    parser_name: str = PARSER_CHAIN_OF_BOX,
) -> float:
    parser = COMPLETION_PARSERS[parser_name]
    return float(parser(completion).format_valid)


def spatial_reward_from_box(
    box: Optional[Sequence[int]],
    gt_boxes_norm_1000: Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """MedGround-R1 spatial reward on one committed final box."""
    if box is None or not is_valid_geometry(box):
        return 0.0, 0.0
    best_iou = max(
        (box_iou(box, gt_box) for gt_box in gt_boxes_norm_1000),
        default=0.0,
    )
    return float(best_iou > threshold), float(best_iou)


def spatial_reward(
    completion: str,
    gt_boxes_norm_1000: Sequence[Sequence[float]],
    *,
    threshold: float = 0.5,
    parser_name: str = PARSER_CHAIN_OF_BOX,
    committed_final_box: Optional[Sequence[int]] = None,
) -> Tuple[float, float]:
    """Spatial reward; prefers an explicit decoder-committed box when given."""
    if committed_final_box is not None:
        return spatial_reward_from_box(
            committed_final_box,
            gt_boxes_norm_1000,
            threshold=threshold,
        )
    parsed = COMPLETION_PARSERS[parser_name](completion)
    if (
        not parsed.format_valid
        or parsed.final_box_norm_1000 is None
        or not is_valid_geometry(parsed.final_box_norm_1000)
    ):
        return 0.0, 0.0
    return spatial_reward_from_box(
        parsed.final_box_norm_1000,
        gt_boxes_norm_1000,
        threshold=threshold,
    )


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
        parser_name: str = PARSER_CHAIN_OF_BOX,
        invalid_box_semantic_fallback: float = 0.0,
    ) -> None:
        if parser_name not in COMPLETION_PARSERS:
            raise ValueError(f"unsupported parser_name={parser_name!r}")
        self.semantic_scorer = semantic_scorer
        self.format_weight = float(format_weight)
        self.spatial_weight = float(spatial_weight)
        self.semantic_weight = float(semantic_weight)
        self.iou_threshold = float(iou_threshold)
        self.parser_name = parser_name
        self.parse = COMPLETION_PARSERS[parser_name]
        self.invalid_box_semantic_fallback = float(invalid_box_semantic_fallback)

    def score(
        self,
        completion: str,
        pair: Dict[str, Any],
        *,
        committed_final_box: Optional[Sequence[int]] = None,
        has_unambiguous_committed_box: Optional[bool] = None,
        reward_branch: Optional[str] = None,
    ) -> RewardComponents:
        """Score format from text and spatial/semantic from the committed box.

        Native PBD/Hybrid callers should pass the decoder-committed box from
        ``RolloutTrace``. When ``has_unambiguous_committed_box`` is False,
        spatial=0, semantic=fallback, and format=0.

        Legacy Chain-of-Box callers may omit committed-box fields; the answer
        box from the CoB parser is then used as the committed final box.
        """
        parsed = self.parse(completion)

        if has_unambiguous_committed_box is None:
            # Legacy / CoB path: parser answer box is the committed action.
            if parsed.format_valid and parsed.final_box_norm_1000 is not None:
                has_unambiguous_committed_box = True
                if committed_final_box is None:
                    committed_final_box = parsed.final_box_norm_1000
            else:
                has_unambiguous_committed_box = False
                committed_final_box = None

        if not has_unambiguous_committed_box or committed_final_box is None:
            return RewardComponents(
                format_reward=0.0,
                spatial_reward=0.0,
                semantic_reward=float(self.invalid_box_semantic_fallback),
                total_reward=float(
                    self.semantic_weight * self.invalid_box_semantic_fallback
                ),
                final_iou=0.0,
                format_valid=False,
                geometry_valid=False,
                final_box_norm_1000=None,
                parse_error=parsed.error or "no unambiguous committed final box",
                reward_branch=reward_branch or "none",
                has_unambiguous_committed_box=False,
            )

        box: Box = tuple(int(v) for v in committed_final_box)  # type: ignore[assignment]
        fmt = float(parsed.format_valid)
        spatial, iou = spatial_reward_from_box(
            box,
            pair["gt_boxes_norm_1000"],
            threshold=self.iou_threshold,
        )
        geometry_valid = bool(is_valid_geometry(box))
        semantic = self.invalid_box_semantic_fallback
        if geometry_valid:
            semantic = self.semantic_scorer.score(
                pair["image_path"],
                box,
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
            final_box_norm_1000=box,
            parse_error=parsed.error,
            reward_branch=reward_branch,
            has_unambiguous_committed_box=True,
        )

    def score_from_trace(
        self,
        trace: Any,
        pair: Dict[str, Any],
    ) -> RewardComponents:
        """Score using the decoder-committed final bbox on ``RolloutTrace``."""
        return self.score(
            trace.decoded_text or "",
            pair,
            committed_final_box=getattr(
                trace, "committed_final_box_norm_1000", None
            ),
            has_unambiguous_committed_box=bool(
                getattr(trace, "has_unambiguous_committed_box", False)
            ),
            reward_branch=getattr(trace, "reward_branch", None),
        )


def build_reward_pipeline_from_config(
    config: Dict[str, Any],
    semantic_scorer: SemanticScorer,
) -> ProductionRewardPipeline:
    reward_cfg = config["rewards"]
    parser_name = resolve_parser_name(config)
    invalid_fallback = float(
        reward_cfg.get("semantic", {}).get("invalid_box_fallback", 0.0)
    )
    return ProductionRewardPipeline(
        semantic_scorer,
        format_weight=float(reward_cfg["format"]["weight"]),
        spatial_weight=float(reward_cfg["spatial"]["weight"]),
        semantic_weight=float(reward_cfg["semantic"]["weight"]),
        iou_threshold=float(reward_cfg["spatial"]["iou_threshold"]),
        parser_name=parser_name,
        invalid_box_semantic_fallback=invalid_fallback,
    )
