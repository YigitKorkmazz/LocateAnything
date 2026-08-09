"""RL prompt builders for ChestX-ray8 GRPO.

Two modes:
  - chain_of_box: wraps pair["user_query"] in the Chain-of-Box instruction
  - native_locateanything: uses the bare SFT/eval user query unchanged
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PIL import Image

from eval_locateanything_bbox import (
    DIRECT_DISEASE_QUERY_TEMPLATE,
    build_final_user_query,
)


CHAIN_OF_BOX_PROMPT_TEMPLATE = """You are given a chest X-ray image and a medical finding.

Locate the image region corresponding to the following finding:
"{query}"

First output a step-by-step reasoning process inside <think> and </think>
tags.

Whenever you identify or discuss a specific region of interest during
reasoning, append its bounding box using the native LocateAnything format:

<box><x1><y1><x2><y2></box>

Then output exactly one final bounding box inside <answer> and </answer>
tags.

The output must follow exactly this structure:

<think>
Textual reasoning containing zero or more intermediate
<box><x1><y1><x2><y2></box> blocks.
</think>
<answer>
Exactly one final <box><x1><y1><x2><y2></box> block.
</answer>

Rules:
- Intermediate boxes are allowed only inside <think>.
- The final prediction is only the box inside <answer>.
- Use native single coordinate tokens <0> through <1000>.
- Coordinate order is x1, y1, x2, y2.
- Do not output more than one box inside <answer>.
- Do not output any text after </answer>."""

PROMPT_MODE_CHAIN_OF_BOX = "chain_of_box"
PROMPT_MODE_NATIVE = "native_locateanything"
VALID_PROMPT_MODES = (PROMPT_MODE_CHAIN_OF_BOX, PROMPT_MODE_NATIVE)


def build_chain_of_box_prompt(query: str) -> str:
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a nonempty string")
    return CHAIN_OF_BOX_PROMPT_TEMPLATE.format(query=query)


def native_direct_disease_query(disease: str) -> str:
    """Exact executable query from the SFT/eval direct_disease template."""
    _, user_query = build_final_user_query(disease, "direct_disease")
    expected = DIRECT_DISEASE_QUERY_TEMPLATE.format(disease=disease)
    if user_query != expected:
        raise RuntimeError(
            f"direct_disease query mismatch: {user_query!r} != {expected!r}"
        )
    return user_query


def build_native_locateanything_prompt(pair: Dict[str, Any]) -> str:
    """Reuse the exact stored SFT/eval user_query (no CoB wrapper).

    Train/test JSONL pairs already contain
    ``user_query = "Locate the {Disease} in this chest X-ray"`` from
    ``build_final_user_query(..., "direct_disease")``.
    """
    query = pair.get("user_query")
    if not isinstance(query, str) or not query:
        raise ValueError("pair['user_query'] must be a nonempty string")
    disease = pair.get("disease")
    if disease:
        expected = native_direct_disease_query(str(disease))
        if query != expected:
            raise ValueError(
                "pair user_query does not match direct_disease SFT/eval prompt: "
                f"{query!r} != {expected!r}"
            )
    return query


def resolve_prompt_mode(config: Optional[Dict[str, Any]] = None) -> str:
    if config is None:
        return PROMPT_MODE_CHAIN_OF_BOX
    mode = config.get("prompt", {}).get("mode", PROMPT_MODE_CHAIN_OF_BOX)
    if mode not in VALID_PROMPT_MODES:
        raise ValueError(f"unsupported prompt.mode={mode!r}")
    return mode


def build_rl_user_text(
    pair: Dict[str, Any],
    *,
    prompt_mode: str = PROMPT_MODE_CHAIN_OF_BOX,
) -> str:
    if prompt_mode == PROMPT_MODE_NATIVE:
        return build_native_locateanything_prompt(pair)
    if prompt_mode == PROMPT_MODE_CHAIN_OF_BOX:
        return build_chain_of_box_prompt(pair["user_query"])
    raise ValueError(f"unsupported prompt_mode={prompt_mode!r}")


def build_rl_messages(
    pair: Dict[str, Any],
    *,
    prompt_mode: str = PROMPT_MODE_CHAIN_OF_BOX,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build the image/user message without touching the existing SFT path."""
    mode = resolve_prompt_mode(config) if config is not None else prompt_mode
    image = Image.open(pair["image_path"]).convert("RGB")
    return [
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
