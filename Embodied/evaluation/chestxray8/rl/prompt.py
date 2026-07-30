"""Dedicated Chain-of-Box prompt for ChestX-ray8 RL only."""

from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image


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


def build_chain_of_box_prompt(query: str) -> str:
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a nonempty string")
    return CHAIN_OF_BOX_PROMPT_TEMPLATE.format(query=query)


def build_rl_messages(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the image/user message without touching the existing SFT path."""
    image = Image.open(pair["image_path"]).convert("RGB")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": build_chain_of_box_prompt(pair["user_query"]),
                },
            ],
        }
    ]
