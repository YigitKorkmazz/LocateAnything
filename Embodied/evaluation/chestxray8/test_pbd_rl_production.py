#!/usr/bin/env python3
"""Lightweight production-path tests for Chain-of-Box PBD-RL."""

from __future__ import annotations

import inspect
import math
import sys
import tempfile
from pathlib import Path

import torch
import yaml

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.prompt import (  # noqa: E402
    CHAIN_OF_BOX_PROMPT_TEMPLATE,
    build_chain_of_box_prompt,
)
from rl.rewards import (  # noqa: E402
    ProductionRewardPipeline,
    format_reward,
    parse_chain_of_box_completion,
    spatial_reward,
)
from rl.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    fixed_sample_indices,
    load_resolved_config,
)
from run_pbd_rl_viability import build_viability_plan  # noqa: E402


class _MockSemanticScorer:
    def __init__(self, value: float = 0.25) -> None:
        self.value = value

    def score(self, image_path, box_norm_1000, query) -> float:
        return float(self.value)


VALID = (
    "<think>\n"
    "Looking near the lower lung.\n"
    "<box><100><200><300><400></box>\n"
    "Confirming the finding.\n"
    "</think>\n"
    "<answer>\n"
    "<box><110><210><320><430></box>\n"
    "</answer>"
)


def test_exact_prompt_construction() -> None:
    query = "Locate the Infiltration in this chest X-ray"
    rendered = build_chain_of_box_prompt(query)
    assert '"{query}"' not in rendered
    assert f'"{query}"' in rendered
    assert rendered == CHAIN_OF_BOX_PROMPT_TEMPLATE.format(query=query)
    config = load_resolved_config(DEFAULT_CONFIG)
    assert config["prompt"]["template"].format(query=query) == rendered


def test_final_box_extraction_ignores_intermediate_boxes() -> None:
    parsed = parse_chain_of_box_completion(VALID)
    assert parsed.format_valid
    assert parsed.final_box_norm_1000 == (110, 210, 320, 430)
    assert "<box><100><200><300><400></box>" in (parsed.think_text or "")


def test_format_spatial_and_semantic_rewards() -> None:
    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[110, 210, 320, 430]],
    }
    pipeline = ProductionRewardPipeline(_MockSemanticScorer(0.4))
    components = pipeline.score(VALID, pair)
    assert components.format_reward == 1.0
    assert components.spatial_reward == 1.0
    assert math.isclose(components.semantic_reward, 0.4)
    assert math.isclose(components.total_reward, 2.4)
    assert math.isclose(components.final_iou, 1.0)


def test_invalid_output_behavior() -> None:
    bad = "<think>no answer</think>"
    assert format_reward(bad) == 0.0
    spatial, iou = spatial_reward(bad, [[10, 20, 30, 40]])
    assert spatial == 0.0 and iou == 0.0
    multi = (
        "<think></think>\n"
        "<answer><box><1><2><3><4></box><box><5><6><7><8></box></answer>"
    )
    assert format_reward(multi) == 0.0
    after = VALID + "\nextra"
    assert format_reward(after) == 0.0
    # Trailing eos markers are ignored for reward parsing.
    assert format_reward(VALID + "<|im_end|>") == 1.0


def test_viability_dry_run_configuration() -> None:
    config = load_resolved_config(DEFAULT_CONFIG)
    plan = build_viability_plan(config, population_size=790)
    assert plan["sample_count"] == 100
    assert plan["optimizer_updates"] == 0
    assert plan["save_checkpoints"] is False
    assert len(plan["sample_indices"]) == 100
    assert plan["sample_indices"] == fixed_sample_indices(790, 100, 42)
    with tempfile.TemporaryDirectory() as tmp:
        from rl.runtime import assert_new_output_dir, write_json

        out = assert_new_output_dir(Path(tmp) / "viability_dry")
        write_json(out / "viability_plan.json", plan)
        assert (out / "viability_plan.json").is_file()


def test_one_optimizer_step_grpo_uses_production_rewards_only() -> None:
    pipeline = ProductionRewardPipeline(_MockSemanticScorer(0.1))
    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[110, 210, 320, 430]],
    }
    completions = [
        VALID,
        "<think>x</think><answer><box><900><900><950><950></box></answer>",
        "<think></think>",
        VALID.replace("<box><110><210><320><430></box>", "<box><120><220><330><440></box>"),
    ]
    components = [pipeline.score(text, pair) for text in completions]
    rewards = torch.tensor([c.total_reward for c in components], dtype=torch.float32)
    advantages = group_relative_advantages(rewards)
    old = torch.zeros(4)
    current = old.clone().requires_grad_(True)
    loss = grpo_clipped_loss(current, old, advantages, clip_epsilon=0.2)
    loss.backward()
    assert current.grad is not None
    source = inspect.getsource(grpo_clipped_loss).lower()
    for forbidden in ("cross_entropy", "labels", "ntp", "mtp"):
        assert forbidden not in source
    assert abs(float(torch.exp(current.detach() - old).mean()) - 1.0) < 1e-6


def test_resolved_yaml_has_no_supervised_loss() -> None:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    assert config["objective"]["loss_total"] == "L_GRPO"
    assert config["objective"]["supervised_losses"] == []
    assert config["rollout"]["force_first_box_block"] is False
    assert config["rollout"]["chain_of_box"] is True
    assert config["rewards"]["use_final_answer_box_only"] is True
    assert config["rewards"]["intermediate_box_reward"] is False


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PBD-RL PRODUCTION TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
