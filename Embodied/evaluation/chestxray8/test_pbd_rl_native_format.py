#!/usr/bin/env python3
"""Lightweight tests for native LocateAnything GRPO format mode."""

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

from eval_locateanything_bbox import (  # noqa: E402
    DIRECT_DISEASE_QUERY_TEMPLATE,
    BOX_RE,
    build_final_user_query,
)
from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.prompt import (  # noqa: E402
    build_native_locateanything_prompt,
    build_rl_user_text,
    native_direct_disease_query,
)
from rl.rewards import (  # noqa: E402
    ProductionRewardPipeline,
    build_reward_pipeline_from_config,
    format_reward,
    parse_native_locateanything_completion,
    spatial_reward,
)
from rl.runtime import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_NATIVE_CONFIG,
    load_resolved_config,
)
from run_pbd_rl_viability import build_viability_plan  # noqa: E402
from sft_common import build_assistant_target  # noqa: E402


class _MockSemanticScorer:
    def __init__(self, value: float = 0.25) -> None:
        self.value = value

    def score(self, image_path, box_norm_1000, query) -> float:
        return float(self.value)


VALID_NATIVE = "<ref>Infiltration</ref><box><110><210><320><430></box>"
VALID_NATIVE_EOS = VALID_NATIVE + "<|im_end|>"


def test_native_prompt_matches_direct_disease_sft_eval() -> None:
    disease = "Infiltration"
    phrase, user_query = build_final_user_query(disease, "direct_disease")
    assert phrase == "Infiltration"
    assert user_query == DIRECT_DISEASE_QUERY_TEMPLATE.format(disease=disease)
    assert user_query == "Locate the Infiltration in this chest X-ray"
    assert native_direct_disease_query(disease) == user_query
    pair = {"disease": disease, "user_query": user_query, "image_path": "/tmp/x.png"}
    assert build_native_locateanything_prompt(pair) == user_query
    assert build_rl_user_text(pair, prompt_mode="native_locateanything") == user_query
    # Must NOT wrap with Chain-of-Box / think-answer instructions.
    text = build_rl_user_text(pair, prompt_mode="native_locateanything")
    assert "<think>" not in text
    assert "<answer>" not in text
    assert "Chain-of-Box" not in text
    assert text == pair["user_query"]


def test_native_assistant_target_grammar() -> None:
    target = build_assistant_target("Infiltration", [[110, 210, 320, 430]])
    assert target == VALID_NATIVE
    assert BOX_RE.search(target) is not None


def test_valid_native_ref_box_output() -> None:
    parsed = parse_native_locateanything_completion(VALID_NATIVE)
    assert parsed.format_valid
    assert parsed.final_box_norm_1000 == (110, 210, 320, 430)
    assert parsed.error is None
    assert format_reward(VALID_NATIVE, parser_name="native_locateanything") == 1.0
    assert format_reward(VALID_NATIVE_EOS, parser_name="native_locateanything") == 1.0


def test_malformed_and_invalid_geometry_native() -> None:
    none_box = "<ref>Mass</ref><box>None</box>"
    assert format_reward(none_box, parser_name="native_locateanything") == 0.0
    malformed = "<ref>Mass</ref><box><1><2><3></box>"
    assert format_reward(malformed, parser_name="native_locateanything") == 0.0
    bad_geom = "<ref>Mass</ref><box><300><200><100><400></box>"
    parsed = parse_native_locateanything_completion(bad_geom)
    assert not parsed.format_valid
    assert parsed.error == "invalid geometry"
    spatial, iou = spatial_reward(
        bad_geom, [[100, 200, 300, 400]], parser_name="native_locateanything"
    )
    assert spatial == 0.0 and iou == 0.0


def test_multiple_box_behavior_matches_single_object_rule() -> None:
    # Existing multi-GT *eval* keeps all preds; RL single-object format reward
    # requires exactly one unambiguous native box (same spirit as CoB answer).
    multi = (
        "<ref>Mass</ref>"
        "<box><10><20><30><40></box>"
        "<box><50><60><70><80></box>"
    )
    parsed = parse_native_locateanything_completion(multi)
    assert not parsed.format_valid
    assert "exactly one" in (parsed.error or "")
    assert format_reward(multi, parser_name="native_locateanything") == 0.0


def test_native_spatial_and_semantic_rewards() -> None:
    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[110, 210, 320, 430]],
    }
    pipeline = ProductionRewardPipeline(
        _MockSemanticScorer(0.4),
        parser_name="native_locateanything",
    )
    components = pipeline.score(VALID_NATIVE, pair)
    assert components.format_reward == 1.0
    assert components.spatial_reward == 1.0
    assert math.isclose(components.semantic_reward, 0.4)
    assert math.isclose(components.total_reward, 2.4)
    assert math.isclose(components.final_iou, 1.0)

    # No think/answer required.
    assert "<think>" not in VALID_NATIVE
    assert "<answer>" not in VALID_NATIVE

    miss = "<ref>Mass</ref><box><900><900><950><950></box>"
    miss_components = pipeline.score(miss, pair)
    assert miss_components.format_reward == 1.0
    assert miss_components.spatial_reward == 0.0
    assert miss_components.final_iou < 0.5


def test_invalid_native_uses_semantic_fallback() -> None:
    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[110, 210, 320, 430]],
    }
    pipeline = ProductionRewardPipeline(
        _MockSemanticScorer(0.9),
        parser_name="native_locateanything",
        invalid_box_semantic_fallback=0.0,
    )
    components = pipeline.score("<ref>Mass</ref><box>None</box>", pair)
    assert components.format_reward == 0.0
    assert components.spatial_reward == 0.0
    assert components.semantic_reward == 0.0
    assert components.total_reward == 0.0


def test_native_yaml_zero_supervised_and_pbd_flags() -> None:
    config = load_resolved_config(DEFAULT_NATIVE_CONFIG)
    assert config["prompt"]["mode"] == "native_locateanything"
    assert config["rewards"]["parser"] == "native_locateanything"
    assert config["objective"]["loss_total"] == "L_GRPO"
    assert config["objective"]["supervised_losses"] == []
    assert config["rollout"]["chain_of_box"] is False
    assert config["rollout"]["force_first_box_block"] is False
    assert config["rollout"]["geometry_repair"] is False
    assert config["rollout"]["reconstruct_actions_from_text"] is False
    assert float(config["rollout"]["temperature"]) == 1.0
    assert int(config["rollout"]["top_k"]) == 0
    assert float(config["rollout"]["top_p"]) == 1.0
    assert float(config["rollout"]["repetition_penalty"]) == 1.0
    assert int(config["objective"]["group_size"]) == 4
    assert float(config["objective"]["ppo_clip_epsilon"]) == 0.2
    assert config["rewards"]["format"]["weight"] == 1.0
    assert config["rewards"]["spatial"]["weight"] == 1.0
    assert config["rewards"]["semantic"]["weight"] == 1.0
    raw = yaml.safe_load(DEFAULT_NATIVE_CONFIG.read_text(encoding="utf-8"))
    assert raw["rewards"]["require_think_answer"] is False


def test_cob_yaml_still_loads() -> None:
    config = load_resolved_config(DEFAULT_CONFIG)
    assert config["prompt"]["mode"] == "chain_of_box"
    assert config["rewards"]["parser"] == "chain_of_box"
    assert config["rollout"]["chain_of_box"] is True


def test_native_viability_dry_run_and_pipeline_factory() -> None:
    config = load_resolved_config(DEFAULT_NATIVE_CONFIG)
    plan = build_viability_plan(config, population_size=790)
    assert plan["sample_count"] == 100
    assert plan["optimizer_updates"] == 0
    pipeline = build_reward_pipeline_from_config(config, _MockSemanticScorer(0.2))
    assert pipeline.parser_name == "native_locateanything"
    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[110, 210, 320, 430]],
    }
    components = pipeline.score(VALID_NATIVE, pair)
    assert components.format_valid
    with tempfile.TemporaryDirectory() as tmp:
        from rl.runtime import assert_new_output_dir, write_json

        out = assert_new_output_dir(Path(tmp) / "native_viability_dry")
        write_json(out / "viability_plan.json", plan)
        assert (out / "viability_plan.json").is_file()


def test_score_from_trace_prefers_committed_box() -> None:
    from rl.pbd_rl import PBDSamplingConfig, RolloutTrace

    pair = {
        "image_path": "/tmp/unused.png",
        "user_query": "Locate the Mass in this chest X-ray",
        "gt_boxes_norm_1000": [[110, 210, 320, 430]],
    }
    pipeline = ProductionRewardPipeline(
        _MockSemanticScorer(0.5), parser_name="native_locateanything"
    )
    # Clean text but committed box misses GT → spatial 0 while format 1.
    trace = RolloutTrace(
        prompt_token_ids=[1],
        generated_token_ids=[],
        blocks=[],
        sampling=PBDSamplingConfig(),
        stopped_on_eos=True,
        truncated=False,
        decoded_text="<ref>Mass</ref><box><900><900><950><950></box>",
        decoder_path="pbd",
        reward_branch="pbd",
        committed_final_box_norm_1000=(900, 900, 950, 950),
        has_unambiguous_committed_box=True,
    )
    components = pipeline.score_from_trace(trace, pair)
    assert components.format_reward == 1.0
    assert components.spatial_reward == 0.0
    assert components.final_box_norm_1000 == (900, 900, 950, 950)
    assert components.has_unambiguous_committed_box


def test_grpo_loss_has_no_supervised_terms() -> None:
    rewards = torch.tensor([2.0, 0.0, 1.0, 0.5])
    advantages = group_relative_advantages(rewards)
    old = torch.zeros(4)
    current = old.clone().requires_grad_(True)
    loss = grpo_clipped_loss(current, old, advantages, clip_epsilon=0.2)
    loss.backward()
    source = inspect.getsource(grpo_clipped_loss).lower()
    for forbidden in ("cross_entropy", "labels", "ntp", "mtp", "bbox"):
        assert forbidden not in source


def main() -> None:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"NATIVE PBD-RL TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
