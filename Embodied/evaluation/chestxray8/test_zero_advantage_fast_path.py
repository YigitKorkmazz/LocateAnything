#!/usr/bin/env python3
"""CPU regressions for the mathematically exact zero-advantage fast path."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import sys
from pathlib import Path

import torch

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

import two_gpu_g4_grpo_multistep_smoke as training


class ZeroAdvantageFastPathTest(unittest.TestCase):
    @staticmethod
    def _guard_group(*, matching=True):
        trajectories = []
        for _ in range(training.GROUP_SIZE):
            predicates = {
                "token_budget_truncation": matching,
                "generated_length_equals_maximum_reachable": True,
                "branch_is_none": True,
                "parse_error_is_no_native_box": True,
                "total_reward_exactly_zero": True,
            }
            trajectories.append(
                {"guard": {"predicates": predicates, "matches": all(predicates.values())}}
            )
        return {"trajectories": trajectories}

    def test_equal_rewards_do_not_call_replay(self):
        advantages = training.group_relative_advantages([0.0, 0.0, 0.0, 0.0])
        replay = Mock()
        result = training._replay_each_if_nonzero(
            advantages, [object()] * 4, replay
        )
        self.assertEqual(result, [])
        replay.assert_not_called()

    def test_mixed_rewards_replay_exactly_four_trajectories(self):
        advantages = training.group_relative_advantages([0.0, 1.0, 0.5, 1.0])
        replay = Mock(side_effect=lambda index, _trace: index)
        result = training._replay_each_if_nonzero(
            advantages, [object()] * 4, replay
        )
        self.assertEqual(result, [0, 1, 2, 3])
        self.assertEqual(replay.call_count, 4)

    def test_skip_advances_only_cursor_and_skip_count(self):
        attempted_group_count = 104  # incremented before reward evaluation
        optimizer_step_count = 100
        cursor, skipped = training._advance_skipped_group(103, 3)
        self.assertEqual(attempted_group_count, 104)
        self.assertEqual(cursor, 104)
        self.assertEqual(skipped, 4)
        self.assertEqual(optimizer_step_count, 100)

    def test_attempt_cap_summary_counters_can_track_a_skipped_group(self):
        summary = {
            "attempted_group_count": 103,
            "skipped_zero_advantage_group_count": 3,
            "optimizer_step_count": 100,
        }
        attempted = 104
        cursor, skipped = training._advance_skipped_group(103, 3)
        summary.update(
            {
                "attempted_group_count": attempted,
                "skipped_zero_advantage_group_count": skipped,
                "optimizer_step_count": 100,
                "total_trajectories_across_checkpoint_lineage": attempted * 4,
            }
        )
        self.assertEqual(cursor, 104)
        self.assertEqual(summary["attempted_group_count"], 104)
        self.assertEqual(summary["skipped_zero_advantage_group_count"], 4)
        self.assertEqual(summary["optimizer_step_count"], 100)

    def test_resume_after_skip_reproduces_next_sample_and_seeds(self):
        cursor, skipped = training._advance_skipped_group(103, 3)
        checkpoint = {
            "optimizer_step_count": 100,
            "attempted_group_count": 104,
            "skipped_zero_advantage_group_count": skipped,
            "sample_cursor": cursor,
        }
        uninterrupted = training._attempt_schedule(
            seed=42,
            attempted_group_count=105,
            sample_cursor=104,
            sample_count=90,
        )
        resumed = training._attempt_schedule(
            seed=42,
            attempted_group_count=checkpoint["attempted_group_count"] + 1,
            sample_cursor=checkpoint["sample_cursor"],
            sample_count=90,
        )
        self.assertEqual(resumed, uninterrupted)

    def test_skip_metrics_are_complete(self):
        with patch.object(
            training,
            "_trace_summary",
            side_effect=lambda _trace, _component: {"generated_token_count": 7},
        ):
            records = training._zero_advantage_trajectory_records(
                [object()] * 4, [object()] * 4, [100, 101, 102, 103]
            )
        self.assertEqual(len(records), 4)
        for index, record in enumerate(records):
            self.assertEqual(record["group_index"], index)
            self.assertEqual(record["advantage"], 0.0)
            self.assertEqual(record["loss_grpo_unscaled"], 0.0)
            self.assertEqual(record["loss_grpo_scaled"], 0.0)
            self.assertFalse(record["replay_executed"])
            self.assertIsNone(record["cumulative_gradient_norm"])
            self.assertIn("not_computed", record["gradient_norm_not_computed_reason"])
        group = training._zero_advantage_group_record(
            {"reward_group": {"rewards": [0.0] * 4, "advantages": [0.0] * 4}},
            trajectories=records,
            optimizer_step_count=100,
            sample_cursor=104,
            skipped_count=4,
            optimizer_state_devices={},
            per_gpu_memory={},
        )
        for key in (
            "skipped_zero_advantage",
            "optimizer_step_executed",
            "replay_executed",
            "loss",
            "per_trajectory_losses",
            "gradient_norm",
            "gradient_norm_not_computed_reason",
        ):
            self.assertIn(key, group)
        self.assertTrue(group["skipped_zero_advantage"])
        self.assertFalse(group["optimizer_step_executed"])
        self.assertFalse(group["replay_executed"])
        self.assertEqual(group["loss"], 0.0)
        self.assertEqual(group["per_trajectory_losses"], [0.0] * 4)

    def test_fresh_start_guard_positive_case_requires_all_32(self):
        groups = [self._guard_group() for _ in range(training.FRESH_START_GUARD_GROUPS)]
        status = training._fresh_start_eight_group_guard_status(groups)
        self.assertTrue(status["triggered"])
        self.assertEqual(status["observed_trajectory_count"], 32)

    def test_fresh_start_guard_negative_cases_do_not_trigger(self):
        only_seven = [self._guard_group() for _ in range(7)]
        self.assertFalse(training._fresh_start_eight_group_guard_status(only_seven)["triggered"])
        one_good_trajectory = [self._guard_group() for _ in range(8)]
        one_good_trajectory[-1] = self._guard_group(matching=False)
        self.assertFalse(
            training._fresh_start_eight_group_guard_status(one_good_trajectory)["triggered"]
        )

    def test_fresh_start_guard_predicates_are_exact(self):
        trace = SimpleNamespace(
            generated_token_ids=[1] * 510,
            truncated=True,
            stop_reason="proposal_exceeds_remaining_token_budget",
            reward_branch="none",
        )
        component = SimpleNamespace(parse_error="no native box", total_reward=0.0)
        result = training._fresh_start_trajectory_guard_predicates(
            trace, component, max_new_tokens=512, block_size=6
        )
        self.assertTrue(result["matches"])
        trace.generated_token_ids.pop()
        self.assertFalse(
            training._fresh_start_trajectory_guard_predicates(
                trace, component, max_new_tokens=512, block_size=6
            )["matches"]
        )


if __name__ == "__main__":
    unittest.main()
