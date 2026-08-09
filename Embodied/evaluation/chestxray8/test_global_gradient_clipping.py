#!/usr/bin/env python3
"""Focused CPU tests for cross-shard global gradient clipping."""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

import two_gpu_g4_grpo_multistep_smoke as training
from rl.runtime import DEFAULT_HYBRID_NATIVE_CONFIG, load_resolved_config


def _parameter(name: str, gradient: float):
    parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    parameter.grad = torch.tensor([gradient], dtype=torch.float32)
    return name, parameter


class GlobalGradientClippingTest(unittest.TestCase):
    def test_norm_below_threshold_is_not_scaled(self):
        parameters = [_parameter("a", 0.3), _parameter("b", 0.4)]
        before = [parameter.grad.clone() for _, parameter in parameters]
        report = training._global_clip_gradients(
            parameters, max_grad_norm=1.0, expected_parameter_count=2
        )
        self.assertAlmostEqual(report["global_grad_norm_before_clip"], 0.5, places=6)
        self.assertAlmostEqual(report["global_grad_norm_after_clip"], 0.5, places=6)
        self.assertEqual(report["clip_coefficient"], 1.0)
        self.assertFalse(report["clipping_applied"])
        for original, (_, parameter) in zip(before, parameters):
            self.assertTrue(torch.equal(original, parameter.grad))

    def test_norm_above_threshold_uses_one_shared_coefficient(self):
        parameters = [_parameter("early_shard", 3.0), _parameter("late_shard", 4.0)]
        report = training._global_clip_gradients(
            parameters, max_grad_norm=1.0, expected_parameter_count=2
        )
        self.assertAlmostEqual(report["global_grad_norm_before_clip"], 5.0, places=6)
        self.assertAlmostEqual(report["clip_coefficient"], 0.2, places=7)
        self.assertTrue(report["clipping_applied"])
        self.assertAlmostEqual(parameters[0][1].grad.item(), 0.6, places=6)
        self.assertAlmostEqual(parameters[1][1].grad.item(), 0.8, places=6)
        self.assertAlmostEqual(report["global_grad_norm_after_clip"], 1.0, places=6)

    def test_zero_gradients(self):
        parameters = [_parameter("a", 0.0), _parameter("b", 0.0)]
        report = training._global_clip_gradients(
            parameters, max_grad_norm=1.0, expected_parameter_count=2
        )
        self.assertEqual(report["global_grad_norm_before_clip"], 0.0)
        self.assertEqual(report["global_grad_norm_after_clip"], 0.0)
        self.assertEqual(report["clip_coefficient"], 1.0)
        self.assertFalse(report["clipping_applied"])

    def test_nonfinite_gradient_fails_before_optimizer_step(self):
        parameters = [_parameter("finite", 1.0), _parameter("bad", float("nan"))]
        optimizer = Mock()
        with self.assertRaisesRegex(RuntimeError, "nonfinite gradient"):
            training._global_clip_gradients(
                parameters, max_grad_norm=1.0, expected_parameter_count=2
            )
        optimizer.step.assert_not_called()

    def test_exact_threshold_does_not_scale(self):
        parameters = [_parameter("a", 3.0), _parameter("b", 4.0)]
        report = training._global_clip_gradients(
            parameters, max_grad_norm=5.0, expected_parameter_count=2
        )
        self.assertTrue(math.isclose(report["global_grad_norm_before_clip"], 5.0))
        self.assertEqual(report["clip_coefficient"], 1.0)
        self.assertFalse(report["clipping_applied"])

    def test_all_510_parameters_are_included_exactly_once(self):
        parameters = [_parameter(f"p{index}", 1.0) for index in range(510)]
        report = training._global_clip_gradients(
            parameters, max_grad_norm=100.0, expected_parameter_count=510
        )
        self.assertEqual(report["trainable_parameter_count"], 510)
        self.assertEqual(report["unique_trainable_parameter_count"], 510)
        duplicated = parameters + [parameters[0]]
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            training._global_clip_gradients(
                duplicated, max_grad_norm=100.0, expected_parameter_count=511
            )

    def test_yaml_runtime_contract_is_authoritative(self):
        config = load_resolved_config(DEFAULT_HYBRID_NATIVE_CONFIG)
        self.assertEqual(
            training._effective_runtime_contract(config),
            training.EXPECTED_RUNTIME_CONTRACT,
        )
        contradictory = copy.deepcopy(config)
        contradictory["runtime_contract"]["replay_cache"] = False
        with self.assertRaisesRegex(RuntimeError, "differs from validated"):
            training._effective_runtime_contract(contradictory)
        hidden_alias = copy.deepcopy(config)
        hidden_alias["training"]["gradient_replay_use_cache"] = False
        with self.assertRaisesRegex(RuntimeError, "contradicts"):
            training._effective_runtime_contract(hidden_alias)


if __name__ == "__main__":
    unittest.main()
