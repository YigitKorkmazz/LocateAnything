#!/usr/bin/env python3
"""CPU regression checks for ndarray-safe checkpoint comparison."""

import unittest

import numpy as np

from two_gpu_g4_resume_equivalence_oracle import _exact_comparison, _hash, _summary


class ExactComparatorTest(unittest.TestCase):
    def test_numpy_rng_state_roundtrip_is_exact(self):
        state = ("MT19937", np.array([1, 2, 3], dtype=np.uint32), 7, 0, 0.0)
        result = _exact_comparison(state, ("MT19937", state[1].copy(), 7, 0, 0.0), "numpy_rng")
        self.assertTrue(result["equal"])
        self.assertEqual(result["mismatch_count"], 0)

    def test_numpy_rng_state_change_has_clean_path(self):
        left = ("MT19937", np.array([1, 2, 3], dtype=np.uint32), 7, 0, 0.0)
        right = ("MT19937", np.array([1, 9, 3], dtype=np.uint32), 7, 0, 0.0)
        result = _exact_comparison(left, right, "numpy_rng")
        self.assertFalse(result["equal"])
        self.assertEqual(result["first_mismatch"]["path"], "numpy_rng[1]")
        self.assertEqual(result["first_mismatch"]["reason"], "ndarray_value_mismatch")

    def test_bfloat16_exact_and_hashing(self):
        left = __import__("torch").tensor([1.0, 2.0], dtype=__import__("torch").bfloat16)
        same = _exact_comparison(left, left.clone(), "bf16")
        self.assertTrue(same["equal"])
        changed = left.clone(); changed[1] = 3.0
        different = _exact_comparison(left, changed, "bf16")
        self.assertFalse(different["equal"])
        self.assertEqual(different["first_mismatch"]["path"], "bf16")
        self.assertIsInstance(_hash({"tensor": left}), str)
        self.assertEqual(_summary(left)["dtype"], "torch.bfloat16")


if __name__ == "__main__":
    unittest.main()
