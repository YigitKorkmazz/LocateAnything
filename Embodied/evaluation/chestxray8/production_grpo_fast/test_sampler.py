#!/usr/bin/env python3
"""Correctness gates for data order / accumulation-window / distributed sharding.

No GPU / model / dataset file required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from production_grpo_fast import sampler  # noqa: E402


def test_epoch_order_is_a_permutation() -> None:
    for epoch in range(3):
        order = sampler.epoch_order(epoch)
        assert sorted(order) == list(range(sampler.TRAIN_SIZE))
        assert order == sampler.epoch_order(epoch), "epoch_order must be deterministic"


def test_epoch_orders_differ_across_epochs() -> None:
    assert sampler.epoch_order(0) != sampler.epoch_order(1)


def test_total_groups_divide_evenly_by_four() -> None:
    assert sampler.TOTAL_GROUPS == sampler.TRAIN_SIZE * sampler.EPOCHS == 15800
    assert sampler.TOTAL_GROUPS % sampler.GROUPS_PER_UPDATE == 0
    assert sampler.TOTAL_OPTIMIZER_STEPS == 3950


def test_single_epoch_no_duplicate_no_skip() -> None:
    seen = set()
    for g in range(sampler.TRAIN_SIZE):
        spec = sampler.resolve_group(g)
        assert spec.epoch == 0
        assert spec.index_in_epoch == g
        seen.add(spec.manifest_index)
    assert seen == set(range(sampler.TRAIN_SIZE))


def test_continuous_stream_no_duplicate_no_skip_within_each_epoch() -> None:
    for epoch in range(3):
        seen = set()
        for idx in range(sampler.TRAIN_SIZE):
            g = epoch * sampler.TRAIN_SIZE + idx
            spec = sampler.resolve_group(g)
            assert spec.epoch == epoch
            seen.add(spec.manifest_index)
        assert seen == set(range(sampler.TRAIN_SIZE))


def test_sample_seeds_deterministic_and_distinct_across_rollouts() -> None:
    spec = sampler.resolve_group(123)
    seeds_a = sampler.rollout_seeds(spec.sample_seed)
    seeds_b = sampler.rollout_seeds(spec.sample_seed)
    assert seeds_a == seeds_b
    assert len(set(seeds_a)) == len(seeds_a) == sampler.GENERATIONS


def test_window_index_and_slot() -> None:
    for g in range(20):
        w, s = sampler.window_index_and_slot(g)
        assert w * sampler.GROUPS_PER_UPDATE + s == g
        assert 0 <= s < sampler.GROUPS_PER_UPDATE


def test_worker_group_indices_partition_exactly_for_supported_world_sizes() -> None:
    for world_size in (1, 2, 3, 4):
        for window_index in (0, 1, 197, 1975):
            covered = []
            for rank in range(world_size):
                indices = sampler.worker_group_indices(window_index, world_size=world_size, rank=rank)
                covered.extend(indices)
            expected = set(range(window_index * 4, window_index * 4 + 4))
            assert sorted(covered) == sorted(expected), (world_size, window_index)
            assert len(covered) == len(set(covered)) == 4


def test_worker_group_indices_two_gpu_split_is_even() -> None:
    for rank in (0, 1):
        indices = sampler.worker_group_indices(5, world_size=2, rank=rank)
        assert len(indices) == 2


def test_worker_group_indices_three_gpu_split_is_2_1_1() -> None:
    counts = [len(sampler.worker_group_indices(5, world_size=3, rank=r)) for r in range(3)]
    assert sorted(counts) == [1, 1, 2]


def test_resume_cursor_matches_window_boundary() -> None:
    for step in (0, 1, 197, 3950):
        g = sampler.resume_cursor_for_optimizer_step(step)
        assert g % sampler.GROUPS_PER_UPDATE == 0
        assert g // sampler.GROUPS_PER_UPDATE == step


def test_heldout_never_referenced() -> None:
    source = Path(sampler.__file__).read_text(encoding="utf-8")
    assert "test_pairs" not in source
    assert "val_pairs" not in source
    assert "heldout" not in source.lower()


def main() -> None:
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} SAMPLER TESTS PASSED")


if __name__ == "__main__":
    main()
