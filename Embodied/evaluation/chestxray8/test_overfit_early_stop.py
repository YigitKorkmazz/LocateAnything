#!/usr/bin/env python3
"""Unit tests for overfit early-stop / metric success logic (no GPU required).

Run:
  /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python \\
    evaluation/chestxray8/test_overfit_early_stop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from train_chestxray8_sft import (  # noqa: E402
    exact_coordinate_match,
    overfit_should_stop_early,
)


def _metrics(*, n=4, n_valid=4, mean_iou=0.2, recall_at_05=0.25, exact=0):
    return {
        "n": n,
        "n_valid": n_valid,
        "mean_iou": mean_iou,
        "recall_at_05": recall_at_05,
        "exact_match_count": exact,
    }


def test_valid_but_low_iou_does_not_trigger_metric_success():
    m = _metrics(n_valid=4, mean_iou=0.2, recall_at_05=0.25)
    stop, reason = overfit_should_stop_early(
        m, disable_early_stop=False, success_miou=0.95, success_recall=None
    )
    assert stop is False, reason
    assert reason.startswith("metric:")


def test_disable_early_stop_overrides_everything():
    m = _metrics(n_valid=4, mean_iou=1.0, recall_at_05=1.0)
    stop, reason = overfit_should_stop_early(
        m,
        disable_early_stop=True,
        success_miou=0.95,
        success_recall=1.0,
    )
    assert stop is False
    assert reason == "disabled"

    # Also overrides legacy validity success
    stop2, reason2 = overfit_should_stop_early(
        m, disable_early_stop=True, success_miou=None, success_recall=None
    )
    assert stop2 is False
    assert reason2 == "disabled"


def test_miou_threshold_stopping():
    m_low = _metrics(mean_iou=0.94, recall_at_05=1.0, n_valid=4)
    stop, _ = overfit_should_stop_early(
        m_low, success_miou=0.95, success_recall=None
    )
    assert stop is False

    m_ok = _metrics(mean_iou=0.95, recall_at_05=0.0, n_valid=1)
    stop_ok, reason = overfit_should_stop_early(
        m_ok, success_miou=0.95, success_recall=None
    )
    assert stop_ok is True
    assert reason.startswith("metric:")


def test_recall_threshold_stopping():
    m_low = _metrics(mean_iou=1.0, recall_at_05=0.99, n_valid=4)
    stop, _ = overfit_should_stop_early(
        m_low, success_miou=None, success_recall=1.0
    )
    assert stop is False

    m_ok = _metrics(mean_iou=0.0, recall_at_05=1.0, n_valid=0)
    stop_ok, reason = overfit_should_stop_early(
        m_ok, success_miou=None, success_recall=1.0
    )
    assert stop_ok is True
    assert "recall" in reason


def test_both_thresholds_required():
    m = _metrics(mean_iou=0.99, recall_at_05=0.5, n_valid=4)
    stop, _ = overfit_should_stop_early(
        m, success_miou=0.95, success_recall=1.0
    )
    assert stop is False

    m2 = _metrics(mean_iou=0.95, recall_at_05=1.0, n_valid=4)
    stop2, _ = overfit_should_stop_early(
        m2, success_miou=0.95, success_recall=1.0
    )
    assert stop2 is True


def test_default_validity_backward_compatible():
    # All valid → success (legacy)
    m = _metrics(n=4, n_valid=4, mean_iou=0.01, recall_at_05=0.0)
    stop, reason = overfit_should_stop_early(
        m, disable_early_stop=False, success_miou=None, success_recall=None
    )
    assert stop is True
    assert reason == "valid_syntax"

    # Not all valid → continue
    m2 = _metrics(n=4, n_valid=3, mean_iou=1.0, recall_at_05=1.0)
    stop2, reason2 = overfit_should_stop_early(
        m2, disable_early_stop=False, success_miou=None, success_recall=None
    )
    assert stop2 is False
    assert reason2 == "continue"


def test_exact_coordinate_match():
    gt = [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]
    pred_ok = [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0]]
    pred_perm = [[50.0, 60.0, 70.0, 80.0], [10.0, 20.0, 30.0, 40.0]]
    pred_bad = [[10.0, 20.0, 30.0, 40.0], [51.0, 60.0, 70.0, 80.0]]
    assert exact_coordinate_match(gt, pred_ok) is True
    assert exact_coordinate_match(gt, pred_perm) is True
    assert exact_coordinate_match(gt, pred_bad) is False
    assert exact_coordinate_match(gt, pred_ok[:1]) is False


def main() -> None:
    test_valid_but_low_iou_does_not_trigger_metric_success()
    print("OK: valid+low IoU does not trigger metric success")
    test_disable_early_stop_overrides_everything()
    print("OK: --disable-overfit-early-stop overrides all early stop")
    test_miou_threshold_stopping()
    print("OK: mIoU threshold stopping")
    test_recall_threshold_stopping()
    print("OK: Recall threshold stopping")
    test_both_thresholds_required()
    print("OK: both thresholds required")
    test_default_validity_backward_compatible()
    print("OK: default validity backward compatible")
    test_exact_coordinate_match()
    print("OK: exact_coordinate_match")
    print("\nALL OVERFIT EARLY-STOP TESTS PASSED")


if __name__ == "__main__":
    main()
