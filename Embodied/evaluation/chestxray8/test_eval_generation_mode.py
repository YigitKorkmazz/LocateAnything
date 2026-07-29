#!/usr/bin/env python3
"""Regression tests for evaluation generation mode, counts, and output safety."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from eval_finetuned_comparison import (  # noqa: E402
    assert_eval_outputs_available,
    summarize_rows,
)
from eval_locateanything_bbox import run_inference  # noqa: E402


class RecordingWorker:
    def __init__(self):
        self.calls = []

    def predict(self, image, query, **kwargs):
        self.calls.append(("predict", query, kwargs))
        return {"answer": "<ref>Mass</ref><box><1><2><3><4></box>"}

    def ground_multi(self, image, phrase, **kwargs):
        self.calls.append(("ground_multi", phrase, kwargs))
        return {"answer": "<ref>Mass</ref><box><1><2><3><4></box>"}


def test_generation_mode_passthrough_and_hybrid_default():
    image = Image.new("RGB", (1, 1))
    worker = RecordingWorker()
    before_answer, _ = run_inference(
        worker,
        image,
        "Mass",
        prompt_strategy="direct_disease",
        final_query="Locate the Mass in this chest X-ray",
    )
    assert before_answer == "<ref>Mass</ref><box><1><2><3><4></box>"
    assert worker.calls[-1][2]["generation_mode"] == "hybrid"

    run_inference(
        worker,
        image,
        "Mass",
        prompt_strategy="direct_disease",
        final_query="Locate the Mass in this chest X-ray",
        generation_mode="slow",
    )
    assert worker.calls[-1][2]["generation_mode"] == "slow"


def test_parse_failure_and_no_prediction_counts():
    rows = [
        {
            "Image": "a.png",
            "Status": "ok",
            "Inference Time": 0.1,
            "Number of GT Boxes": 1,
            "Number of Predicted Boxes": 1,
            "Valid Structured Output": True,
        },
        {
            "Image": "b.png",
            "Status": "ok",
            "Inference Time": 0.1,
            "Number of GT Boxes": 1,
            "Number of Predicted Boxes": 0,
            "Valid Structured Output": False,
        },
    ]
    summary = summarize_rows(
        rows,
        [
            {"IoU": 0.5},
            {"IoU": 0.0},
        ],
        "test",
    )
    assert summary["Parsed-Box Rate"] == 0.5
    assert summary["No-Prediction Rate"] == 0.5
    assert summary["Parse-Failure Count"] == 1
    assert summary["No-Prediction Count"] == 1


def test_occupied_output_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output = root / "eval"
        output.mkdir()
        assert_eval_outputs_available(output, output / "result.xlsx")
        (output / "old.log").write_text("occupied\n")
        try:
            assert_eval_outputs_available(output, output / "result.xlsx")
        except FileExistsError:
            pass
        else:
            raise AssertionError("Occupied evaluation directory was not refused")


def main() -> None:
    test_generation_mode_passthrough_and_hybrid_default()
    test_parse_failure_and_no_prediction_counts()
    test_occupied_output_is_refused()
    print("EVALUATION GENERATION-MODE TESTS PASSED")


if __name__ == "__main__":
    main()
