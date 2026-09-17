#!/usr/bin/env python3
"""Static gate: no file in this backend references the heldout/val manifests.

`splits/test_pairs_seed42.jsonl` (heldout194) and `splits/val_pairs_seed42.jsonl`
(the historical internal-validation split, explicitly forbidden by the task
spec) must never be named anywhere in this package's source.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
# Exact manifest filenames only (not the English word "heldout", which this
# backend's own documentation legitimately uses to describe what it avoids).
FORBIDDEN_SUBSTRINGS = ("test_pairs_seed42", "val_pairs_seed42")


def main() -> None:
    checked = 0
    for path in sorted(HERE.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for needle in FORBIDDEN_SUBSTRINGS:
            assert needle not in text, f"{path} references forbidden manifest file {needle!r}"
        checked += 1
    print(f"PASS no-heldout-access static scan: {checked} files checked, 0 forbidden references")


if __name__ == "__main__":
    main()
