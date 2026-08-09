"""Why cacheless Bfix ≠ production-cached A (compare-blocks evidence).

Measured on the fixed Hybrid smoke trace
(``smoke_exact_single_gpu_step1/rollout_traces.jsonl``, trace 0):

    A  production cached (carry_cache=True, model use_cache=True):
        total = -100.66619873046875
    Bfix full-prefix recompute (carry_cache=False, model use_cache=True):
        total = -100.56805419921875
    A - Bfix = -0.09814453125

Block 0 matches exactly (cache_before=0 ⇒ both run the same MTP window).
Divergence starts at block 1.

Root causes (stacked)
---------------------
1. **Structural protocol, not only the MTP mask helper.**
   Production advances a truncated legacy KV cache after every block:
   truncate to the current prefix length, then append committed actions to
   ``generated`` without putting those new tokens into the cache until the
   *next* block consumes them as the incremental suffix.
   Bfix sets ``past=None`` every block and forwards the entire
   ``window = prefix ‖ dup ‖ mask_tail`` in one SDPA call. That is a different
   q/kv layout and a different reduction order in bf16, even when the same
   ``update_causal_mask_for_one_gen_window_2d`` helper is selected.

2. **Legacy B also switched masks.**
   ``legacy_nocache_masks=True`` forces model ``use_cache=False``, selecting
   ``update_causal_mask_with_pad_non_visible_2d`` instead of
   ``update_causal_mask_for_one_gen_window_2d``. That path is farther from A
   (total ≈ -100.923) and is not Bfix.

3. **NTP rows.**
   Mask-helper choice is irrelevant for NTP (plain causal), but A still ≠
   Bfix because A runs q_len≈1 over cache while Bfix prefills the full
   prefix.

Exact autograd-safe replacement
-------------------------------
Do **not** accept Bfix. Reproduce the production incremental
truncate/advance machine with stop-grad past tensors:

* windows, ``position_ids[0, -block_size:] -= 1``, and model ``use_cache=True``
  identical to production;
* carried ``past_key_values`` detached before each graded block so LoRA
  autograd history cannot accumulate (24GB disguised-OOM);
* per-block ``(∂ℓ/∂logπ · logp_i).backward()`` for memory-safe truncated BPTT.

See ``HybridRolloutReplayer.score_autograd_safe`` /
``accumulate_autograd_safe_grads``.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Canonical numbers from diag_nova/compare_blocks.json (do not silently "fix").
A_TOTAL = -100.66619873046875
BFIX_TOTAL = -100.56805419921875
A_MINUS_BFIX = -0.09814453125


def bfix_discrepancy_report(
    *,
    a_block_logps: List[float],
    bfix_block_logps: List[float],
) -> Dict[str, Any]:
    if len(a_block_logps) != len(bfix_block_logps):
        raise RuntimeError("block logp lists must have equal length")
    diffs = [float(a - b) for a, b in zip(a_block_logps, bfix_block_logps)]
    first = next((i for i, d in enumerate(diffs) if abs(d) > 0.0), None)
    return {
        "a_total": float(sum(a_block_logps)),
        "bfix_total": float(sum(bfix_block_logps)),
        "a_minus_bfix_total": float(sum(a_block_logps) - sum(bfix_block_logps)),
        "first_diverging_block": first,
        "per_block_a_minus_bfix": diffs,
        "diagnosis": (
            "Bfix full-prefix recompute is not an unroll of production truncated "
            "KV replay; reject until every block logp matches A."
        ),
        "canonical_reference": {
            "A_TOTAL": A_TOTAL,
            "BFIX_TOTAL": BFIX_TOTAL,
            "A_MINUS_BFIX": A_MINUS_BFIX,
        },
    }
