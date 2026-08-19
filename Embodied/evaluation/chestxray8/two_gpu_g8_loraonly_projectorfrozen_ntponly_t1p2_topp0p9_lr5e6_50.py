#!/usr/bin/env python3
"""Guarded entry point for the LR=5e-6, 50-update diagnostic."""

from pathlib import Path

from two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr_ablation import main_for_run

HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "rl/chestxray8_grpo_native_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr5e6_50.yaml"
RUN_NAME = "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR5E6_50"
EXECUTE_FLAG = "--execute-authorized-lr5e6-diagnostic"


if __name__ == "__main__":
    main_for_run(
        default_spec=DEFAULT_SPEC,
        expected_lr=5e-6,
        run_name=RUN_NAME,
        file_tag="lr5e6",
        execute_flag=EXECUTE_FLAG,
    )

