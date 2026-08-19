#!/usr/bin/env python3
"""Run the unchanged functional-KV oracle with an LR-ablation validator."""

import ntp_checkpoint_replay_oracle as oracle
from two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr_ablation import (
    validate_known_lr_config,
)


if __name__ == "__main__":
    # The oracle implementation and checkpoint backend are unchanged.  Only
    # its hardcoded LR=2e-5 config validator is generalized to the two sealed
    # diagnostic configs before delegating to the original main function.
    oracle.validate_sampling_ablation_contract = validate_known_lr_config
    oracle.main()

