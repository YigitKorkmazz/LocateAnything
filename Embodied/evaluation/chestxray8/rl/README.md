# ChestX-ray8 stochastic native PBD GRPO

This path is separate from ordinary `decode_bbox_avg`/hybrid inference. It
samples native six-token blocks, records emitted token actions in
`RolloutTrace`, and replays those actions with the same supports, block
boundaries, position IDs, bidirectional intra-block attention, and KV-cache
truncation.

The experiment objective is exactly `L_total = L_GRPO`. There is no NTP, MTP,
bbox cross-entropy, coordinate regression, or reasoning supervision. The
initial engineering choices are PPO clipping epsilon `0.2`, `G=4`, and
old-policy synchronization every one completed optimizer update. Synchronize
only LoRA and `mlp1` after `optimizer.step()`; old policy remains frozen at all
other times.

## Prompt and rewards

RL uses the dedicated Chain-of-Box prompt in `prompt.py` (also stored in
`chestxray8_grpo_pbd.yaml`). The existing SFT prompt path is untouched.

Rewards score only the final native box inside `<answer>`:

- binary format reward
- binary spatial reward (`IoU > 0.5`)
- frozen MedCLIP ROI/text cosine semantic reward

Intermediate boxes inside `<think>` receive no direct reward.

## Entry points

- Viability (100 samples, no optimizer updates): `run_pbd_rl_viability.py`
- Full GRPO training: `run_pbd_rl_train.py`
- Multimode evaluation: `run_pbd_rl_eval.py`

The first resolved config deliberately uses unfiltered sampling
(`temperature=1`, `top_k=0`, `top_p=1`, `repetition_penalty=1`) so support does
not depend on policy-ranked filtering. Invalid coordinate geometry is retained
as sampled.

## Policy memory

A separate old 3B model is simplest but duplicates the frozen LLM and vision
weights. `PolicySnapshot` supports a lower-memory alternative: serialize
LoRA/projector snapshots and swap them onto one shared frozen base for
old/current scoring. Snapshot scoring must be serialized, under `no_grad` for
old/reference, and the current state must be restored before gradient replay.
Do not swap state while an autograd graph is live.

## MedCLIP identity

`medclip_loader.py` loads local pinned backbone snapshots and verifies all
checkpoint hashes before constructing
`MedCLIPModel(vision_cls=MedCLIPVisionModelViT)`. The valid upstream MedCLIP
commit is `9c3396f20d5d54e4fae241b8cb06ca45848e98c9`; the originally supplied value
contained one extra `4` and is not a Git object.

The resolved experiment settings and backbone revisions are in
`chestxray8_grpo_pbd.yaml`.
