# ChestX-ray8 LocateAnything-3B Fine-Tuning

Patient-level 80/20 split (seed=42), **direct-disease** queries, single-GPU A4000 pipeline.

## Target format (from LocateAnything native spec)

```
User: Locate the {Disease} in this chest X-ray
Assistant: <ref>{Disease}</ref><box><x1><y1><x2><y2></box>...
```

Example: `Locate the Atelectasis in this chest X-ray` → `<ref>Atelectasis</ref><box>...`

Coordinates are **normalized integers in `[0, 1000]`** as dedicated tokens `<0>`…`<1000>`.

## Experiments

| Run | Trainable |
|---|---|
| `lora` | LLM LoRA on attention + MLP (`q/k/v/o`, `gate/up/down`); vision / `mlp1` / embeds frozen |
| `lora_projector` | Same LoRA + fully fine-tuned `mlp1` (dual LR) |
| base zero-shot | No training |

## Default hyperparameters

| Setting | Full SFT | LoRA |
|---|---|---|
| Learning rate | **2e-5** (official continual-SFT default) | **2e-5** (official LoRA default; 1e-4 collapsed at step 44) |
| Max grad norm | **1.0** | **1.0** |
| Batch size | 1 | 1 |
| Grad accumulation | 8 | 8 |
| Epochs | 3 | 5 |
| Warmup ratio | 0.03 | 0.03 |
| Weight decay | 0.01 | 0.01 |
| Max seq length | 4096 | 4096 |
| Precision | bf16 | bf16 |
| Grad checkpoint | on | on |
| Optimizer | AdamW8bit | AdamW8bit |
| LoRA r/α/dropout | — | 8 / 16 / 0.05 |

Held-out test = 20% patients. Validation (for checkpoint selection only) is carved from the 80% train partition (~72/8/20 overall).

## Validation status (debug runs)

- Dataset + leakage checks: OK
- Tokenized supervised example: OK
- LoRA one-step: OK (finite loss, ~12 GB peak, parseable inference)
- Full SFT one-step: **CUDA OOM on 1× RTX A4000 16 GB** (peak ~14 GB before failure). Full SFT is **not feasible** on this GPU without changing the task/resolution. Do not silently fall back to LoRA.

## Commands

See the final agent response for exact copy-paste commands.
