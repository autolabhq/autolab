# FLUX.2 klein LoRA Training — musubi-tuner

Train a LoRA adapter on FLUX.2 klein base 9B with musubi-tuner so the model
learns a specific naruto-style visual concept.

Running `bash /app/train.sh` as shipped does not produce a working LoRA at
`/app/output/lora.safetensors`.

## Setup

| Item | Path / Value |
|------|--------------|
| DiT weights | `/models/dit/flux2-klein-base-9b.safetensors` |
| VAE | `/models/vae/ae.safetensors` |
| Text Encoder (Qwen3 8B) | `/models/text_encoder/` |
| Framework | `/opt/musubi-tuner/` |
| GPU | Single L40S (48 GB VRAM) |
| Editable | `/app/train.sh`, `/app/dataset.toml` |
| Concept data | `/data/concept/` (15 images + captions) |
| Eval prompts | `/app/eval_prompts.txt` (8 prompts) |
| Local eval | `python3 /app/evaluate.py` |

The training script is a 3-stage pipeline: cache latents → cache text encoder
outputs → train LoRA. The trained adapter is expected at
`/app/output/lora.safetensors`.

## Your Goal

Maximize the **composite quality score** on 8 evaluation prompts:

- CLIP image similarity (40%) — style match to reference images
- DINO structural similarity (35%) — semantic feature similarity
- CLIP text alignment (25%) — prompt adherence

## Evaluation

The verifier runs `/app/evaluate.py` with your trained LoRA and computes the
weighted composite. Score the same way locally:

```bash
python3 /app/evaluate.py
```

## Rules

- Do not modify `/app/evaluate.py`, `/app/eval_prompts.txt`, or any file under
  `/models/`.
- No external network access.
- Single L40S GPU (48 GB VRAM).
- Time budget: 4 hours (agent will be stopped after this).
