# Multilingual OCR Fine-Tuning with DeepSeek-OCR

Fine-tune DeepSeek-OCR (3B) with Unsloth LoRA to minimize Character Error Rate on Persian and Bengali text images.

## Setup

| Item | Path |
|------|------|
| Editable script | `/app/train.py` |
| Training entrypoint | `/app/train.sh` |
| Local eval | `python3 /app/evaluate_local.py` |
| Base model | `/models/DeepSeek-OCR` (3B) |
| Training data | `/data/ocr_train/` — 9,000 mixed Persian+Bengali text images |

Pre-loaded datasets: 4,500 Persian (synthetic print) + 4,500 Bengali (synthetic print).

## Your Goal

Minimize average CER across both languages on 400 held-out images (200 Persian + 200 Bengali).

## Evaluation

```bash
python3 /app/evaluate_local.py   # approximate local check (20 samples)
```

The verifier evaluates all 400 images and computes:
```
avg_cer = 0.5 × persian_cer + 0.5 × bengali_cer
reward = reference_cer / avg_cer
```

Lower CER = higher reward.

## Rules

- Edit `/app/train.py` only
- Do NOT modify `/orig/`, or `/models/`
- LoRA adapter must be saved to `/app/output/`
- No external network access
- Single GPU
- Time budget: 8 hours
