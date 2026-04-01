# Multi-Source Visual Math Reasoning via GRPO

Fine-tune Qwen2.5-VL-7B with GRPO to maximize accuracy on MathVista visual math problems.

## Setup

| Item | Path / Value |
|------|-------------|
| Training script | `/app/train.py` (editable) |
| Reward functions | `/app/rewards.py` (editable) |
| Training entrypoint | `bash /app/train.sh` |
| Local eval | `python3 /app/evaluate_local.py` |
| Base model | `/models/Qwen2.5-VL-7B-Instruct-bnb-4bit` (4-bit quantized) |

## Training Data Sources

| Dataset | Path | Size | Content |
|---------|------|------|---------|
| Geometry3K | `/data/geometry3k/` | ~2400 | Geometric reasoning with diagrams |
| MathVision | `/data/mathvision/` | ~2000 | Competition-level visual math |
| ChartQA | `/data/chartqa/` | ~1500 | Chart/graph understanding |

All datasets use a unified format: `question`, `answer`, `image`.

## Your Goal

Maximize `mathvista_accuracy` on 100 held-out MathVista problems. A retention gate applies: if general VQA accuracy drops more than 10% relative to the base model, the score is zero.

## Evaluation

```bash
python3 /app/evaluate_local.py   # quick check (20 MathVista + 10 VQA)
```

Reward = `your_score / reference_score`. Matching the reference scores 1.0.

## Rules

- Edit only `/app/train.py` and `/app/rewards.py`
- Do NOT modify `/orig/`, or `/models/`
- LoRA adapter must be saved to `/app/output/`
- No external network access
- Single L40S GPU (48GB)
- Time budget: 8 hours
