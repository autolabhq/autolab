# Compute-Optimal Language Model Pretraining

Train a language model from scratch on WikiText-103 to minimize test-set perplexity within a fixed compute budget.

## Setup

| Item | Path / Value |
|------|-------------|
| Framework | LitGPT (`litgpt.GPT` + `litgpt.Config`) |
| Editable file | `/app/train.py` |
| Training wrapper | `/app/train.sh` |
| Local eval | `python3 /app/evaluate_local.py` |
| Training data | `/data/wikitext103/{train,validation}.pt` (~118M tokens) |
| Tokenizer | GPT-2 (`vocab_size=50257`) at `/models/tokenizer` |
| Checkpoint output | `/app/output/checkpoint.pt` |

## Your Goal

Minimize **perplexity** (lower is better) on the WikiText-103 test set. The starting configuration uses a small model with conservative settings. You may change the model architecture, size, training hyperparameters, precision, and any other aspect of `/app/train.py`.

The checkpoint must contain `model_state_dict` and `model_config` dict. The verifier evaluates with fixed `seq_length=1024`, so `block_size` must be at least 1024.

## Evaluation

The verifier loads your checkpoint, reconstructs the model from saved config, and evaluates perplexity on the held-out test set. Reward is computed as a multiplier against the reference score.

```bash
python3 /app/evaluate_local.py
```

## Rules

- Edit only `/app/train.py`
- No external network access
- Single GPU (H100), 8 CPUs, 32GB RAM
- Time budget: 12 hours
- Do not modify tokenizer, evaluation scripts, or test data
