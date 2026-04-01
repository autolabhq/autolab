# Data Selection for Instruction Following

Select training samples from a 50,000-sample data pool to maximize instruction-following performance after LoRA fine-tuning on Qwen2.5-3B-Instruct.

## Setup

| Item | Path / Value |
|------|-------------|
| Base model | `/models/Qwen2.5-3B-Instruct` |
| Data pool | `/data/pool.json` (~50k samples, 19 sources) |
| Pool metadata | `/data/pool_metadata.json` |
| Editable file | `/app/select_data.py` |
| Training pipeline | `bash /app/train.sh` |
| Local eval | `python3 /app/evaluate_local.py` |

## Your Goal

Maximize `ifeval_prompt_strict` — the prompt-level strict accuracy on the IFEval benchmark (~541 prompts testing format constraints, length requirements, keyword inclusion, etc.).

Write `/app/select_data.py` to select at most 5,000 samples from the pool. The current script uses random selection.

## Evaluation

The verifier runs your selection + fixed training pipeline, then evaluates on the full IFEval benchmark. Your score is compared against a baseline (random selection).

```bash
bash /app/train.sh                        # runs select → convert → train
python3 /app/evaluate_local.py             # quick eval (~50 prompts)
python3 /app/evaluate_local.py --limit 100 # more prompts
```

## Key Information

- The data pool contains 19 sources: instruction-following, math, code, multilingual, safety, and general conversation data
- Each sample has `id`, `messages` (user/assistant turns), and `source` fields
- The base model is already instruction-tuned — not all training data is equally helpful
- The training pipeline (config, hyperparameters, conversion) is fixed; only the data selection changes

## Rules

- Edit only `/app/select_data.py`
- Do NOT modify `train_config.yaml`, `train.sh`, `convert_to_sharegpt.py`
- Do NOT modify files under `/orig/`, or `/models/`
- Maximum 5,000 selected samples
- No external network access
- Single GPU
- Time budget: 8 hours
