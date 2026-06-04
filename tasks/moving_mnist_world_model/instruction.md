# Moving MNIST World Model — Next-Frame Prediction

Train a video world model from scratch on Moving MNIST to maximize **PSNR** on 10-step rollouts within a 4-hour compute budget.

## Setup

| Item | Path / Value |
|------|-------------|
| Framework | PyTorch (2.4) |
| Editable files | `/app/model.py`, `/app/train.py` |
| Training entrypoint | `bash /app/train.sh` |
| Local eval | `python3 /app/evaluate_local.py` |
| Training data | `/data/moving_mnist/{train,val}.pt` |
| Checkpoint output | `/app/output/checkpoint.pt` |

## Data

Moving MNIST clips, generated deterministically offline. Each clip is **20 frames × 1 × 64 × 64** (grayscale, float32 in `[0, 1]`).

| Split | Clips | Where |
|-------|-------|-------|
| train | 8000 | `/data/moving_mnist/train.pt` |
| val   | 1000 | `/data/moving_mnist/val.pt` |
| test  | 1000 | held out by the verifier — **not** present in your filesystem |

The test split is generated fresh at scoring time from the same pipeline as train/val (different seed) and is not accessible to your code at any point — design and tune against val.

Every clip is split as **10 context frames → 10 target frames**. Your model sees the first 10 frames and must predict the next 10 autoregressively.

## Your Goal

Maximize **PSNR (dB)** averaged over the 10 predicted frames across all 1000 test clips. Higher is better.

## Model Interface

`/app/model.py` must expose:

```python
def build_model(config: dict) -> torch.nn.Module:
    """Return an nn.Module reconstructible from config."""

class YourModel(torch.nn.Module):
    def rollout(self, context: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        context: (B, T_ctx, 1, 64, 64) float32 in [0, 1]
        returns: (B, n_steps, 1, 64, 64) float32 in [0, 1]
        """
```

Save a checkpoint with:
```python
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {...},   # whatever build_model() needs
}, "/app/output/checkpoint.pt")
```

The verifier reconstructs your model via `build_model(model_config)`, loads the state dict, and calls `model.rollout(context, n_steps=10)`.

## Evaluation

```bash
python3 /app/evaluate_local.py   # runs on val set, prints PSNR / MSE / estimated reward
```

Higher PSNR is better.

## Rules

- Edit only `/app/model.py` and `/app/train.py`
- Do NOT modify `/app/evaluate_local.py`, `/data/`, or `/orig/`
- No external network access
- Single GPU (H100), 8 CPUs, 16 GB RAM
- Time budget: 4 hours (training + any exploration)
- The rollout must be autoregressive. Do not condition on future ground-truth frames at eval time
