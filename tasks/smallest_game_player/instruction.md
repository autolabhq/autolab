# Smallest Game Player

Optimize `/app/solve.py` to use as **few learnable parameters as possible** while still achieving ≥95% accuracy on a hidden test set of perfect-play positions from 4×4 gravity Connect-3.

## Problem

The dataset contains legal non-terminal board positions and the optimal move chosen by an exact minimax solver:

| File | Contents |
|------|----------|
| `/app/X_train.npy` | `(N, 17)` int8 state matrix |
| `/app/y_train.npy` | `(N,)` int64 optimal column in `[0, 3]` |

State encoding: `X[:, :16]` is the board flattened row-major (bottom to top), cell values `0=empty`, `1=X`, `2=O`; `X[:, 16]` is side to move.

Implement two functions in `/app/solve.py`:

```python
def train(X_train, y_train):
    """Train model. Return a flat dict whose values are numpy arrays."""

def predict(model, X):
    """Return (N,) int64 array of predicted columns in [0, 3]."""
```

Parameter count is `sum(v.size for v in model.values())`. If hidden-test accuracy < 97%, reward is 0.

## Files

| File | Role |
|------|------|
| `solve.py` | **Edit this file only** |
| `X_train.npy`, `y_train.npy` | Training data (read-only) |

## Rules

- Edit `/app/solve.py` only.
- Allowed imports: `numpy`, `torch`. No other libraries.
- `train()` and `predict()` signatures must not change; `train()` must return a flat `dict[str, np.ndarray]`.
- No explicit game-tree search, minimax, alpha-beta, or hardcoded state lookup tables.
- No internet access.
