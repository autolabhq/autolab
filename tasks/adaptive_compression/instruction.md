# Adaptive Compression

Optimize `/app/predictor.py` to minimize the bits-per-byte (bpb) across diverse byte sequences.

## Problem

Given a stream of bytes, predict the probability distribution of each next byte.
Your score is the average cross-entropy:

    bpb = -mean(log2(p[actual_byte]))  over all bytes

Lower bpb is better. Perfect prediction = 0 bpb. Uniform prediction = 8 bpb.

Implement a `Predictor` class with four methods:

```python
class Predictor:
    def __init__(self): ...
    def reset(self): ...            # Called before each new sequence
    def predict(self) -> list: ...  # Return 256 probabilities for next byte
    def update(self, byte): ...     # Observe the actual next byte (0–255)
```

`predict()` must return 256 non-negative floats in `[0, 1]` that sum to ≈ 1.0.
**Every** call is validated; an invalid distribution fails the entire sequence
(scored as 8.0 bpb, equivalent to uniform).

## Data

The evaluation suite contains 9 families of byte sequences with different
structures: Markov chains (order 1 and 2), periodic patterns (clean and noisy),
nested bracket structures, mathematical recurrences, multi-regime switching,
run-length encoding, and incompressible random data. Hidden evaluation uses
the same families with a different runtime-random seed and a randomized
order, so per-family identity cannot be inferred from position.

Visible data lives in `/app/data/visible/` (with `manifest.json`). Use it
freely for development.

## Files

| Path | Permission |
|------|-----------|
| `/app/predictor.py` | **Edit** |
| `/app/main.py`      | Read-only — local evaluator |
| `/app/data/visible/`| Read-only — visible test sequences |

## Evaluation

```bash
python3 /app/main.py                      # full suite
python3 /app/main.py --family periodic_clean,markov_order2   # subset
```

Reports per-family bpb and the byte-weighted overall score. Failed sequences
count as 8.0 bpb — exactly as the verifier scores them.

## Rules

- Edit `/app/predictor.py` only.
- Python standard library and numpy are available.
- `predictor.py` must be under 100 KB.
- Each sequence must be processed within 60 seconds.
- Time budget: 4 hours.
