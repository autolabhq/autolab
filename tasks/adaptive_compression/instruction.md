# Adaptive Compression

Optimize `/app/predictor.py` to minimize the bits-per-byte (bpb) across diverse byte sequences.

## Problem

Given a stream of bytes, predict the probability distribution of each next byte.
Your score is the average cross-entropy:

    bpb = -mean(log2(p[actual_byte]))  over all bytes

Lower bpb means better predictions. Perfect prediction = 0 bpb. Random guessing = 8 bpb.

Implement a `Predictor` class with four methods:

```python
class Predictor:
    def __init__(self): ...
    def reset(self): ...          # Called before each new sequence
    def predict(self) -> list: ...  # Return 256 probabilities for next byte
    def update(self, byte): ...   # Observe the actual next byte (0-255)
```

`predict()` must return 256 non-negative floats that sum to approximately 1.0.

## Data

The evaluation suite contains 9 families of byte sequences with different structures:
Markov chains (order 1 and 2), periodic patterns (clean and noisy), nested bracket
structures, mathematical recurrences, multi-regime switching, run-length encoding,
and incompressible random data.

Visible data is in `/app/data/visible/` (with `manifest.json`).
The data generator `/app/datagen.py` is available for reference.

## Files

| Path | Permission |
|------|-----------|
| `/app/predictor.py` | **Edit** |
| `/app/main.py` | Read-only — local evaluator |
| `/app/datagen.py` | Read-only — data generator |
| `/app/data/visible/` | Read-only — visible test sequences |

## Evaluation

```bash
python3 /app/main.py
```

Reports per-family breakdown and overall bpb on visible sequences.
The verifier evaluates on a hidden suite with the same families but different parameters.

## Rules

- Edit `/app/predictor.py` only. Do not modify other files.
- Python standard library and numpy are available.
- `predictor.py` must be under 100 KB.
- Each sequence must be processed within 60 seconds.
- Time budget: 2 hours.
