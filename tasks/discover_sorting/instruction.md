# Discover Sorting

Optimize `/app/solve.py` to generate a correct **16-input sorting network** with as **few comparators as possible**.

## Problem

A sorting network is a fixed sequence of compare-exchange operations over wires `0..15`. Each comparator `(i, j)` swaps `wires[i]` and `wires[j]` if `wires[i] > wires[j]`.

Implement:

```python
def generate_network(n: int = 16) -> list[tuple[int, int]]:
    ...
```

The verifier calls `generate_network(16)`, runs the network on all 65 536 binary inputs, and checks that all outputs are sorted. By the 0-1 principle this proves correctness for all totally ordered inputs.

## Files

| File | Role |
|------|------|
| `solve.py` | **Edit this file only** |
| `main.py` | Local verifier (read-only) |

## Rules

- Edit `/app/solve.py` only. Python standard library only.
- `generate_network()` must return a `list` of 2-tuples with `0 <= i, j < 16` and `i != j`.
- Incorrect networks score 0.
