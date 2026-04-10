# Fredkin Sort Network

Rewrite `circuit.txt` in `/app/` to use as few gates as possible while remaining exactly correct.

## Problem

Implement a 4-input stable sorting network using reversible gates. Four 2-bit unsigned values are provided on input wires `x0..x7`. Output wires `o0..o7` must contain the sorted values in nondecreasing order. Output wires `p0..p4` must contain the swap decisions of the fixed 5-comparator network. Scratch wires `t0..t3` are available. At the end: inputs must be preserved, outputs must be correct, and all scratch wires must be restored to 0. Every gate counts as 1. Lower gate count is better.

## Files

| File | Role |
|------|------|
| `circuit.txt` | **Edit this file only** |
| `main.py` | Simulator/verifier (read-only) |
| `Dockerfile` | Read-only |

## Rules

- Do not modify `Dockerfile` or `main.py`. Build configuration is fixed.
- Edit `/app/circuit.txt` only.
- Use only the provided gate set (x, cx, ccx, fred).
- Output must be exactly correct for all 256 inputs.
- Time budget: 2 hours
