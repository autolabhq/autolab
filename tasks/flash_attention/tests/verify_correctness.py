#!/usr/bin/env python3
"""
verify_correctness.py — correctness verifier for flash_attention.

Runs the compiled binary with --verify on a small input (N=256, D=32) and
compares the sum of all output values against a reference computed in Python
using the identical LCG and naive double-precision attention.

Exits 0 on pass, 1 on failure.

Usage (called by tests/test.sh after a successful build):
    python3 /tests/verify_correctness.py
"""

import sys
import os
import subprocess
import math

BINARY   = "/app/attn" if os.path.isfile("/app/attn") else \
           os.path.join(os.path.dirname(os.path.dirname(
               os.path.abspath(__file__))), "attn")

VFY_N    = 256
VFY_D    = 32
VFY_SEED = 0xBEEFDEAD42
# 1 % relative tolerance: tight enough to catch wrong implementations,
# loose enough to accept float32 accumulation error.
TOL      = 1e-2

# ── LCG (must match main.c exactly) ──────────────────────────────────────────

MULT64 = 6364136223846793005
INC64  = 1442695040888963407
MASK64 = (1 << 64) - 1


def make_lcg(seed):
    """Returns a stateful LCG float generator matching the C implementation."""
    state = [seed & MASK64]

    def lcg_float():
        state[0] = (state[0] * MULT64 + INC64) & MASK64
        raw = ((state[0] >> 11) & ((1 << 53) - 1)) / (1 << 53)
        return (raw - 0.5) * 0.2   # maps to [-0.1, 0.1]

    return lcg_float


# ── Reference attention (numpy preferred, pure-Python fallback) ───────────────

def reference_sum(Q_flat, K_flat, V_flat, n, d):
    """
    Computes sum(attention(Q, K, V)) using double-precision numpy.
    Falls back to pure Python if numpy is absent.
    """
    try:
        import numpy as np
        Q = np.array(Q_flat, dtype=np.float64).reshape(n, d)
        K = np.array(K_flat, dtype=np.float64).reshape(n, d)
        V = np.array(V_flat, dtype=np.float64).reshape(n, d)
        scale = 1.0 / math.sqrt(d)
        S = (Q @ K.T) * scale                      # (n, n)
        S -= S.max(axis=1, keepdims=True)           # numerical stability
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True)          # row-wise softmax
        out = P @ V                                 # (n, d)
        return float(out.sum())

    except ImportError:
        # Pure-Python fallback — correct but slow (fine for N=256, D=32).
        scale = 1.0 / math.sqrt(d)
        total = 0.0
        for i in range(n):
            qi = Q_flat[i * d:(i + 1) * d]
            s  = []
            for j in range(n):
                kj  = K_flat[j * d:(j + 1) * d]
                dot = sum(qi[k] * kj[k] for k in range(d)) * scale
                s.append(dot)
            mx  = max(s)
            e   = [math.exp(x - mx) for x in s]
            inv = 1.0 / sum(e)
            p   = [x * inv for x in e]
            for k in range(d):
                total += sum(p[j] * V_flat[j * d + k] for j in range(n))
        return total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(BINARY):
        print(f"ERROR: binary not found at {BINARY}", flush=True)
        print("Build with 'make' in /app first.", flush=True)
        sys.exit(1)

    # ── Generate reference input using the same LCG as main.c ────────────────
    print(f"Generating reference input: N={VFY_N}, D={VFY_D} …", flush=True)
    lcg = make_lcg(VFY_SEED)
    nd  = VFY_N * VFY_D
    Q_flat = [lcg() for _ in range(nd)]
    K_flat = [lcg() for _ in range(nd)]
    V_flat = [lcg() for _ in range(nd)]

    print("Computing reference output sum …", flush=True)
    ref_sum = reference_sum(Q_flat, K_flat, V_flat, VFY_N, VFY_D)
    print(f"Reference sum: {ref_sum:.10e}", flush=True)

    # ── Run submission binary in verify mode ──────────────────────────────────
    try:
        result = subprocess.run(
            [BINARY, "--verify"],
            capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        print("ERROR: binary timed out during correctness check", flush=True)
        sys.exit(1)

    if result.returncode != 0:
        print(f"ERROR: binary exited with code {result.returncode}", flush=True)
        print(result.stderr, flush=True)
        sys.exit(1)

    # ── Parse output ─────────────────────────────────────────────────────────
    line = result.stdout.strip()
    print(f"Binary output: {line}", flush=True)

    try:
        parts = dict(tok.split("=") for tok in line.split() if "=" in tok)
        sub_sum = float(parts["sum"])
    except (KeyError, ValueError) as e:
        print(f"ERROR: could not parse binary output '{line}': {e}", flush=True)
        sys.exit(1)

    # ── Compare ───────────────────────────────────────────────────────────────
    denom   = max(abs(ref_sum), 1e-12)
    rel_err = abs(sub_sum - ref_sum) / denom

    print(f"Submission sum:  {sub_sum:.10e}", flush=True)
    print(f"Relative error:  {rel_err:.2e}  (tolerance {TOL:.0e})", flush=True)

    if rel_err <= TOL:
        print("PASS", flush=True)
        sys.exit(0)
    else:
        print(f"FAIL: relative error {rel_err:.2e} exceeds tolerance {TOL:.0e}",
              flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
