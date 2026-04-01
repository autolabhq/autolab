#!/usr/bin/env python3
"""
verify_correctness.py — correctness gate for fft_rust.

Cross-checks the Rust binary's FFT output against a Python reference
on small inputs (N = 8, 64, 512, 4096).  Small inputs run fast even with
the naive O(n²) baseline, so this check always completes in < 30 s.

The binary accepts a VERIFY_N environment variable to override the default
N_DEFAULT = 32768, enabling small-input testing.

DO NOT MODIFY THIS FILE.
"""

import subprocess
import re
import sys
import os
import math

BINARY = "/app/target/release/solve"


def python_fft_checksum(n: int) -> float:
    """
    Compute the spectral checksum that src/main.rs emits for gen_signal(n).

    We use the same gen_signal() formula and DFT definition, then compute
    the weighted power sum Σ_k |X[k]|² * k, rounded to 6 significant figures.
    """
    tau = 2.0 * math.pi
    FREQS = [
        (1,   1.0,     0.0),
        (3,   0.5,     0.5),
        (7,   0.25,    1.0),
        (13,  0.125,   1.5),
        (31,  0.0625,  0.3),
        (63,  0.03125, 0.7),
        (127, 0.015625,1.1),
    ]

    signal = [
        sum(a * math.sin(tau * (f * j) / n + phi) for f, a, phi in FREQS)
        for j in range(n)
    ]

    # Direct DFT (only feasible for small n)
    checksum = 0.0
    for k in range(n):
        re = sum(signal[j] * math.cos(-2 * math.pi * k * j / n) for j in range(n))
        im = sum(signal[j] * math.sin(-2 * math.pi * k * j / n) for j in range(n))
        checksum += (re * re + im * im) * k

    # Round to 6 significant figures (matching Rust: (checksum * 1e-6).round() * 1e6)
    if checksum == 0.0:
        return 0.0
    magnitude = math.floor(math.log10(abs(checksum)))
    factor = 10 ** (magnitude - 5)
    return round(checksum / factor) * factor


def run_case(label: str, n: int) -> bool:
    """Run the binary with VERIFY_N=n and check verify=ok plus checksum."""
    env = os.environ.copy()
    env["VERIFY_N"] = str(n)

    try:
        result = subprocess.run(
            [BINARY],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"FAIL [{label}]: binary timed out after 120 s", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"FAIL [{label}]: binary not found: {BINARY}", file=sys.stderr)
        return False

    if result.returncode != 0:
        print(
            f"FAIL [{label}]: binary exited with code {result.returncode}\n"
            f"  stderr: {result.stderr[:300]}",
            file=sys.stderr,
        )
        return False

    output = result.stdout.strip()

    if "verify=ok" not in output:
        print(
            f"FAIL [{label}]: verify≠ok — output: {output!r}",
            file=sys.stderr,
        )
        return False

    # For small n where Python reference DFT is tractable, also check checksum.
    if n <= 512:
        m = re.search(r"checksum=([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", output)
        if not m:
            print(f"FAIL [{label}]: no checksum in output: {output!r}", file=sys.stderr)
            return False

        got_cs = float(m.group(1))
        ref_cs = python_fft_checksum(n)

        # Relative tolerance 1e-3 (Python DFT has accumulated float error)
        if ref_cs != 0.0:
            rel_err = abs(got_cs - ref_cs) / abs(ref_cs)
        else:
            rel_err = abs(got_cs)

        if rel_err > 1e-3:
            print(
                f"FAIL [{label}]: checksum mismatch\n"
                f"  got      {got_cs:.6e}\n"
                f"  expected {ref_cs:.6e}\n"
                f"  rel_err  {rel_err:.2e}",
                file=sys.stderr,
            )
            return False

    print(f"OK   [{label}]: verify=ok ✓")
    return True


# ── Test cases ────────────────────────────────────────────────────────────────
#
# Sizes chosen so that even the naive O(n²) baseline finishes in < 30 s total:
#   n=8:    trivial smoke test
#   n=64:   exercises butterfly stages
#   n=512:  exercises bit-reversal and multi-stage butterflies
#   n=4096: exercises twiddle-factor correctness; ~1 s for naive baseline
#
CASES = [
    ("n=8",    8),
    ("n=64",   64),
    ("n=512",  512),
    ("n=4096", 4096),
]

if not os.path.exists(BINARY):
    print(f"FAIL: binary not found at {BINARY} (build step failed?)", file=sys.stderr)
    sys.exit(1)

results = [run_case(label, n) for label, n in CASES]

if all(results):
    print("All correctness checks passed.")
    sys.exit(0)
else:
    sys.exit(1)
