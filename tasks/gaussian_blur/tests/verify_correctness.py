#!/usr/bin/env python3
"""
verify_correctness.py — correctness verifier for the gaussian_blur task.

Runs the compiled binary on a small 256×256 test image and compares its
output pixel-by-pixel against a reference separable-Gaussian implementation.
Exits 0 if max(|submission − reference|) ≤ TOLERANCE, else exits 1.

Usage (called by tests/test.sh after a successful build):
    python3 /tests/verify_correctness.py
"""

import sys
import os
import math
import random
import subprocess
import tempfile

# ── Constants ────────────────────────────────────────────────────────────────

BINARY    = "/app/blur" if os.path.isfile("/app/blur") else \
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blur")
WIDTH     = 256
HEIGHT    = 256
PASSES    = 5          # must match BLUR_PASSES in solve.h
SIGMA     = 3.0
RADIUS    = 8          # must match KERNEL_RADIUS in solve.h
TOLERANCE = 4          # must match CORRECTNESS_TOLERANCE in solve.h
SEED      = 0xDEADBEEF

# ── Reference implementation (pure Python, separable, clamp-to-edge) ─────────

def build_1d_kernel(sigma=SIGMA, radius=RADIUS):
    """Normalised 1-D Gaussian kernel of length 2*radius+1."""
    k = [math.exp(-(d * d) / (2.0 * sigma * sigma))
         for d in range(-radius, radius + 1)]
    total = sum(k)
    return [v / total for v in k]


def h_pass(img, w, h, k, r):
    """Horizontal 1-D blur with clamp-to-edge border handling."""
    out = [0.0] * (w * h)
    for y in range(h):
        row_off = y * w
        for x in range(w):
            acc = 0.0
            for i, kv in enumerate(k):
                sx = max(0, min(w - 1, x + i - r))
                acc += kv * img[row_off + sx]
            out[row_off + x] = acc
    return out


def v_pass(img, w, h, k, r):
    """Vertical 1-D blur with clamp-to-edge border handling."""
    out = [0.0] * (w * h)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for i, kv in enumerate(k):
                sy = max(0, min(h - 1, y + i - r))
                acc += kv * img[sy * w + x]
            out[y * w + x] = acc
    return out


def gaussian_blur_ref(pixels, w, h, passes=PASSES):
    """
    Reference: separable Gaussian blur applied `passes` times.
    Returns a list of int values in [0, 255].
    """
    k = build_1d_kernel()
    img = [float(p) for p in pixels]
    for _ in range(passes):
        img = h_pass(img, w, h, k, RADIUS)
        img = v_pass(img, w, h, k, RADIUS)
    return [min(255, max(0, int(v + 0.5))) for v in img]


# ── Test image ────────────────────────────────────────────────────────────────

def make_test_image(w, h, seed=SEED):
    """
    Structured 256×256 test image: gradient + rings + checkerboard + noise.
    Deterministic — same seed always produces the same image.
    """
    rng = random.Random(seed)
    cx, cy = w / 2.0, h / 2.0
    pixels = []
    for y in range(h):
        for x in range(w):
            grad  = int(255 * (x + y) / (w + h - 2))
            r     = math.hypot(x - cx, y - cy)
            ring  = int(127.5 + 127.5 * math.cos(2.0 * math.pi * r / 32.0))
            check = 200 if ((x >> 3) ^ (y >> 3)) & 1 else 40
            noise = rng.randint(0, 20)
            val   = (grad + ring + check + noise) >> 2
            pixels.append(max(0, min(255, val)))
    return pixels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Check binary exists
    if not os.path.isfile(BINARY):
        print(f"ERROR: binary not found at {BINARY}", flush=True)
        print("Build with 'make' in /app first.", flush=True)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "test_input.raw")
        output_path = os.path.join(tmpdir, "test_output.raw")

        # ── Write test image ─────────────────────────────────────────────
        pixels = make_test_image(WIDTH, HEIGHT)
        with open(input_path, "wb") as f:
            f.write(bytes(pixels))

        # ── Run submission binary ────────────────────────────────────────
        cmd = [BINARY, input_path, output_path,
               str(WIDTH), str(HEIGHT)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=120)
        except subprocess.TimeoutExpired:
            print("ERROR: binary timed out during correctness test", flush=True)
            sys.exit(1)

        if result.returncode != 0:
            print(f"ERROR: binary exited with code {result.returncode}",
                  flush=True)
            print(result.stderr, flush=True)
            sys.exit(1)

        # ── Read submission output ───────────────────────────────────────
        try:
            with open(output_path, "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            print("ERROR: binary did not write output file", flush=True)
            sys.exit(1)

        if len(raw) != WIDTH * HEIGHT:
            print(f"ERROR: output size {len(raw)} ≠ expected {WIDTH * HEIGHT}",
                  flush=True)
            sys.exit(1)

        submission = list(raw)

        # ── Compute reference ────────────────────────────────────────────
        print(f"Computing reference for {WIDTH}×{HEIGHT} image, "
              f"{PASSES} passes …", flush=True)
        reference = gaussian_blur_ref(pixels, WIDTH, HEIGHT, PASSES)

        # ── Compare ──────────────────────────────────────────────────────
        max_diff  = 0
        bad_count = 0
        first_bad = None

        for i, (sub, ref) in enumerate(zip(submission, reference)):
            diff = abs(sub - ref)
            if diff > max_diff:
                max_diff = diff
            if diff > TOLERANCE:
                bad_count += 1
                if first_bad is None:
                    y, x = divmod(i, WIDTH)
                    first_bad = (y, x, sub, ref, diff)

        print(f"Max pixel deviation: {max_diff}  (tolerance: {TOLERANCE})",
              flush=True)

        if bad_count > 0:
            y, x, sub, ref, diff = first_bad
            print(f"FAIL: {bad_count} pixel(s) exceed tolerance.", flush=True)
            print(f"  First bad pixel at ({y},{x}): "
                  f"got {sub}, expected {ref}, diff {diff}", flush=True)
            sys.exit(1)

        print(f"PASS: all {WIDTH * HEIGHT} pixels within tolerance.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
