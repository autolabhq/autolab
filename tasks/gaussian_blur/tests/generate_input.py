#!/usr/bin/env python3
"""
generate_input.py — create a 4096×4096 synthetic grayscale benchmark image.

The image is written to stdout as raw bytes (one byte per pixel, row-major),
suitable for use as the input to ./blur.

Content: a mix of gradients, concentric rings, and low-frequency noise so
that the Gaussian blur has visually non-trivial, deterministic output and the
checksum is sensitive to correctness errors.
"""

import sys
import math
import random

WIDTH  = 4096
HEIGHT = 4096
SEED   = 0xC0FFEE42

def main():
    rng = random.Random(SEED)
    buf = bytearray(WIDTH * HEIGHT)

    cx = WIDTH  / 2.0
    cy = HEIGHT / 2.0
    max_r = math.hypot(cx, cy)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            # Diagonal linear gradient [0, 255]
            grad = int(255 * (x + y) / (WIDTH + HEIGHT - 2))

            # Concentric rings (period ~64 pixels in radius)
            r = math.hypot(x - cx, y - cy)
            ring = int(127.5 + 127.5 * math.cos(2.0 * math.pi * r / 64.0))

            # Checkerboard (8×8 squares)
            check = 200 if ((x >> 3) ^ (y >> 3)) & 1 else 40

            # Low-frequency Perlin-ish noise approximation
            noise = rng.randint(0, 30)

            # Blend all four components
            val = (grad + ring + check + noise) >> 2
            buf[y * WIDTH + x] = max(0, min(255, val))

    sys.stdout.buffer.write(buf)

if __name__ == "__main__":
    main()
