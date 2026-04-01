#!/usr/bin/env python3
"""
Correctness verifier for the sha256_throughput benchmark.

Compiles a small test harness against solve.c and tests SHA-256 output
against Python's hashlib.sha256 reference implementation.

Exit 0 -> all tests passed
Exit 1 -> at least one test failed
"""

import subprocess
import sys
import os
import hashlib
import struct

APPDIR = "/app"

# ── Test harness C source ─────────────────────────────────────────────────────
# Reads raw bytes from stdin, computes SHA-256, prints hex digest to stdout.
HARNESS_SRC = r"""
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "solve.h"

int main(void) {
    /* Read all of stdin into a dynamically-grown buffer. */
    size_t cap = 4096, len = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return 1;
    int c;
    while ((c = fgetc(stdin)) != EOF) {
        if (len == cap) {
            cap *= 2;
            buf = (uint8_t *)realloc(buf, cap);
            if (!buf) return 1;
        }
        buf[len++] = (uint8_t)c;
    }
    uint8_t digest[32];
    sha256(buf, len, digest);
    for (int i = 0; i < 32; i++)
        printf("%02x", digest[i]);
    printf("\n");
    free(buf);
    return 0;
}
"""

HARNESS_BIN = "/tmp/sha256_verify_harness"


def build_harness():
    src = "/tmp/sha256_verify_harness.c"
    with open(src, "w") as f:
        f.write(HARNESS_SRC)
    r = subprocess.run(
        ["gcc", "-O2", "-march=native", "-std=c99",
         "-I", APPDIR,
         "-o", HARNESS_BIN,
         src,
         os.path.join(APPDIR, "solve.c")],
        capture_output=True, text=True
    )
    try:
        os.unlink(src)
    except OSError:
        pass
    if r.returncode != 0:
        print("Harness build failed:", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)


def run_case(data: bytes) -> str:
    """Run the harness on the given bytes; return the hex digest string."""
    r = subprocess.run(
        [HARNESS_BIN],
        input=data,
        capture_output=True,
        timeout=30
    )
    if r.returncode != 0:
        return None
    return r.stdout.decode().strip()


def expected(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Test cases ────────────────────────────────────────────────────────────────

def make_tests():
    tests = []

    # FIPS 180-4 / NIST standard test vectors
    tests.append(("NIST: empty string",          b""))
    tests.append(("NIST: 'abc'",                 b"abc"))
    tests.append(("NIST: 448-bit message",
                  b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))

    # Boundary cases around SHA-256 padding blocks
    tests.append(("55 bytes (fills first padding block exactly)",
                  b"a" * 55))
    tests.append(("56 bytes (needs second padding block)",
                  b"a" * 56))
    tests.append(("63 bytes (one byte short of full block)",
                  bytes(range(63))))
    tests.append(("64 bytes (exactly one full block)",
                  b"a" * 64))
    tests.append(("64 null bytes (boundary, one block)",
                  b"\x00" * 64))
    tests.append(("65 bytes (one byte into second block)",
                  bytes(range(65))))
    tests.append(("127 bytes",
                  bytes(k % 256 for k in range(127))))
    tests.append(("128 bytes (two full blocks)",
                  b"a" * 128))
    tests.append(("191 bytes",
                  bytes(k % 256 for k in range(191))))
    tests.append(("192 bytes (three full blocks)",
                  bytes(k % 256 for k in range(192))))

    # Larger inputs
    tests.append(("1000 bytes (arbitrary multi-block)",
                  bytes(k % 251 for k in range(1000))))
    tests.append(("4096 bytes",
                  bytes(k % 256 for k in range(4096))))

    # Non-trivial content
    tests.append(("512 bytes of alternating 0xAA/0x55",
                  bytes([0xAA if i % 2 == 0 else 0x55 for i in range(512)])))
    tests.append(("All 0xFF bytes, 64 bytes",
                  b"\xFF" * 64))
    tests.append(("Single 0x80 byte",
                  b"\x80"))
    tests.append(("Single 0x00 byte",
                  b"\x00"))

    return tests


def main():
    build_harness()

    tests = make_tests()
    failures = 0

    for desc, data in tests:
        ref = expected(data)
        got = run_case(data)
        if got is None:
            print(f"FAIL: {desc}  [harness error]", file=sys.stderr)
            failures += 1
        elif got == ref:
            print(f"OK:   {desc}")
        else:
            print(f"FAIL: {desc}", file=sys.stderr)
            print(f"      got:      {got}", file=sys.stderr)
            print(f"      expected: {ref}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"\n{failures}/{len(tests)} test(s) FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} correctness tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
