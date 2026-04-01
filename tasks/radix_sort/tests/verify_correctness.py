#!/usr/bin/env python3
"""
Correctness verifier for the radix_sort benchmark.

Compiles a small test harness against solve.c and runs radix_sort on
several hand-crafted inputs, comparing output against Python's sorted().

Exit 0 → all tests passed
Exit 1 → at least one test failed
"""

import subprocess
import sys
import os
import tempfile

APPDIR = "/app"

# ── Test harness C source ─────────────────────────────────────────────────────
HARNESS_SRC = r"""
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "solve.h"

int main(void) {
    uint32_t n;
    while (scanf("%u", &n) == 1) {
        uint32_t *arr = (uint32_t *)malloc((n + 1) * sizeof(uint32_t));
        for (uint32_t i = 0; i < n; i++) scanf("%u", &arr[i]);
        radix_sort(arr, n);
        printf("%u", n);
        for (uint32_t i = 0; i < n; i++) printf(" %u", arr[i]);
        printf("\n");
        free(arr);
    }
    return 0;
}
"""

HARNESS_BIN = "/tmp/rs_verify_harness"


def build_harness():
    src = "/tmp/rs_harness.c"
    with open(src, "w") as f:
        f.write(HARNESS_SRC)
    r = subprocess.run(
        ["gcc", "-O2", "-std=c99",
         "-I", APPDIR,
         "-o", HARNESS_BIN,
         src,
         os.path.join(APPDIR, "solve.c")],
        capture_output=True, text=True
    )
    os.unlink(src)
    if r.returncode != 0:
        print("Harness build failed:", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)


def run_case(arr):
    """Run radix_sort on arr via the harness. Returns sorted list."""
    inp = f"{len(arr)}" + "".join(f" {x}" for x in arr) + "\n"
    r = subprocess.run([HARNESS_BIN], input=inp,
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    parts = r.stdout.strip().split()
    if not parts:
        return None
    n = int(parts[0])
    return [int(x) for x in parts[1:n+1]]


# ── Test cases ────────────────────────────────────────────────────────────────
import random
random.seed(42)


def make_test(desc, arr):
    return (desc, arr, sorted(arr))


TESTS = [
    make_test("empty array n=0",                      []),
    make_test("single element",                       [42]),
    make_test("two elements sorted",                  [1, 2]),
    make_test("two elements reverse",                 [2, 1]),
    make_test("all equal",                            [7] * 10),
    make_test("already sorted 1..10",                 list(range(1, 11))),
    make_test("reverse sorted 10..1",                 list(range(10, 0, -1))),
    make_test("max uint32 values",                    [0xFFFFFFFF, 0, 0xFFFFFFFF, 1]),
    make_test("powers of two",                        [1 << k for k in range(0, 32)]),
    make_test("boundary: 0 and UINT32_MAX mixed",     [0xFFFFFFFF, 0, 0x80000000, 1, 0x7FFFFFFF]),
    make_test("random 100 elements",                  [random.randint(0, 0xFFFFFFFF) for _ in range(100)]),
    make_test("random 1000 elements",                 [random.randint(0, 0xFFFFFFFF) for _ in range(1000)]),
    make_test("random 10000 elements",                [random.randint(0, 0xFFFFFFFF) for _ in range(10000)]),
    make_test("random 100000 elements",               [random.randint(0, 0xFFFFFFFF) for _ in range(100000)]),
    make_test("many duplicates 5 distinct values",    [random.choice([0,1,2,3,4]) for _ in range(1000)]),
    make_test("all zeros",                            [0] * 100),
    make_test("lo-16 stress (hi-16 all same)",        [0x00010000 | (i % 65536) for i in range(1000)]),
    make_test("hi-16 stress (lo-16 all same)",        [(i % 65536) << 16 for i in range(1000)]),
]


def main():
    build_harness()

    failures = 0
    for desc, arr, expected in TESTS:
        if len(arr) == 0:
            # Empty array: harness should return 0 elements
            got = run_case([])
            if got == [] or got is None:
                print(f"OK:   {desc}  []")
            else:
                print(f"FAIL: {desc}  got={got}", file=sys.stderr)
                failures += 1
            continue

        got = run_case(arr)
        if got is None:
            print(f"FAIL: {desc}  [harness error]", file=sys.stderr)
            failures += 1
        elif got == expected:
            print(f"OK:   {desc}  (n={len(arr)}, first={got[0]}, last={got[-1]})")
        else:
            # Find first mismatch
            for i, (g, e) in enumerate(zip(got, expected)):
                if g != e:
                    print(f"FAIL: {desc}  first diff at [{i}]: got {g}, expected {e}",
                          file=sys.stderr)
                    break
            else:
                print(f"FAIL: {desc}  length mismatch got={len(got)} expected={len(expected)}",
                      file=sys.stderr)
            failures += 1

    if failures:
        print(f"\n{failures}/{len(TESTS)} test(s) FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(TESTS)} correctness tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
