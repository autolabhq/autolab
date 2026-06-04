#!/usr/bin/env python3
"""
Correctness verifier for the levenshtein_distance benchmark.

Compiles a small test harness against solve.c and verifies the output
against a pure-Python reference implementation.

Exit 0 -> all tests passed
Exit 1 -> at least one test failed
"""

import subprocess
import sys
import os
import random

APPDIR = "/app"

HARNESS_SRC = r"""
#include <stdio.h>
#include <string.h>
#include "solve.h"

int main(void) {
    char line[256];

    while (fgets(line, sizeof(line), stdin)) {
        char *tab = strchr(line, '\t');
        if (tab == NULL) {
            fprintf(stderr, "Malformed input line\n");
            return 2;
        }

        *tab = '\0';
        char *a = line;
        char *b = tab + 1;
        size_t lb = strlen(b);
        while (lb > 0 && (b[lb - 1] == '\n' || b[lb - 1] == '\r')) {
            b[--lb] = '\0';
        }

        int d = levenshtein(a, (int)strlen(a), b, (int)lb);
        printf("%d\n", d);
    }

    return 0;
}
"""

HARNESS_BIN = "/tmp/lev_verify_harness"


def build_harness():
    src = "/tmp/lev_verify_harness.c"
    with open(src, "w") as f:
        f.write(HARNESS_SRC)
    r = subprocess.run(
        ["gcc", "-O2", "-march=native", "-std=c11",
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


def run_cases(cases):
    data = "".join(f"{a}\t{b}\n" for _desc, a, b in cases)
    r = subprocess.run(
        [HARNESS_BIN],
        input=data, capture_output=True, text=True, timeout=20
    )
    if r.returncode != 0:
        print("Harness run failed:", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)

    lines = r.stdout.splitlines()
    if len(lines) != len(cases):
        print(f"Harness returned {len(lines)} line(s), expected {len(cases)}", file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        sys.exit(1)

    try:
        return [int(x) for x in lines]
    except ValueError:
        print("Harness produced non-integer output:", file=sys.stderr)
        print(r.stdout[:1000], file=sys.stderr)
        sys.exit(1)


def ref_levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return prev[lb]


def make_tests():
    tests = []

    def add(desc, a, b):
        tests.append((desc, a, b))

    # Classic textbook examples
    add("identical single char",        "a",             "a")
    add("single insertion",             "a",             "ab")
    add("single deletion",              "ab",            "a")
    add("single substitution",          "a",             "b")
    add("empty a",                      "",              "hello")
    add("empty b",                      "world",         "")
    add("both empty",                   "",              "")
    add("kitten -> sitting",            "kitten",        "sitting")
    add("sunday -> saturday",           "sunday",        "saturday")
    add("identical word",               "algorithm",     "algorithm")

    # Boundary around benchmark MIN_LEN = 16.
    add("16 chars identical",           "abcdefghijklmnop",  "abcdefghijklmnop")
    add("16 chars one substitution",    "abcdefghijklmnop",  "abcdefghijklmnoq")
    add("16 chars fully different",     "aaaaaaaaaaaaaaaa",  "bbbbbbbbbbbbbbbb")

    # At MAX_LEN = 64
    add("64 chars identical",           "a" * 64,        "a" * 64)
    add("64 chars one substitution",    "a" * 63 + "b",  "a" * 64)
    add("64 chars fully different",     "a" * 64,        "b" * 64)
    add("64 vs 16 chars",               "a" * 64,        "a" * 16)

    # Lowercase DNA-like sequences.
    add("dna 32 chars same",
        "atcgatcgatcgatcgatcgatcgatcgatcg",
        "atcgatcgatcgatcgatcgatcgatcgatcg")
    add("dna 32 shifted by 1",
        "atcgatcgatcgatcgatcgatcgatcgatcg",
        "tcgatcgatcgatcgatcgatcgatcgatcga")
    add("dna 64 half-and-half swap",    "a" * 32 + "t" * 32,
                                           "t" * 32 + "a" * 32)

    # Asymmetric lengths
    add("16 vs 64 chars",               "abcdefghijklmnop",
                                           "abcdefghijklmnop" + "x" * 48)
    add("32 vs 48 chars",               "z" * 32,         "z" * 48)
    add("all deletions (16)",           "abcdefghijklmnop", "")
    add("all insertions (16)",          "",  "zyxwvutsrqponmlk")

    rng = random.Random(0x1EAFD157)
    alphabets = ["a", "ab", "abc", "abcd", "abcdefghijklmnopqrstuvwxyz"]

    # Repeated-character and low-alphabet cases catch many bit-vector edge bugs.
    for n in range(0, 65):
        ch1 = chr(ord("a") + (n % 26))
        ch2 = chr(ord("a") + ((n + 7) % 26))
        add(f"repeat same len {n}", ch1 * n, ch1 * n)
        add(f"repeat different len {n}", ch1 * n, ch2 * n)
        add(f"repeat asymmetric {n}", ch1 * n, ch1 * max(0, 64 - n))

    # Shifted/rotated strings stress insertion/deletion alignment.
    for n in range(1, 65):
        s = "".join(chr(ord("a") + (i % 26)) for i in range(n))
        add(f"shift left {n}", s, s[1:] + s[:1])
        add(f"shift right {n}", s, s[-1:] + s[:-1])

    # Near-identical strings with deterministic edits.
    for i in range(350):
        la = rng.randrange(0, 65)
        s = [rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(la)]
        t = list(s)
        edits = rng.randrange(0, 7)
        for _ in range(edits):
            op = rng.choice(["sub", "ins", "del"])
            if op == "sub" and t:
                p = rng.randrange(len(t))
                t[p] = rng.choice("abcdefghijklmnopqrstuvwxyz")
            elif op == "ins" and len(t) < 64:
                p = rng.randrange(len(t) + 1)
                t.insert(p, rng.choice("abcdefghijklmnopqrstuvwxyz"))
            elif op == "del" and t:
                del t[rng.randrange(len(t))]
        add(f"near identical {i}", "".join(s), "".join(t))

    # Full random hidden cases across lengths 0..64 and mixed alphabets.
    for i in range(1200):
        alphabet = rng.choice(alphabets)
        la = rng.randrange(0, 65)
        lb = rng.randrange(0, 65)
        a = "".join(rng.choice(alphabet) for _ in range(la))
        b = "".join(rng.choice(alphabet) for _ in range(lb))
        add(f"random {i}", a, b)

    return tests


def main():
    build_harness()
    tests = make_tests()
    got_values = run_cases(tests)
    failures = []

    for (desc, a, b), got in zip(tests, got_values):
        expected = ref_levenshtein(a, b)
        if got != expected:
            failures.append((desc, a, b, got, expected))

    if failures:
        for desc, a, b, got, expected in failures[:20]:
            print(f"FAIL: {desc}", file=sys.stderr)
            print(f"      a='{a[:64]}' b='{b[:64]}'", file=sys.stderr)
            print(f"      got={got}  expected={expected}", file=sys.stderr)
        if len(failures) > 20:
            print(f"... plus {len(failures) - 20} more failure(s)", file=sys.stderr)
        print(f"\n{len(failures)}/{len(tests)} test(s) FAILED", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} correctness tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
