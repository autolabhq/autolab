#!/usr/bin/env python3
"""
verify_correctness.py — correctness verifier for the hash_join task.

Runs the compiled binary in --verify mode on a small dataset
(VERIFY_R_ROWS=200, VERIFY_S_ROWS=2000) and compares its output against
a pure-Python reference nested-loop join on the same input.

Exits 0 on pass, 1 on failure.

Usage (called by tests/test.sh after a successful build):
    python3 /tests/verify_correctness.py
"""

import sys
import os
import subprocess

# ── Constants (must match solve.h) ───────────────────────────────────────────

BINARY        = "/app/join" if os.path.isfile("/app/join") else \
                os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), "join")
KEY_RANGE     = 50000
VERIFY_R_ROWS = 200
VERIFY_S_ROWS = 2000

# ── xorshift64 PRNG (must match main.c) ──────────────────────────────────────

def make_rng(seed):
    """Returns a stateful generator matching the C xorshift64."""
    state = [seed & 0xFFFFFFFFFFFFFFFF]

    def next_val():
        s = state[0]
        s ^= (s << 13) & 0xFFFFFFFFFFFFFFFF
        s ^= (s >> 7)
        s ^= (s << 17) & 0xFFFFFFFFFFFFFFFF
        state[0] = s & 0xFFFFFFFFFFFFFFFF
        return (s >> 32) & 0xFFFFFFFF

    return next_val


def gen_table(n, key_range, seed):
    """Generate n rows as list of (key, payload) using the same PRNG as main.c."""
    rng = make_rng(seed)
    rows = []
    for _ in range(n):
        key     = rng() % key_range
        payload = rng()
        # payload is uint32; reinterpret as int32 to match C's int32_t cast
        if payload >= 0x80000000:
            payload -= 0x100000000
        rows.append((key, payload))
    return rows


# ── Reference join ────────────────────────────────────────────────────────────

def reference_join(r_rows, s_rows):
    """
    Pure-Python inner equi-join.  Returns (match_count, checksum) where
    checksum = SUM of ((uint32_t)r.payload + (uint32_t)s.payload) mod 2^64,
    matching the C uint64_t accumulation in solve.c.
    """
    # Build index from key -> list of payloads in R (for correctness only; OK to be slow)
    from collections import defaultdict
    r_index = defaultdict(list)
    for key, payload in r_rows:
        r_index[key].append(payload)

    count    = 0
    checksum = 0
    MASK64   = 0xFFFFFFFFFFFFFFFF

    for s_key, s_payload in s_rows:
        for r_payload in r_index.get(s_key, []):
            count += 1
            # match solve.c: (uint32_t)r.payload + (uint32_t)s.payload, widened
            checksum = (checksum + (r_payload & 0xFFFFFFFF)
                        + (s_payload & 0xFFFFFFFF)) & MASK64

    return count, checksum


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(BINARY):
        print(f"ERROR: binary not found at {BINARY}", flush=True)
        print("Build with 'make' in /app first.", flush=True)
        sys.exit(1)

    # ── Compute reference ────────────────────────────────────────────────────
    print(f"Generating reference tables: R={VERIFY_R_ROWS}, S={VERIFY_S_ROWS} …",
          flush=True)
    r_rows = gen_table(VERIFY_R_ROWS, KEY_RANGE, 0xDEADBEEF12345678)
    s_rows = gen_table(VERIFY_S_ROWS, KEY_RANGE, 0xCAFEBABE87654321)

    print("Computing reference join …", flush=True)
    ref_count, ref_checksum = reference_join(r_rows, s_rows)
    print(f"Reference: matches={ref_count}  checksum={ref_checksum}", flush=True)

    # ── Run submission binary ─────────────────────────────────────────────────
    try:
        result = subprocess.run(
            [BINARY, "--verify"],
            capture_output=True, text=True, timeout=60
        )
    except subprocess.TimeoutExpired:
        print("ERROR: binary timed out during correctness check", flush=True)
        sys.exit(1)

    if result.returncode != 0:
        print(f"ERROR: binary exited with code {result.returncode}", flush=True)
        print(result.stderr, flush=True)
        sys.exit(1)

    # ── Parse output ─────────────────────────────────────────────────────────
    # Expected format: "verify matches=<N> checksum=<C>"
    line = result.stdout.strip()
    print(f"Binary output: {line}", flush=True)

    try:
        parts = dict(tok.split("=") for tok in line.split() if "=" in tok)
        sub_count    = int(parts["matches"])
        sub_checksum = int(parts["checksum"])
    except (KeyError, ValueError) as e:
        print(f"ERROR: could not parse binary output: {e}", flush=True)
        sys.exit(1)

    # ── Compare ───────────────────────────────────────────────────────────────
    ok = True

    if sub_count != ref_count:
        print(f"FAIL: match_count mismatch: got {sub_count}, expected {ref_count}",
              flush=True)
        ok = False

    if sub_checksum != ref_checksum:
        print(f"FAIL: checksum mismatch: got {sub_checksum}, expected {ref_checksum}",
              flush=True)
        ok = False

    if ok:
        print(f"PASS: matches={sub_count}, checksum={sub_checksum}", flush=True)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
