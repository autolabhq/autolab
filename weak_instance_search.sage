#!/usr/bin/env sage
"""Exhaustive weakness check on the 4 benchmark LMFDB curves.

For each (curve, p) target in the benchmark:
  1. trace t = p + 1 - n; is t = 1 (anomalous)?
  2. embedding degree k: smallest k with n | p^k - 1
  3. p - 1 factorization: is the smooth part useful for Petit-Quisquater?
  4. twist order: is twist order smooth (invalid-curve attack)?
  5. CM detection: does E have endomorphism beyond [n]?
  6. j-invariant: is j ∈ {0, 1728}?

Any of these would unlock a specific attack.
"""
import time
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, factor, gcd

print("Weakness search on benchmark LMFDB curves")
print("=" * 78)

# Hardcoded from lmfdb_curves.json benchmark
TARGETS = [
    ("67.a1",      [0,  1,  1, -12, -21],         1208925819614629469615699, 1208925819616653179121727, 1208925819612605760109673),
    ("21175.bc1",  None,                          1208925819614629639728787, 1208925819612684087177953, None),
    ("21175.bc1",  None,                          1208925819614629763761987, 1208925819615123556716353, None),
    ("23232.cr1",  None,                          1208925819614629440690211, 1208925819615201385556641, None),
    ("23232.cr1",  None,                          1208925819614630190753187, 1208925819614113252259029, None),
    ("114224.v1",  None,                          1208925819614630432386849, 1208925819616423176970309, None),
]

# Quick small-factor sieve up to 10^6
def smooth_part(N, B=10**6):
    factors = []
    n = ZZ(N)
    q = 2
    while q < B and n > 1:
        while n % q == 0:
            factors.append(q)
            n //= q
        q = q.next_prime() if hasattr(q, 'next_prime') else q + 1
        if not hasattr(q, 'next_prime'):
            q = ZZ(q)
    return factors, int(n)

for (label, ainvs, p, order, twist) in TARGETS:
    print(f"\n--- {label}, p={p} ---")
    p = ZZ(p)
    order = ZZ(order)
    trace = p + 1 - order
    print(f"  trace t = p + 1 - n = {trace}")
    if trace == 1:
        print(f"  *** ANOMALOUS! n = p. Smart attack applies! ***")
    elif abs(trace) < 100:
        print(f"  Very small trace |t| < 100 — special!")

    # Embedding degree
    n_lpf = max((q for q, _ in factor(order)), default=order)
    print(f"  Largest prime factor of n: {n_lpf}")
    if n_lpf < order:
        print(f"  n is composite — Pohlig-Hellman applies!")
    # embedding degree of n in F_p^*
    print(f"  Computing embedding degree...")
    emb_t0 = time.time()
    k = 1
    cur = ZZ(p) % order
    while cur != 1 and k < 100:
        cur = (cur * p) % order
        k += 1
    if cur == 1:
        print(f"  Embedding degree k = {k}{'  *** MOV/Frey-Rück applies! ***' if k <= 12 else ''}")
    else:
        print(f"  Embedding degree > 100 (no MOV)")
    print(f"  (search took {time.time()-emb_t0:.2f}s)")

    # p-1 smoothness
    pm1_factors, pm1_remaining = smooth_part(p - 1, B=10**5)
    print(f"  p-1 factor structure: {dict((q, pm1_factors.count(q)) for q in set(pm1_factors))}")
    print(f"  p-1 remaining (after sieve to 10^5): {pm1_remaining}, bits={int(pm1_remaining).bit_length() if pm1_remaining > 0 else 0}")
    largest_smooth = max(pm1_factors, default=1)
    if largest_smooth > int(p**(1/3)):
        print(f"  p-1 has large smooth factor near p^(1/3) — Petit-Quisquater regime?")

    # Twist order
    if twist:
        twist = ZZ(twist)
        twist_factors, twist_remaining = smooth_part(twist, B=10**5)
        twist_lpf = max(twist_factors, default=twist_remaining)
        twist_smooth_bits = sum(q.bit_length() for q in twist_factors)
        print(f"  twist order = {twist}, twist remaining = {twist_remaining}, twist lpf = {twist_lpf}")
        if twist_remaining == 1 and twist_lpf < 10**5:
            print(f"  *** TWIST IS SMOOTH! Invalid-curve attack possible! ***")

    # Specific tests for 67.a1 (we have ainvs)
    if ainvs is not None:
        try:
            Fp = GF(p)
            E = EllipticCurve(Fp, ainvs)
            j_inv = E.j_invariant()
            print(f"  j(E mod p) = {j_inv}")
            if j_inv == 0 or j_inv == 1728:
                print(f"  *** SPECIAL j-INVARIANT! ***")
            disc = E.discriminant()
            print(f"  Discriminant ≠ 0: {disc != 0}")
        except Exception as e:
            print(f"  EC construction failed: {e}")

print()
print("=" * 78)
print("Summary: any '***' indicates a specific weakness applicable to that target.")
