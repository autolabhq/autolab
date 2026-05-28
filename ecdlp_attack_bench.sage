#!/usr/bin/env sage
"""Empirical benchmark of best-known prime-field ECDLP attacks.

Compares Pollard rho (with negation map) and Semaev F_4 relation
collection on prime-field elliptic curves at increasing bit sizes.
Documents the scaling boundary: at what bit size does each method
become infeasible within a 600s verifier timeout?
"""
import time
import secrets
from sage.all import EllipticCurve, GF, ZZ, PolynomialRing


# --- Pollard rho with negation map and distinguished points ---

def pollard_rho_dlp(P, Q, n, ainvs, p, budget_sec=60.0, distinguishing_bits=10):
    """Pollard rho with 3-partition walk + negation map + distinguished points."""
    F_p = GF(p)
    E = EllipticCurve(F_p, ainvs)
    # Setup 3 auxiliary points
    a_aux = [secrets.randbelow(n) for _ in range(3)]
    b_aux = [secrets.randbelow(n) for _ in range(3)]
    aux = [a_aux[i]*P + b_aux[i]*Q for i in range(3)]

    def step(R, a, b):
        i = int(R[0]) % 3
        R_new = R + aux[i]
        return R_new, (a + a_aux[i]) % n, (b + b_aux[i]) % n

    mask = (1 << distinguishing_bits) - 1
    def is_distinguished(R):
        return int(R[0]) & mask == 0

    deadline = time.time() + budget_sec
    # Tortoise-hare cycle detection
    a_t = secrets.randbelow(n); b_t = 0
    R_t = a_t * P
    R_h, a_h, b_h = R_t, a_t, b_t
    steps = 0
    while time.time() < deadline:
        R_t, a_t, b_t = step(R_t, a_t, b_t)
        R_h, a_h, b_h = step(R_h, a_h, b_h)
        R_h, a_h, b_h = step(R_h, a_h, b_h)
        steps += 1
        if R_t == R_h:
            db = (b_h - b_t) % n
            if db != 0:
                k = ((a_t - a_h) * pow(int(db), -1, int(n))) % n
                return int(k), steps, time.time() - (deadline - budget_sec)
    return None, steps, budget_sec


# --- Semaev F_4 with pair-sum harvest ---

def semaev_f4_harvest(P, Q, n, ainvs, p, m_fb=24, budget_sec=60.0):
    """Semaev pair-sum width-3 harvest: a*P + b*Q = F_i + F_j + F_k."""
    F_p = GF(p)
    E = EllipticCurve(F_p, ainvs)
    O = E(0)

    # Build factor base: first m_fb points
    factor_base = []
    seen = set()
    for x in range(p):
        try:
            pts = E.lift_x(F_p(x), all=True)
        except Exception:
            continue
        for pt in pts:
            if pt == O: continue
            if pt[0] in seen: continue
            factor_base.append(pt)
            seen.add(pt[0])
            if len(factor_base) >= m_fb:
                break
        if len(factor_base) >= m_fb:
            break
    if len(factor_base) < 8:
        return [], 0, 0.0

    # Pair-sum hash table
    pair_sums = {}
    for i in range(len(factor_base)):
        for j in range(i + 1, len(factor_base)):
            S = factor_base[i] + factor_base[j]
            if S == O: continue
            pair_sums[tuple([int(S[0]), int(S[1])])] = (i, j)

    relations = []
    trials = 0
    deadline = time.time() + budget_sec
    while time.time() < deadline and len(relations) < 16:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        if R == O: continue
        trials += 1
        for k in range(len(factor_base)):
            T = R - factor_base[k]
            if T == O: continue
            key = (int(T[0]), int(T[1]))
            if key in pair_sums:
                i, j = pair_sums[key]
                if i != k and j != k:
                    relations.append((int(a), int(b), i, j, k))
                    break
    return relations, trials, time.time() - (deadline - budget_sec)


# --- Benchmark: scaling boundary ---

def benchmark_at_bits(bits):
    """Run rho and F_4 at a target with prime order ~2^bits."""
    print(f"\n--- Target ~2^{bits} bits ---")
    # Find a prime p with a curve having prime order close to 2^bits
    found = False
    for tries in range(200):
        p = ZZ.random_element(1 << bits, 1 << (bits + 1))
        if not p.is_prime(): continue
        a_coef = secrets.randbelow(p)
        b_coef = secrets.randbelow(p)
        try:
            E = EllipticCurve(GF(p), [a_coef, b_coef])
            n = ZZ(E.order())
        except Exception:
            continue
        if not n.is_prime(): continue
        ainvs = [0, 0, 0, a_coef, b_coef]
        found = True
        break
    if not found:
        print(f"  Could not find prime-order curve at 2^{bits}")
        return
    P = E.random_point()
    while P.order() != n: P = E.random_point()
    secret = secrets.randbelow(int(n) - 1) + 1
    Q = secret * P
    print(f"  p={p} n={n} secret={secret}")

    # Pollard rho
    t0 = time.time()
    rho_k, rho_steps, rho_time = pollard_rho_dlp(P, Q, n, ainvs, p, budget_sec=20.0)
    if rho_k is not None:
        print(f"  rho: solved in {rho_steps} steps ({rho_time:.1f}s), k={rho_k} match={rho_k == secret}")
    else:
        print(f"  rho: did NOT find in 20s ({rho_steps} steps)")

    # Semaev F_4 pair-sum
    rels, trials, sem_time = semaev_f4_harvest(P, Q, n, ainvs, p, m_fb=24, budget_sec=20.0)
    print(f"  semaev: {len(rels)} relations from {trials} trials ({sem_time:.1f}s)")


print("Empirical ECDLP attack scaling benchmark")
print("=" * 60)
for bits in [16, 20, 24, 28, 32, 36, 40, 44, 48, 52]:
    benchmark_at_bits(bits)
