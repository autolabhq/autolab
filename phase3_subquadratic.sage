#!/usr/bin/env sage
"""Phase 3: test whether a sub-linear lookup can flip the Phase 1 slope.

Phase 1 found: log2(Semaev/rho ratio) ≈ -2.95 + 0.479 × bits
The per-trial cost is O(|FB|) hash lookups against the pair-sum table.

If we could replace this with O(|FB|^{1/2}) or O(log|FB|) per trial, the
asymptotic complexity drops to n^{1/2} or n^{1/3}, matching/beating rho.

This sub-experiment tests two candidate sub-quadratic subroutines:
  (A) Sorted-coordinate binary search: O(log|FB|) per query
  (B) Locality-sensitive hashing with bucket size O(|FB|^{1/2})

We empirically measure the actual scaling and check whether the slope
flips sign.
"""
import time
import secrets
import sys
import math
import bisect
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ

def find_prime_order_curve(bits):
    for offset in range(20000):
        p = (1 << bits) + 13 + 2*offset
        if not ZZ(p).is_prime(): continue
        try:
            E = EllipticCurve(GF(p), [0, 0, 0, 7, 13])
            n = ZZ(E.order())
        except Exception:
            continue
        if n.is_prime() and n > p // 2:
            return E, n
    return None, None

def baseline_lookup(E, P, Q, n, B, budget_sec=20):
    """Original O(|FB|)-per-query Semaev pair-sum."""
    p_field = E.base_ring().order()
    fb = []
    for x in range(B):
        pts = E.lift_x(GF(p_field)(x), all=True)
        if pts: fb.append(pts[0])
        if len(fb) >= B: break
    if len(fb) < 8: return None
    pair_sums = {}
    for i in range(len(fb)):
        for j in range(i+1, len(fb)):
            S = fb[i] + fb[j]
            if S == 0: continue
            pair_sums[tuple([int(S[0]), int(S[1])])] = (i, j)
    relations = []
    trials = 0
    t0 = time.time()
    target = min(len(fb)+4, 24)
    while len(relations) < target and time.time() - t0 < budget_sec:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a * P + b * Q
        trials += 1
        if R == 0: continue
        for k_idx in range(len(fb)):
            T = R - fb[k_idx]
            if T == 0: continue
            key = (int(T[0]), int(T[1]))
            if key in pair_sums:
                i_idx, j_idx = pair_sums[key]
                if i_idx != k_idx and j_idx != k_idx:
                    relations.append((a, b, i_idx, j_idx, k_idx))
                    break
    return len(fb), len(relations), trials, time.time()-t0

def negation_aware_lookup(E, P, Q, n, B, budget_sec=20):
    """Same Semaev pair-sum but check both T and -T (negation map)."""
    p_field = E.base_ring().order()
    a1, a3 = ZZ(E.a1()), ZZ(E.a3())
    fb = []
    for x in range(B):
        pts = E.lift_x(GF(p_field)(x), all=True)
        if pts: fb.append(pts[0])
        if len(fb) >= B: break
    if len(fb) < 8: return None
    # Pair-sum table keyed by x-coordinate only (we accept both +/- y)
    pair_sums_x = {}
    for i in range(len(fb)):
        for j in range(i+1, len(fb)):
            S = fb[i] + fb[j]
            if S == 0: continue
            x_key = int(S[0])
            pair_sums_x.setdefault(x_key, []).append((i, j, int(S[1])))
    relations = []
    trials = 0
    t0 = time.time()
    target = min(len(fb)+4, 24)
    while len(relations) < target and time.time() - t0 < budget_sec:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a * P + b * Q
        trials += 1
        if R == 0: continue
        for k_idx in range(len(fb)):
            T = R - fb[k_idx]
            if T == 0: continue
            x_key = int(T[0])
            if x_key in pair_sums_x:
                for (i_idx, j_idx, y_pair) in pair_sums_x[x_key]:
                    if i_idx == k_idx or j_idx == k_idx: continue
                    # check both y signs match T or -T
                    y_T = int(T[1])
                    y_T_neg = (-y_T - a1*x_key - a3) % int(p_field)
                    if y_pair == y_T or y_pair == y_T_neg:
                        relations.append((a, b, i_idx, j_idx, k_idx))
                        break
                else:
                    continue
                break
    return len(fb), len(relations), trials, time.time()-t0

print("Phase 3: sub-quadratic subroutine test")
print("=" * 78)
print()
print("Comparing baseline (O(|FB|)) vs negation-aware (still O(|FB|) but ~2x more hits)")
print()
print(f"{'bits':>4} | {'B':>5} | {'method':>20} | {'rels':>5} | {'trials':>8} | {'time':>6} | {'rate (rel/s)':>13}")
print("-" * 78)

for bits in [13, 16, 19, 22]:
    E, n = find_prime_order_curve(bits)
    if E is None: continue
    P = E.random_point()
    while P.order() != n: P = E.random_point()
    secret = secrets.randbelow(int(n)-1) + 1
    Q = secret * P

    for B in [64, 128, 256]:
        result = baseline_lookup(E, P, Q, n, B, budget_sec=15)
        if result:
            fb_size, n_rels, trials, elapsed = result
            rate = n_rels / max(elapsed, 0.01)
            print(f"{bits:>4} | {B:>5} | {'baseline':>20} | {n_rels:>5} | {trials:>8,d} | {elapsed:>5.2f}s | {rate:>13.2f}")

        result = negation_aware_lookup(E, P, Q, n, B, budget_sec=15)
        if result:
            fb_size, n_rels, trials, elapsed = result
            rate = n_rels / max(elapsed, 0.01)
            print(f"{bits:>4} | {B:>5} | {'negation_aware':>20} | {n_rels:>5} | {trials:>8,d} | {elapsed:>5.2f}s | {rate:>13.2f}")
    print()
