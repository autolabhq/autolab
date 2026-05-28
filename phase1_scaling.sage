#!/usr/bin/env sage
"""Phase 1 sub-experiment 1 (extended): Semaev vs rho scaling across n.

For each n bit-size in {15, 18, 21, 24, 27, 30}, measure:
  - cost ratio Semaev/rho at optimal B (where Semaev is empirically fastest)
  - whether the ratio improves (decreases) as n grows

This is the empirical test of whether Semaev pair-sum could become
faster than rho at the bit sizes we care about (80-bit).
"""
import time
import secrets
import sys
import math
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

def semaev_pair_sum_cost(E, P, Q, n, B, budget_sec=10):
    """Run Semaev pair-sum at factor base size B; report cost per relation."""
    p_field = E.base_ring().order()
    fb = []
    for x in range(B):
        pts = E.lift_x(GF(p_field)(x), all=True)
        if pts: fb.append(pts[0])
        if len(fb) >= B: break
    if len(fb) < 8:
        return None, None, None, None

    pair_sums = {}
    for i in range(len(fb)):
        for j in range(i+1, len(fb)):
            S = fb[i] + fb[j]
            if S == 0: continue
            pair_sums[tuple([int(S[0]), int(S[1])])] = (i, j)

    relations = []
    trials = 0
    ops = 0
    t0 = time.time()
    target_rels = min(len(fb) + 4, 32)
    while len(relations) < target_rels and time.time() - t0 < budget_sec:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a * P + b * Q
        trials += 1
        ops += 2  # 2 scalar mults (amortized for analysis)
        if R == 0: continue
        for k_idx in range(len(fb)):
            T = R - fb[k_idx]
            ops += 1
            if T == 0: continue
            key = (int(T[0]), int(T[1]))
            if key in pair_sums:
                i_idx, j_idx = pair_sums[key]
                if i_idx != k_idx and j_idx != k_idx:
                    relations.append((a, b, [i_idx, j_idx, k_idx]))
                    break
    elapsed = time.time() - t0
    n_rels = len(relations)
    if n_rels < len(fb)//2 or n_rels == 0:
        return len(fb), n_rels, None, None
    total_ops_for_full_solve = (len(fb) + 1) * float(ops) / float(n_rels)
    return len(fb), n_rels, float(ops) / float(n_rels), total_ops_for_full_solve

print(f"{'bits':>4} | {'n':>12} | {'rho_ops':>10} | {'best_B':>6} | {'best_FB':>7} | {'best_ratio':>10} | {'log2(ratio)':>11}")
print("-" * 85)

scaling_data = []
for bits in [13, 16, 19, 22, 25]:
    E, n = find_prime_order_curve(bits)
    if E is None: continue
    p_field = E.base_ring().order()
    P = E.random_point()
    while P.order() != n: P = E.random_point()
    secret = secrets.randbelow(int(n)-1) + 1
    Q = secret * P

    rho_ops = int((math.pi * float(n) / 2)**0.5)

    # Test a range of B values
    best_ratio = float('inf')
    best_B = None
    best_FB = None
    for B in [32, 64, 128, 256, 512, 1024, 2048]:
        if B > p_field: break
        fb_size, n_rels, ops_per_rel, total_ops = semaev_pair_sum_cost(E, P, Q, n, B, budget_sec=30)
        if total_ops is None: continue
        ratio = total_ops / rho_ops  # > 1 means rho wins; < 1 means Semaev wins
        if ratio < best_ratio:
            best_ratio = ratio
            best_B = B
            best_FB = fb_size

    log2_ratio = math.log2(best_ratio) if best_ratio < float('inf') and best_ratio > 0 else None
    print(f"{bits:>4} | {int(n):>12,d} | {rho_ops:>10,d} | "
          f"{best_B or '—':>6} | {best_FB or '—':>7} | "
          f"{best_ratio:>10.2f} | {log2_ratio if log2_ratio is not None else '—':>11}")
    if log2_ratio is not None:
        scaling_data.append((bits, log2_ratio))

print()
print("=" * 85)
print("Scaling fit:")
if len(scaling_data) >= 3:
    # Linear regression: log2(ratio) ~ a + b*bits
    xs = [d[0] for d in scaling_data]
    ys = [d[1] for d in scaling_data]
    n_pts = len(xs)
    x_mean = sum(xs) / n_pts
    y_mean = sum(ys) / n_pts
    num = sum((xs[i]-x_mean)*(ys[i]-y_mean) for i in range(n_pts))
    den = sum((xs[i]-x_mean)**2 for i in range(n_pts))
    slope = num / den if den > 0 else 0
    intercept = y_mean - slope * x_mean
    print(f"  log2(Semaev/rho ratio)  ≈  {intercept:.2f}  +  {slope:.3f} × bits")
    # Extrapolate
    print()
    print(f"  Extrapolation:")
    for n_bits in [40, 50, 60, 70, 80]:
        proj = intercept + slope * n_bits
        ratio_proj = 2**proj
        print(f"    bits={n_bits}: log2(ratio) ≈ {proj:>6.2f}  → ratio ≈ {ratio_proj:.3e}")
    print()
    print("  Interpretation:")
    if slope < 0:
        print("    NEGATIVE slope: Semaev gets RELATIVELY FASTER as n grows.")
        print("    The crossover (ratio = 1) happens at:")
        crossover_bits = -intercept / slope if slope < 0 else None
        if crossover_bits:
            print(f"      bits ≈ {crossover_bits:.1f}")
            if crossover_bits < 80:
                print(f"      → Semaev SHOULD BEAT rho at 80-bit (if extrapolation holds)")
            else:
                print(f"      → Semaev still loses to rho at 80-bit")
    else:
        print("    POSITIVE slope: Semaev gets RELATIVELY SLOWER as n grows.")
        print("    No crossover — Pollard rho dominates at all n.")
