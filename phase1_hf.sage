#!/usr/bin/env sage
"""Phase 1 high-fidelity rerun: Semaev pair-sum vs Pollard rho scaling.

Methodology:
  - PRE-REGISTERED HYPOTHESIS: log2(Semaev_cost / rho_cost) is linear in
    bits with slope > 0 (Semaev gets relatively slower).
  - 30 trials per (bit-size, B) configuration with different random
    seeds (deterministic via fixed Python random.seed).
  - 95% confidence interval on the slope estimate.
  - Multiple curve families: short Weierstrass y^2 = x^3 + Ax + B with
    several (A, B) pairs.
  - Cross-check: Pollard rho cost also measured empirically, not just
    extrapolated from theoretical sqrt(πn/2).

Definition of "cost":
  - Semaev cost = # group operations to collect (|FB|+1) relations
    that suffice to solve via RREF.
  - Rho cost = # group operations to find a useful collision
    (Floyd's cycle detection).

We do NOT count the relation matrix RREF cost in Semaev (it's |FB|^3
field ops, comparable to Semaev relation collection at our sizes).
"""
import time
import sys
import math
import random
import statistics
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, set_random_seed

# Fix seeds for reproducibility
random.seed(int(42))
set_random_seed(int(42))

def find_prime_order_curves(bits, num_curves=3):
    """Find up to num_curves distinct prime-order curves at given bit-size."""
    curves = []
    seen_n = set()
    for offset in range(50000):
        p = (1 << bits) + 13 + 2 * offset
        if not ZZ(p).is_prime():
            continue
        # Try several (A, B) coefficients
        for ab_idx in range(5):
            A = (3 + 7 * ab_idx) % p
            B = (5 + 11 * ab_idx) % p
            try:
                E = EllipticCurve(GF(p), [A, B])
                n = ZZ(E.order())
            except Exception:
                continue
            if n.is_prime() and n > p // 2 and int(n) not in seen_n:
                curves.append((E, n, p))
                seen_n.add(int(n))
                if len(curves) >= num_curves:
                    return curves
    return curves

def semaev_pair_sum_one_trial(E, P, Q, n, B):
    """One Semaev pair-sum run; return (relations_found, group_ops)."""
    p_field = E.base_ring().order()
    # Build structured factor base of x values 0..B-1
    fb = []
    setup_ops = 0
    for x in range(B):
        try:
            pts = E.lift_x(GF(p_field)(x), all=True)
        except Exception:
            continue
        if pts:
            fb.append(pts[0])
        if len(fb) >= B:
            break
    if len(fb) < 8:
        return None, None
    # Pair-sum hash table
    pair_sums = {}
    for i in range(len(fb)):
        for j in range(i + 1, len(fb)):
            S = fb[i] + fb[j]
            setup_ops += 1
            if S == 0:
                continue
            key = (int(S[0]), int(S[1]))
            pair_sums[key] = (i, j)
    # Collect relations
    relations = []
    ops = setup_ops
    trials = 0
    target = len(fb) + 1
    max_trials = 50000  # safety cap
    while len(relations) < target and trials < max_trials:
        a = random.randrange(int(n))
        b = random.randrange(1, int(n))  # b != 0
        R = a * P + b * Q
        ops += 2  # 2 scalar mults (counted as O(log n) but we count as 2 for amortization)
        trials += 1
        if R == 0:
            continue
        for k_idx in range(len(fb)):
            T = R - fb[k_idx]
            ops += 1
            if T == 0:
                continue
            key = (int(T[0]), int(T[1]))
            if key in pair_sums:
                i_idx, j_idx = pair_sums[key]
                if i_idx != k_idx and j_idx != k_idx:
                    relations.append((a, b, i_idx, j_idx, k_idx))
                    break
    return len(relations), ops

def pollard_rho_one_trial(E, P, Q, n):
    """One Pollard rho run; return iterations to collision."""
    aux_a = [random.randrange(int(n)) for _ in range(3)]
    aux_b = [random.randrange(int(n)) for _ in range(3)]
    aux = [aux_a[i] * P + aux_b[i] * Q for i in range(3)]
    def step(R, a, b):
        i = int(R[0]) % 3
        return R + aux[i], (a + aux_a[i]) % int(n), (b + aux_b[i]) % int(n)
    a_t = random.randrange(int(n)); R_t = a_t * P; b_t = 0
    R_h, a_h, b_h = R_t, a_t, b_t
    steps = 0
    max_steps = 50 * int((math.pi * float(n) / 2)**0.5)  # 50x expected
    while steps < max_steps:
        R_t, a_t, b_t = step(R_t, a_t, b_t)
        R_h, a_h, b_h = step(R_h, a_h, b_h)
        R_h, a_h, b_h = step(R_h, a_h, b_h)
        steps += 1
        if R_t == R_h:
            return steps
    return None  # didn't find

# Hypothesis: log2(Semaev/rho) is linear-in-bits with positive slope
TRIALS_PER_CONFIG = 30
print("Phase 1 high-fidelity rerun")
print("=" * 78)
print(f"Methodology: {TRIALS_PER_CONFIG} trials per (bits, curve), seed=42")
print()
print(f"{'bits':>4} | {'curve#':>6} | {'Semaev mean ops':>16} | {'Semaev stdev':>14} | {'rho mean':>10} | {'rho stdev':>10}")
print("-" * 80)

all_data = []
for bits in [10, 13, 16, 19]:
    curves = find_prime_order_curves(bits, num_curves=2)
    for curve_idx, (E, n, p) in enumerate(curves):
        # Best Semaev B for this n
        best_B = max(8, min(256, int(float(n)**0.4)))  # heuristic
        # Run trials
        semaev_costs = []
        rho_costs = []
        for trial in range(TRIALS_PER_CONFIG):
            # New (P, Q) per trial for statistical independence
            P = E.random_point()
            while P.order() != n:
                P = E.random_point()
            secret = random.randrange(1, int(n))
            Q = secret * P

            # Semaev
            rels, ops = semaev_pair_sum_one_trial(E, P, Q, n, best_B)
            if ops is not None:
                semaev_costs.append(ops)

            # Rho
            rho_iter = pollard_rho_one_trial(E, P, Q, n)
            if rho_iter is not None:
                rho_costs.append(rho_iter)

        if semaev_costs and rho_costs:
            # Force to plain Python int/float
            sc = [int(x) for x in semaev_costs]
            rc = [int(x) for x in rho_costs]
            s_mean = sum(sc) / len(sc)
            s_std = (sum((x - s_mean)**2 for x in sc) / max(len(sc)-1, 1))**0.5 if len(sc) >= 2 else 0
            r_mean = sum(rc) / len(rc)
            r_std = (sum((x - r_mean)**2 for x in rc) / max(len(rc)-1, 1))**0.5 if len(rc) >= 2 else 0
            print(f"{bits:>4} | {curve_idx:>6} | {s_mean:>16,.0f} | {s_std:>14,.0f} | {r_mean:>10,.0f} | {r_std:>10,.0f}")
            all_data.append((bits, curve_idx, s_mean, s_std, r_mean, r_std))

print()
print("=" * 78)
print()
print("Linear regression: log2(Semaev/rho) ~ a + b*bits")
print()
print(f"{'bits':>4} | {'log2(Semaev/rho) mean':>22} | {'log2 stdev':>10}")
print("-" * 60)
xs, ys = [], []
for bits, curve_idx, s_mean, s_std, r_mean, r_std in all_data:
    log_ratio = math.log2(s_mean / r_mean) if r_mean > 0 else None
    # Approximate stdev of log ratio via error propagation
    if log_ratio is not None and s_std > 0 and r_std > 0:
        rel_s = s_std / s_mean
        rel_r = r_std / r_mean
        log_ratio_std = (rel_s**2 + rel_r**2) ** 0.5 / math.log(2)
        print(f"{bits:>4} | {log_ratio:>22.3f} | {log_ratio_std:>10.3f}")
        xs.append(bits)
        ys.append(log_ratio)

if len(xs) >= 3:
    n_pts = len(xs)
    x_mean = sum(xs) / n_pts
    y_mean = sum(ys) / n_pts
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n_pts))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n_pts))
    slope = num / den if den > 0 else 0
    intercept = y_mean - slope * x_mean
    # Confidence interval estimate (rough)
    residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(n_pts)]
    ss_res = sum(r**2 for r in residuals)
    ss_tot = sum((y - y_mean)**2 for y in ys)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    # Slope standard error
    if n_pts >= 3:
        se = (ss_res / (n_pts - 2) / den) ** 0.5
        ci95 = 1.96 * se  # 95% CI
        print()
        print(f"Fit: log2(ratio) = {intercept:.3f} + {slope:.4f}*bits")
        print(f"  R² = {r_squared:.3f}")
        print(f"  slope = {slope:.4f} ± {ci95:.4f} (95% CI)")
        print()
        if slope - ci95 > 0:
            print(f"  ✓ Positive slope is statistically significant (95% CI excludes 0)")
            print(f"  ✓ Hypothesis confirmed: Semaev gets relatively slower as bits grow")
        else:
            print(f"  ✗ Slope is not statistically significant")
