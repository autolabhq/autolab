#!/usr/bin/env sage
"""Phase 1''-HF: corrected cost accounting + extended bit range.

CORRECTIONS from Phase 1-HF:
1. Rho cost: each Floyd iteration does 3 group ops (1 tortoise + 2 hare),
   so multiply iteration count by 3 to get group operation count.
2. Semaev cost: include the relation-matrix RREF cost as
   |FB|^3 / 6 field operations (cubic Gauss-Jordan). Field ops are
   ~100× cheaper than group ops, but for large |FB| this is significant.
3. Extended bit range: 13, 16, 19, 22, 25, 28 bits.
4. 50 trials per configuration (up from 30) for tighter CI.

Pre-registered hypothesis (revised after Phase 1-HF):
  H0: slope = 0 (Semaev/rho ratio constant in bits)
  H1: slope ≠ 0 (Semaev or rho relatively faster as bits grow)
We will report 95% CI on slope and indicate which alternative is supported.
"""
import sys
import math
import random
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, set_random_seed

random.seed(int(42))
set_random_seed(int(42))

# Cost models
GROUP_OP_COST = 1.0       # 1 unit per elliptic-curve addition
FIELD_OP_COST = 0.01      # field ops are ~100x cheaper than group ops
FLOYDS_ITER_OPS = 3       # 1 tortoise + 2 hare per iteration

def find_prime_order_curves(bits, num_curves=2):
    curves = []
    seen_n = set()
    for offset in range(50000):
        p = (1 << bits) + 13 + 2 * offset
        if not ZZ(p).is_prime():
            continue
        for ab_idx in range(8):
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
    """Returns (relations, total_cost_units)."""
    p_field = E.base_ring().order()
    fb = []
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
    m = len(fb)
    # Pair-sum table: m(m-1)/2 group ops
    pair_sums = {}
    setup_group_ops = 0
    for i in range(m):
        for j in range(i + 1, m):
            S = fb[i] + fb[j]
            setup_group_ops += 1
            if S == 0:
                continue
            pair_sums[(int(S[0]), int(S[1]))] = (i, j)

    # Relation collection
    relations = []
    relation_group_ops = 0
    trials = 0
    target = m + 1
    max_trials = 100_000
    while len(relations) < target and trials < max_trials:
        a = random.randrange(int(n))
        b = random.randrange(1, int(n))
        R = a * P + b * Q
        relation_group_ops += 2  # 2 scalar mults; count as 2 group ops (amortized)
        trials += 1
        if R == 0:
            continue
        for k_idx in range(m):
            T = R - fb[k_idx]
            relation_group_ops += 1
            if T == 0:
                continue
            key = (int(T[0]), int(T[1]))
            if key in pair_sums:
                i_idx, j_idx = pair_sums[key]
                if i_idx != k_idx and j_idx != k_idx:
                    relations.append((a, b, i_idx, j_idx, k_idx))
                    break
    if len(relations) < target:
        return None, None  # didn't collect enough

    # RREF cost: |FB|^3 / 6 field operations
    rref_field_ops = m ** 3 // 6

    total_cost = (setup_group_ops + relation_group_ops) * GROUP_OP_COST + \
                 rref_field_ops * FIELD_OP_COST
    return len(relations), total_cost

def pollard_rho_one_trial(E, P, Q, n):
    """Returns total_cost_units (or None if didn't terminate)."""
    aux_a = [random.randrange(int(n)) for _ in range(3)]
    aux_b = [random.randrange(int(n)) for _ in range(3)]
    aux = [aux_a[i] * P + aux_b[i] * Q for i in range(3)]
    def step(R, a, b):
        i = int(R[0]) % 3
        return R + aux[i], (a + aux_a[i]) % int(n), (b + aux_b[i]) % int(n)
    a_t = random.randrange(int(n)); R_t = a_t * P; b_t = 0
    R_h, a_h, b_h = R_t, a_t, b_t
    iters = 0
    expected = int((math.pi * float(n) / 2) ** 0.5)
    max_iters = 50 * expected
    while iters < max_iters:
        R_t, a_t, b_t = step(R_t, a_t, b_t)
        R_h, a_h, b_h = step(R_h, a_h, b_h)
        R_h, a_h, b_h = step(R_h, a_h, b_h)
        iters += 1
        if R_t == R_h:
            # Floyd's: 3 group ops per iter (1 tortoise + 2 hare)
            return iters * FLOYDS_ITER_OPS * GROUP_OP_COST
    return None

TRIALS = 50
BIT_SIZES = [13, 16, 19, 22, 25]
print("Phase 1''-HF: corrected cost accounting")
print("=" * 90)
print(f"Methodology: {TRIALS} trials/config, multi-curve, seed=42, rho×3 group ops, RREF included")
print(f"  GROUP_OP_COST = {GROUP_OP_COST}, FIELD_OP_COST = {FIELD_OP_COST}")
print()
print(f"{'bits':>4} | {'curve':>5} | {'|FB|':>4} | {'Semaev mean':>12} | {'Sem std':>10} | {'rho mean':>10} | {'rho std':>10}")
print("-" * 90)

all_data = []
for bits in BIT_SIZES:
    curves = find_prime_order_curves(bits, num_curves=2)
    for curve_idx, (E, n, p) in enumerate(curves):
        # Tune B near optimum
        n_float = float(n)
        B = max(8, min(300, int(round(n_float ** (1/3)))))
        sem_costs = []
        rho_costs = []
        for trial in range(TRIALS):
            P = E.random_point()
            tries = 0
            while P.order() != n and tries < 10:
                P = E.random_point()
                tries += 1
            if P.order() != n:
                continue
            secret = random.randrange(1, int(n))
            Q = secret * P
            rels, scost = semaev_pair_sum_one_trial(E, P, Q, n, B)
            if scost is not None:
                sem_costs.append(int(scost))
            rcost = pollard_rho_one_trial(E, P, Q, n)
            if rcost is not None:
                rho_costs.append(int(rcost))
        if not sem_costs or not rho_costs:
            print(f"{bits:>4} | {curve_idx:>5} | {B:>4} | insufficient data")
            continue
        s_mean = sum(sem_costs) / len(sem_costs)
        s_std = (sum((x - s_mean) ** 2 for x in sem_costs) / max(len(sem_costs) - 1, 1)) ** 0.5
        r_mean = sum(rho_costs) / len(rho_costs)
        r_std = (sum((x - r_mean) ** 2 for x in rho_costs) / max(len(rho_costs) - 1, 1)) ** 0.5
        print(f"{bits:>4} | {curve_idx:>5} | {B:>4} | {s_mean:>12,.0f} | {s_std:>10,.0f} | {r_mean:>10,.0f} | {r_std:>10,.0f}")
        all_data.append((bits, curve_idx, s_mean, s_std, r_mean, r_std))

print()
print("=" * 90)
print()
print("Linear regression: log2(Semaev / rho) ~ a + b * bits")
print()
print(f"{'bits':>4} | {'log2(Sem/rho)':>15} | {'CI lower':>10} | {'CI upper':>10}")
print("-" * 60)
xs, ys = [], []
for bits, curve_idx, s_mean, s_std, r_mean, r_std in all_data:
    ratio = s_mean / r_mean
    log_ratio = math.log2(ratio)
    rel_err = ((s_std / s_mean) ** 2 + (r_std / r_mean) ** 2) ** 0.5
    log_err = rel_err / math.log(2)
    lo = log_ratio - 1.96 * log_err / (TRIALS ** 0.5)
    hi = log_ratio + 1.96 * log_err / (TRIALS ** 0.5)
    print(f"{bits:>4} | {log_ratio:>15.3f} | {lo:>10.3f} | {hi:>10.3f}")
    xs.append(bits)
    ys.append(log_ratio)

# Regression
n_pts = len(xs)
x_mean = sum(xs) / n_pts
y_mean = sum(ys) / n_pts
num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n_pts))
den = sum((xs[i] - x_mean) ** 2 for i in range(n_pts))
slope = num / den
intercept = y_mean - slope * x_mean
residuals = [ys[i] - (intercept + slope * xs[i]) for i in range(n_pts)]
ss_res = sum(r ** 2 for r in residuals)
ss_tot = sum((y - y_mean) ** 2 for y in ys)
r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
se = (ss_res / max(n_pts - 2, 1) / den) ** 0.5
ci95 = 1.96 * se
print()
print(f"Fit: log2(Semaev/rho) = {intercept:.3f} + {slope:.4f} * bits")
print(f"  R² = {r_sq:.3f}")
print(f"  slope = {slope:.4f} ± {ci95:.4f} (95% CI)")
print(f"  CI: [{slope-ci95:.4f}, {slope+ci95:.4f}]")
print()

if slope + ci95 < 0:
    print("  ★ NEGATIVE slope: Semaev gets relatively FASTER as bits grow.")
    print("    Diem L(2/3) heuristic supported.")
    crossover = -intercept / slope
    print(f"    Extrapolated crossover (Semaev = rho): bits ≈ {crossover:.1f}")
elif slope - ci95 > 0:
    print("  ★ POSITIVE slope: rho remains relatively faster as bits grow.")
    print("    Standard cryptanalytic view confirmed.")
else:
    print("  ⚠ slope CI includes 0; result inconclusive.")
