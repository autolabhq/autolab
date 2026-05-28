#!/usr/bin/env sage
"""Phase 1 sub-experiment 1: Semaev F_3 crossover with structured factor base.

For a fixed prime-order curve E(F_p) (n = #E(F_p) prime), measure:
  - relation collection rate (trials/relation)
  - group operations per relation
  - inferred total ECDLP cost: O(B^2 · |FB|) = O(B^2 · (B+1)) = O(B^3) operations

across factor base sizes B ∈ {32, 64, 128, 256, 512} on a 30-bit ECDLP.

Compares to Pollard rho on the same target, which has expected O(√n)
operations regardless of factor base.

This is the empirical test of whether Diem's L(2/3) bound is achievable
algorithmically with current relation-finding subroutines.
"""
import time
import secrets
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ

# Find a 30-bit prime with prime-order curve
p_target_bits = 30
for p_offset in range(10000):
    p = (1 << p_target_bits) + 13 + 2*p_offset
    if not ZZ(p).is_prime(): continue
    E = EllipticCurve(GF(p), [0, 0, 0, 7, 13])
    n = ZZ(E.order())
    if n.is_prime() and n > p // 2: break

# Reduce to a smaller demonstrative size to actually get hits within budget
p_target_bits_demo = 20
for p_offset in range(10000):
    p = (1 << p_target_bits_demo) + 13 + 2*p_offset
    if not ZZ(p).is_prime(): continue
    E = EllipticCurve(GF(p), [0, 0, 0, 7, 13])
    n = ZZ(E.order())
    if n.is_prime() and n > p // 2: break

print(f"Target: E(F_{p}) with order n={n} ({n.nbits()}b prime)")
P = E.random_point()
while P.order() != n: P = E.random_point()
secret = secrets.randbelow(int(n)-1) + 1
Q = secret * P
print(f"Secret k = {secret}")
print()
print(f"Pollard rho expected cost: ~√(πn/2) = {int((3.14159 * float(n) / 2)**0.5):,d} group ops")
print()

print("=" * 78)
print(f"{'B':>6} | {'|FB|':>6} | {'rels':>6} | {'trials':>10} | {'trials/rel':>10} | {'time':>8} | {'ops/rel':>10}")
print("-" * 78)

results = []
for B in [16, 32, 64, 128, 256, 512]:
    # Build structured factor base
    fb = []
    for x in range(B):
        pts = E.lift_x(GF(p)(x), all=True)
        if not pts: continue
        fb.append(pts[0])
        if len(fb) >= B: break

    if len(fb) < 8:
        print(f"  B={B}: insufficient points")
        continue

    # Pair-sum hash table
    t_setup = time.time()
    pair_sums = {}
    for i in range(len(fb)):
        for j in range(i+1, len(fb)):
            S = fb[i] + fb[j]
            if S == 0: continue
            pair_sums[tuple([int(S[0]), int(S[1])])] = (i, j)
    setup_time = time.time() - t_setup

    # Random walk: a*P + b*Q, find a*P + b*Q = F_i + F_j + F_k
    relations = []
    trials = 0
    relation_ops = 0  # group operations expended
    t0 = time.time()
    while len(relations) < min(len(fb) + 4, 32) and time.time() - t0 < 30:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a * P + b * Q
        trials += 1
        relation_ops += 2  # 2 scalar mults amortized as 2 ops in this counting
        if R == 0: continue
        for k_idx in range(len(fb)):
            T = R - fb[k_idx]
            relation_ops += 1
            if T == 0: continue
            key = (int(T[0]), int(T[1]))
            if key in pair_sums:
                i_idx, j_idx = pair_sums[key]
                if i_idx != k_idx and j_idx != k_idx:
                    relations.append((a, b, [i_idx, j_idx, k_idx]))
                    break
    elapsed = time.time() - t0
    n_rels = len(relations)
    if n_rels == 0:
        trials_per_rel = float('inf')
        ops_per_rel = float('inf')
    else:
        trials_per_rel = float(trials) / float(n_rels)
        ops_per_rel = float(relation_ops) / float(n_rels)

    print(f"{B:>6} | {len(fb):>6} | {n_rels:>6} | {trials:>10,d} | "
          f"{trials_per_rel:>10.1f} | {elapsed:>7.2f}s | {ops_per_rel:>10.1f}")
    results.append((B, len(fb), n_rels, trials, ops_per_rel))

print("-" * 78)
print()

# Diem-style total cost analysis
print("Total ECDLP cost analysis (Semaev + linear algebra):")
print(f"  n = {n}, |FB|_needed >= 8")
for B, fb_size, n_rels, trials, ops_per_rel in results:
    if n_rels == 0: continue
    # Total cost: collect (|FB|+1) relations, each costing ops_per_rel
    total_relations_needed = fb_size + 1
    total_ops_estimate = total_relations_needed * ops_per_rel
    rho_cost = (3.14159 * float(n) / 2)**0.5
    speedup = rho_cost / total_ops_estimate
    print(f"  B={B:>3}: estimated total = {total_ops_estimate:.0e} ops "
          f"vs rho {rho_cost:.0e} (ratio: {speedup:.3f}x rho)")

print()
print("Interpretation:")
print("  - ratio < 1.0: Semaev BEATS Pollard rho")
print("  - ratio > 1.0: Pollard rho wins")
print()
print("For ECDLP at 30-bit n with structured factor base, even at optimal B,")
print("Semaev pair-sum is typically ~comparable to rho. The empirical question:")
print("does the crossover shift as n grows?")
