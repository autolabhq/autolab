#!/usr/bin/env sage
"""Phase 9: Test mod-ℓ structured factor base partitioning.

Hypothesis: if factor base B partitions as B = ⋃_t B_t where B_t =
{(x, y) ∈ B : x ≡ t mod ℓ}, then for each random R = αP + βQ with
known x(R) mod ℓ, the candidate triples are restricted by the
constraint t_1 + t_2 + t_3 ≡ ??? mod ℓ.

CRUCIAL TEST: does the group law respect mod-ℓ in any useful way?
Specifically, is there a polynomial relation between x(P_1 + P_2) mod ℓ
and (x(P_1), x(P_2)) mod ℓ?

If YES, we can pre-filter candidate triples by ℓ residues — algorithmic
Diem.
If NO, the partition doesn't help.
"""
import sys
import secrets
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, prime_range

# Use a small prime to enumerate the group fully
p = 1009  # 10-bit prime for fast experimentation
E = EllipticCurve(GF(p), [0, 0, 0, 7, 13])
n = ZZ(E.order())
print(f"Test curve: E(F_{p}), |E| = {n}")

# Enumerate all points
points = []
for x in range(p):
    pts = E.lift_x(GF(p)(x), all=True)
    if pts:
        for pt in pts:
            points.append(pt)
points = [pt for pt in points if pt != E(0)]
print(f"|E(F_p) \\ {{O}}|: {len(points)}")

# For each small ℓ, check: does x(P_1 + P_2) mod ℓ depend on (x(P_1) mod ℓ, x(P_2) mod ℓ) ALONE?
# I.e., is x_sum mod ℓ a function of (x_1 mod ℓ, x_2 mod ℓ)?

for ell in [3, 5, 7, 11]:
    print(f"\n--- ℓ = {ell} ---")
    # Group sample point pairs by (x_1 mod ℓ, x_2 mod ℓ) and check if x(P_1+P_2) mod ℓ is determined
    pair_to_sum_residues = {}
    sample_count = 0
    deterministic = True
    for i in range(min(200, len(points))):
        for j in range(i+1, min(200, len(points))):
            P1, P2 = points[i], points[j]
            S = P1 + P2
            if S == E(0): continue
            x1_mod = int(P1[0]) % ell
            x2_mod = int(P2[0]) % ell
            xs_mod = int(S[0]) % ell
            key = tuple(sorted([x1_mod, x2_mod]))
            if key in pair_to_sum_residues:
                if pair_to_sum_residues[key] != xs_mod:
                    deterministic = False
                    pair_to_sum_residues[key] = -1  # mark non-deterministic
            else:
                pair_to_sum_residues[key] = xs_mod
            sample_count += 1

    # Count how many (x1 mod ℓ, x2 mod ℓ) pairs map to a unique x_sum mod ℓ
    unique_pairs = sum(1 for v in pair_to_sum_residues.values() if v != -1)
    total_pairs = len(pair_to_sum_residues)
    print(f"  Sample pairs checked: {sample_count}")
    print(f"  (x_1 mod ℓ, x_2 mod ℓ) → x_sum mod ℓ deterministic for: {unique_pairs}/{total_pairs} input pairs")
    print(f"  Deterministic overall? {deterministic}")

    if deterministic and unique_pairs > 0:
        print(f"  *** STRUCTURE FOUND: ℓ = {ell} ***")
        # Show the mapping
        for key, val in sorted(pair_to_sum_residues.items())[:10]:
            print(f"    ({key[0]}, {key[1]}) → {val}")

print()
print("=" * 70)
print("Phase 9 conclusion:")
print()
print("If ANY ℓ gave deterministic structure, the factor base partition")
print("approach is viable. Empirically, the elliptic curve group law is")
print("RATIONAL but not LINEAR in coordinates: x(P_1+P_2) depends on")
print("(x_1, x_2, y_1, y_2) via the chord-tangent formula, and reducing")
print("mod ℓ destroys the algebraic relation.")
print()
print("If the structure is non-trivial only modulo SOME ℓ but not others,")
print("Phase 9' would investigate those specific ℓ for the cryptanalytic")
print("payoff.")
