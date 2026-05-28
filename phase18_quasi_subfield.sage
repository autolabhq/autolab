#!/usr/bin/env sage
"""Phase 18.2: Quasi-subfield polynomials in artificially-embedded F_{p^2}.

Idea: base-change E/F_p to E/F_{p^2}. The group E(F_{p^2}) has order
  |E(F_{p^2})| = (p^2 + 1) - t_2
where t_2 = t^2 - 2p (by Weil conjectures).

Note: |E(F_{p^2})| factors as (p + 1 - t)(p + 1 + t) = #E(F_p) · #E^d(F_p)
where E^d is the quadratic twist.

The subgroup E(F_p) ⊂ E(F_{p^2}) gives us our target ECDLP.

Apply Huang et al. 2020 quasi-subfield polynomial construction:
  Find λ(X) of small degree such that X^p - λ(X) splits completely in F_{p^2}.
  Factor base = {(x, y) ∈ E(F_{p^2}) : x is a root of X^p - λ(X)}.

For ECDLP relation: a*P + b*Q = ∑ F_i where F_i ∈ FB.
This holds in E(F_{p^2}); descend to E(F_p) via trace.
"""
import time
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, PolynomialRing, set_random_seed

set_random_seed(int(42))

print("Phase 18.2: Quasi-subfield polynomials in F_{p^2}")
print("=" * 78)
print()

def find_prime_order_curve(bits):
    for offset in range(20000):
        p = (1 << bits) + 13 + 2*offset
        if not ZZ(p).is_prime(): continue
        E = EllipticCurve(GF(p), [3, 5])
        n = ZZ(E.order())
        if n.is_prime() and n > p // 2:
            return E, n, p
    return None, None, None

# Test at small bit size first
bits = 13
print(f"--- Toy example at bits={bits} ---")
E_Fp, n, p = find_prime_order_curve(bits)
print(f"  E/F_{p}: y^2 = x^3 + 3x + 5")
print(f"  #E(F_p) = n = {n}")

# Base-change to F_{p^2}
print(f"\n  Base-changing to F_p^2 = GF({p}^2)...")
Fp2 = GF(p**2, name='z')
z = Fp2.gen()
E_Fp2 = EllipticCurve(Fp2, [3, 5])
n_Fp2 = ZZ(E_Fp2.order())
print(f"  #E(F_p^2) = {n_Fp2}")
print(f"  Factor check: n * twist_order = {n * (p + 1 - (p + 1 - n))} = ??")

# Actually #E(F_p^2) = (p^2 + 1 - t^2 + 2p) where t = p + 1 - n
t = p + 1 - n
expected_n2 = p**2 + 1 - (t**2 - 2*p)
print(f"  Expected #E(F_p^2) = p^2 + 1 - (t^2 - 2p) = {expected_n2}")
print(f"  Matches: {n_Fp2 == expected_n2}")

# Construct quasi-subfield polynomial
# Need λ(X) of small degree d such that X^p - λ(X) factors completely in F_{p^2}
# The simplest case: λ(X) = X (degree 1). Then X^p - X factors as ∏_{a ∈ F_p}(X - a).
# This gives the SUBFIELD F_p, which is what we already have. Trivial.
#
# For nontrivial quasi-subfield in F_{p^2}, need λ(X) with degree > 1.
# Try λ(X) = X^d for various small d.
print(f"\n  Searching for nontrivial quasi-subfield polynomials...")
PRx = PolynomialRing(Fp2, name='X')
X = PRx.gen()

# Try simple λ(X) = c*X^d
found_quasi = []
for d in range(2, 5):
    for c in [Fp2(1), Fp2(2), z, z + 1]:
        lambda_poly = c * X**d
        F = X**p - lambda_poly
        # Check if F factors completely in F_{p^2}
        # If so, F has p roots in F_{p^2}.
        try:
            t0 = time.time()
            facs = F.factor()
            elapsed = time.time() - t0
            # Check if all factors are linear
            all_linear = all(f[0].degree() == 1 for f in facs)
            print(f"  λ = {c}*X^{d}: F = X^p - λ has factorization with {len(facs)} factors, all linear? {all_linear} ({elapsed:.2f}s)")
            if all_linear and len(facs) >= 2:
                found_quasi.append((c, d, F))
        except Exception as e:
            print(f"  λ = {c}*X^{d}: factorization failed: {e}")

if not found_quasi:
    print(f"\n  No nontrivial quasi-subfield polynomial found at this bit size.")
    print(f"  This is consistent with Huang et al.'s observation: existence")
    print(f"  is rare; statistical arguments show generic (q, n) don't have them.")

# What about using degree d = (p-1)/m for some m | p^2 - 1?
print(f"\n  Trying X^p - X^c for various c relating to F_p^2 - F_p structure...")
# The multiplicative group F_{p^2}^* has order p^2 - 1 = (p-1)(p+1).
# Elements of order dividing (p+1) are in F_p^2 but not F_p (generally).
# F = X^(p+1) - 1 = ∏_{x ∈ F_p^2*, ord(x) | p+1} (X - x)
# This is a structured polynomial of degree p+1.
F_pplus1 = X**(p+1) - 1
print(f"  X^(p+1) - 1 has degree {F_pplus1.degree()} = p+1 = {p+1}")
t0 = time.time()
facs = F_pplus1.factor()
elapsed = time.time() - t0
print(f"  Factorization: {len(facs)} factors, all linear? {all(f[0].degree() == 1 for f in facs)} ({elapsed:.2f}s)")
if all(f[0].degree() == 1 for f in facs):
    # This gives a factor base of size p+1 in F_{p^2}^*
    # Apply to ECDLP: factor base of E(F_{p^2}) with x ∈ roots
    print(f"\n  *** Constructed factor base: {p+1} roots of X^(p+1) - 1 in F_p^2 ***")
    # Check how many are x-coords of E(F_p^2) points
    fb_E = []
    fb_x_set = set()
    for fac in facs:
        x_val = -fac[0].constant_coefficient()  # root of (X - x_val)
        pts = E_Fp2.lift_x(x_val, all=True)
        if pts:
            fb_E.append(pts[0])
            fb_x_set.add(x_val)
    print(f"  Factor base x ∈ quasi-subfield ∩ E_x: {len(fb_E)}")

    # Check if factor base provides relations
    # Relation: a*P + b*Q = F_i + F_j + F_k for F_i ∈ fb_E
    P_Fp = E_Fp.random_point()
    while P_Fp.order() != n: P_Fp = E_Fp.random_point()
    import secrets
    secret = secrets.randbelow(int(n)-1) + 1
    Q_Fp = secret * P_Fp

    # Lift to F_{p^2}
    P_Fp2 = E_Fp2(int(P_Fp[0]), int(P_Fp[1]))
    Q_Fp2 = E_Fp2(int(Q_Fp[0]), int(Q_Fp[1]))

    print(f"\n  Searching for relations a*P + b*Q = F_i + F_j in F_p^2 factor base...")
    # Width-3 pair sum
    pair_sums = {}
    for i in range(len(fb_E)):
        for j in range(i+1, len(fb_E)):
            S = fb_E[i] + fb_E[j]
            if S == 0: continue
            pair_sums[(S[0], S[1])] = (i, j)

    relations = 0
    queries = 0
    t0 = time.time()
    while relations < 5 and time.time() - t0 < 30:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P_Fp2 + b*Q_Fp2
        queries += 1
        if R == 0: continue
        for k_idx in range(len(fb_E)):
            T = R - fb_E[k_idx]
            if T == 0: continue
            if (T[0], T[1]) in pair_sums:
                relations += 1
                break
    elapsed = time.time() - t0
    print(f"  Found {relations} relations in {queries} queries, {elapsed:.2f}s")
    if relations > 0:
        # Compare to standard F_p factor base
        print(f"  *** SUCCESS: quasi-subfield FB in F_p^2 gives valid relations ***")
        print(f"    (but these need to be descended to F_p via trace map — see below)")
        print(f"  Per-query relation rate: {relations/max(queries,1):.4f}")

print()
print("=" * 78)
print()
print("CONCLUSION:")
print("  Phase 18.2 status: the F_p^2 quasi-subfield construction is")
print("  computable. Whether relations descend to F_p ECDLP via the")
print("  trace map remains to be analyzed (this is the Weil-restriction")
print("  question, which Phase 6.2 already showed is closed for our")
print("  prime-field-original curves).")
print()
print("  If F_p^2 relations directly contribute to F_p ECDLP via Weil")
print("  trace, the attack works. Phase 6.2 result suggests NO direct")
print("  contribution because the prime-field-original curves are not")
print("  cover-curve constructions.")
