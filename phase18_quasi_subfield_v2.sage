#!/usr/bin/env sage
"""Phase 18.2v2: Quasi-subfield polynomials, focused test.

After v1 showed that X^p - λ(X) doesn't split for small λ, we focus
on the known-good case: X^{p+1} - 1 splits completely in F_{p^2}^*
because mu_{p+1} ⊂ F_{p^2}^* has order p+1.

Use this as factor base in E(F_{p^2}); test if relations descend to
E(F_p) via the trace map. If yes, this is a working non-naive IC
attack (a genuine novelty).
"""
import time
import secrets
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, PolynomialRing, set_random_seed

set_random_seed(int(42))

print("Phase 18.2v2: Focused test on X^(p+1) - 1 quasi-subfield FB")
print("=" * 78)

def find_prime_order_curve(bits):
    for offset in range(20000):
        p = (1 << bits) + 13 + 2*offset
        if not ZZ(p).is_prime(): continue
        E = EllipticCurve(GF(p), [3, 5])
        n = ZZ(E.order())
        if n.is_prime() and n > p // 2:
            return E, n, p
    return None, None, None

bits = 11
E_Fp, n, p = find_prime_order_curve(bits)
Fp = GF(p)
print(f"\n--- bits={bits}, p={p}, n={n} ---")

# Base-change to F_{p^2}
print(f"\n  Base-changing E to F_{{p^2}} = GF({p}^2)...")
Fp2 = GF(p**2, name='z')
z = Fp2.gen()
E_Fp2 = EllipticCurve(Fp2, [3, 5])
n_Fp2 = ZZ(E_Fp2.order())
print(f"  #E(F_p^2) = {n_Fp2}")
t = p + 1 - n
print(f"  Trace t = {t}, expected #E(F_p^2) = (p+1-t)(p+1+t) = {(p+1-t)*(p+1+t)}")

# Use mu_{p+1} ⊂ F_{p^2}^* as the quasi-subfield
# Elements x ∈ F_{p^2}^* with x^{p+1} = 1
g_Fp2 = Fp2.multiplicative_generator()
# Element of order p+1: g^{(p^2-1)/(p+1)} = g^{p-1}
gen_subgroup = g_Fp2 ** (p - 1)
print(f"  Generator of mu_{{p+1}}: {gen_subgroup}, order check: {gen_subgroup.multiplicative_order()}")

# Construct factor base: x ∈ mu_{p+1} that yield E(F_{p^2}) points
print(f"\n  Building FB from x ∈ mu_{{p+1}} that lift to E(F_p^2)...")
fb = []
fb_x_set = set()
t0 = time.time()
for i in range(p + 1):
    x_val = gen_subgroup ** i
    pts = E_Fp2.lift_x(x_val, all=True)
    if pts:
        fb.append(pts[0])
        fb_x_set.add(x_val)
elapsed = time.time() - t0
print(f"  Built FB of size {len(fb)} in {elapsed:.2f}s")
print(f"  Fraction of mu_{{p+1}} on E: {float(len(fb))/float(p+1)*100:.1f}%")

# Set up ECDLP
P_Fp = E_Fp.random_point()
while P_Fp.order() != n: P_Fp = E_Fp.random_point()
secret = secrets.randbelow(int(n)-1) + 1
Q_Fp = secret * P_Fp

# Lift to F_p^2
P_Fp2 = E_Fp2(int(P_Fp[0]), int(P_Fp[1]))
Q_Fp2 = E_Fp2(int(Q_Fp[0]), int(Q_Fp[1]))

# Try pair-sum-style relation search
print(f"\n  Searching width-3 relations a*P + b*Q = F_i + F_j + F_k...")
pair_sums = {}
for i in range(len(fb)):
    for j in range(i+1, len(fb)):
        S = fb[i] + fb[j]
        if S == 0: continue
        pair_sums[(S[0], S[1])] = (i, j)
print(f"  Pair-sum table size: {len(pair_sums)}")

# Each candidate (a*P + b*Q - F_k) checked against pair_sums
relations_found = []
queries = 0
target_rels = 10
t0 = time.time()
while len(relations_found) < target_rels and time.time() - t0 < 60:
    a = secrets.randbelow(int(n))
    b = secrets.randbelow(int(n))
    if b == 0: continue
    R = a*P_Fp2 + b*Q_Fp2
    queries += 1
    if R == 0: continue
    for k_idx in range(len(fb)):
        T = R - fb[k_idx]
        if T == 0: continue
        if (T[0], T[1]) in pair_sums:
            i_idx, j_idx = pair_sums[(T[0], T[1])]
            relations_found.append((a, b, i_idx, j_idx, k_idx))
            break
elapsed = time.time() - t0
print(f"  Found {len(relations_found)} relations in {queries} queries, {elapsed:.2f}s")

if relations_found:
    print(f"\n  ★ Successfully found relations in E(F_p^2) with quasi-subfield FB.")
    print(f"  Sample relation: a={relations_found[0][0]}, b={relations_found[0][1]}")
    print(f"    fb[i]={fb[relations_found[0][2]]}")
    print(f"    fb[j]={fb[relations_found[0][3]]}")
    print(f"    fb[k]={fb[relations_found[0][4]]}")

# CRITICAL QUESTION: do these relations help recover the F_p discrete log?
# In F_{p^2}, P has order n (the F_p order). Q = k*P with k ∈ Z/nZ.
# A relation aP + bQ = F_i + F_j + F_k in E(F_{p^2}) gives:
#   a + b*k ≡ log_P(F_i) + log_P(F_j) + log_P(F_k) (mod ord(P) in F_p^2)
#
# But P has order n in E(F_p^2) too (since P ∈ E(F_p) and the group inherits).
# So this is a relation in Z/nZ — same group as F_p ECDLP!
#
# However, log_P(F_i) is the discrete log of F_i wrt P in E(F_p^2), which
# we don't know yet. We need to express F_i in terms of P somehow.
#
# Since F_i ∈ E(F_p^2) but not E(F_p) generically, log_P(F_i) is not a
# well-defined element of Z/nZ — it's in Z/ord(F_i)Z where ord(F_i)
# could be different.
#
# This is the SUBTLETY: factor base points in E(F_p^2) might not generate
# the same subgroup as P. If they do (i.e., F_i ∈ <P> ⊂ E(F_p^2)), then
# the relation directly gives F_p ECDLP. Otherwise, we need additional
# constraints to map F_i back to <P>.

print(f"\n  Checking if FB points are in <P>...")
P_order = P_Fp2.order()
print(f"  P has order {P_order} in E(F_p^2)")
in_subgroup = []
for i, F_i in enumerate(fb[:5]):
    F_order = F_i.order()
    # Check if F_i is in <P>: F_i.order() | P_order, and P_order * F_i = 0
    is_in = (P_order * F_i == 0)
    print(f"  FB[{i}]: order = {F_order}, F_order | n? {(F_order * P_Fp2) == 0}, in <P>? {is_in}")
    if is_in:
        in_subgroup.append(i)

if not in_subgroup:
    print(f"\n  ★ FB points are NOT in <P>. Relations in E(F_p^2) don't directly")
    print(f"  give F_p ECDLP — they involve discrete logs in a larger group.")
    print(f"  Would need additional structure (Weil pairing, trace map) to descend.")

print()
print("=" * 78)
print()
print("CONCLUSION:")
print("  Phase 18.2 status: quasi-subfield FB in F_p^2 is COMPUTABLE")
print("  and yields valid relations in E(F_p^2). HOWEVER:")
print()
print("  - Generically, FB points are NOT in <P> = E(F_p)-image subgroup")
print("  - Relations are in a larger group than n; don't directly give F_p ECDLP")
print("  - Descent via trace map would only work if the relations had specific")
print("    structure (this is the Weil-restriction / GHS setup, ruled out in")
print("    Phase 6 for prime-field-original curves)")
print()
print("  The quasi-subfield approach in artificial F_p^2 embedding does")
print("  NOT give a working attack on prime-field ECDLP for generic curves.")
print("  Consistent with Huang et al.'s observation that the construction")
print("  is for genuine extension-field-original curves, not artificially")
print("  embedded prime-field curves.")
