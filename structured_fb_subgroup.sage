#!/usr/bin/env sage
"""Test: multiplicative-subgroup factor base for Semaev F_3/F_4 Groebner basis.

Key insight: if factor base x-coords are constrained to a multiplicative
subgroup H ⊂ F_p^* of order m, then the constraint polynomial is

  ∏_{h ∈ H} (X - h) = X^m - 1

which is SPARSE (only 2 monomials) instead of dense (m+1 monomials).

This dramatically simplifies the Groebner basis computation, potentially
making the Semaev attack practical for prime fields.

We test:
  - F_3 Groebner with subgroup constraints
  - F_4 Groebner with subgroup constraints (the Phase 9'' attempt)

Compare to:
  - Standard pair-sum (small-x baseline)
  - Generic-FB Groebner (Phase 16 result)
"""
import time
import secrets
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, PolynomialRing, Ideal, set_random_seed

set_random_seed(int(42))

def find_prime_order_curve(bits, m_subgroup):
    """Find p with prime-order curve, AND p-1 divisible by m_subgroup."""
    for offset in range(20000):
        p = (1 << bits) + 13 + 2*offset
        if not ZZ(p).is_prime(): continue
        if (p - 1) % m_subgroup != 0: continue  # ensure subgroup of order m exists
        E = EllipticCurve(GF(p), [3, 5])
        n = ZZ(E.order())
        if n.is_prime() and n > p // 2:
            return E, n, p
    return None, None, None

def multiplicative_subgroup_x_values(p, m):
    """Return the multiplicative subgroup of F_p^* of order m (as set of x-values)."""
    Fp = GF(p)
    g = Fp.multiplicative_generator()
    h = g ** ((p - 1) // m)  # generator of order-m subgroup
    return [int(h ** i) for i in range(m)]

print("Structured factor base via multiplicative subgroup of F_p^*")
print("=" * 78)

for bits, m_target in [(13, 16), (16, 32), (19, 64)]:
    E, n, p = find_prime_order_curve(bits, m_target)
    if E is None:
        print(f"\nbits={bits}: no curve with subgroup order {m_target} found")
        continue
    Fp = GF(p)
    a_coef = ZZ(E.a4())
    b_coef = ZZ(E.a6())
    print(f"\n--- bits={bits}, p={p}, n={n} ---")

    # Multiplicative subgroup of order m
    H_x = multiplicative_subgroup_x_values(p, m_target)
    print(f"  Multiplicative subgroup H ⊂ F_p^* of order {len(H_x)}")

    # Find which H_x values yield curve points
    fb = []
    fb_x_set = set()
    for x in H_x:
        pts = E.lift_x(Fp(x), all=True)
        if pts:
            fb.append(pts[0])
            fb_x_set.add(int(x))
    print(f"  Factor base size (H ∩ E_x): {len(fb)}")
    if len(fb) < 4:
        print(f"  Too small; skipping")
        continue

    # Random P, Q
    P = E.random_point()
    while P.order() != n: P = E.random_point()
    secret = secrets.randbelow(int(n)-1) + 1
    Q = secret * P

    # Method C-sparse: F_3 Groebner with sparse constraint polynomials
    # I = <F_3(xR, X2, X3), X2^m - 1, X3^m - 1>
    # Only solutions with X2, X3 ∈ H are returned; we then filter for E-membership.
    R3 = PolynomialRing(Fp, names=['X2', 'X3'], order='lex')
    X2, X3 = R3.gens()

    relations_C = 0
    queries_C = 0
    gb_time = 0.0
    t0 = time.time()
    target_rels = 5
    while relations_C < target_rels and time.time()-t0 < 60 and queries_C < 30:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        queries_C += 1
        if R == 0: continue
        xR = int(R[0])
        # F_3(xR, X2, X3)
        f3_spec = (xR - X2)**2 * X3**2 \
                - 2*((xR + X2)*(xR*X2 + a_coef) + 2*b_coef)*X3 \
                + (xR*X2 - a_coef)**2 - 4*b_coef*(xR + X2)
        # Sparse constraints
        constraint_X2 = X2 ** m_target - 1
        constraint_X3 = X3 ** m_target - 1
        I = Ideal([f3_spec, constraint_X2, constraint_X3])
        gb_t0 = time.time()
        try:
            V = I.variety()
            gb_time += time.time() - gb_t0
            for sol in V:
                x2_val = int(sol[X2])
                x3_val = int(sol[X3])
                if x2_val in fb_x_set and x3_val in fb_x_set:
                    relations_C += 1
                    break
        except Exception as e:
            gb_time += time.time() - gb_t0
    method_C_time = time.time() - t0
    print(f"  Method C-sparse (F_3 + X^m): {relations_C} rels, {queries_C} queries, {method_C_time:.2f}s")
    print(f"    Average GB+variety per query: {gb_time/max(queries_C, 1)*1000:.1f}ms")

    # Compare to standard small-x factor base of similar size
    fb_size_small = len(fb)
    fb_small = []
    fb_small_x = set()
    for x in range(p):
        pts = E.lift_x(Fp(x), all=True)
        if pts:
            fb_small.append(pts[0])
            fb_small_x.add(int(x))
        if len(fb_small) >= fb_size_small: break
    # Method A: pair-sum hash with small-x factor base
    pair_sums = {}
    for i in range(len(fb_small)):
        for j in range(i+1, len(fb_small)):
            S = fb_small[i] + fb_small[j]
            if S == 0: continue
            pair_sums[(int(S[0]), int(S[1]))] = (i, j)

    relations_A = 0
    queries_A = 0
    t0 = time.time()
    while relations_A < target_rels and time.time()-t0 < 10:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        queries_A += 1
        if R == 0: continue
        for k_idx in range(len(fb_small)):
            T = R - fb_small[k_idx]
            if T == 0: continue
            if (int(T[0]), int(T[1])) in pair_sums:
                relations_A += 1
                break
    method_A_time = time.time() - t0
    print(f"  Method A (small-x pair-sum, baseline): {relations_A} rels, {queries_A} queries, {method_A_time:.3f}s")

print()
print("=" * 78)
print()
print("KEY QUESTION: does the multiplicative-subgroup constraint make")
print("Groebner basis tractable enough to beat pair-sum?")
print()
print("If 'Average GB per query' for Method C-sparse is much smaller than")
print("the previous F_3 Groebner result (~30ms for |FB|=50 generic FB),")
print("then the sparse constraint is helping. Otherwise, the algebraic")
print("structure of H doesn't propagate through the F_3 polynomial.")
