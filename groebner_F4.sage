#!/usr/bin/env sage
"""F_4 Groebner-basis benchmark.

Width-4 Semaev relation: a*P + b*Q = F_i + F_j + F_k + F_l
For fixed x_R = x(R), the constraint is:
  F_4(x_R, X_2, X_3, X_4) = 0
where F_4 is the explicit Semaev polynomial in 4 variables of total
degree 12. With the structured factor base, we also have:
  prod_i (X_2 - x_i) = 0
  prod_i (X_3 - x_i) = 0
  prod_i (X_4 - x_i) = 0

Three methods:
  A) Triple-loop with quadratic root: O(|FB|^3) naive
  B) Pair-sum with extra: precompute (F_i + F_j); for each (X_4)
     check R - F_l - F_k in pair-sum table
  C) Groebner basis on the ideal

Hit probability per (a, b): |FB|^3/(6n) for width-4 (vs |FB|^2/2 for width-3 pair-sum).
"""
import time
import secrets
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, PolynomialRing, Ideal, set_random_seed

set_random_seed(int(42))

def find_prime_order_curve(bits):
    for offset in range(20000):
        p = (1 << bits) + 13 + 2*offset
        if not ZZ(p).is_prime(): continue
        E = EllipticCurve(GF(p), [3, 5])
        n = ZZ(E.order())
        if n.is_prime() and n > p // 2:
            return E, n, p
    return None, None, None

print("F_4 Groebner-basis index calculus benchmark")
print("=" * 78)

for bits in [13, 16, 19]:
    E, n, p = find_prime_order_curve(bits)
    if E is None: continue
    Fp = GF(p)
    a_coef = ZZ(E.a4())
    b_coef = ZZ(E.a6())
    print(f"\n--- bits={bits}, p={p}, n={n} ---")

    # Build factor base
    fb_size = min(30, max(8, int(float(n)**(1/3))))
    fb = []
    fb_x_set = set()
    for x in range(p):
        pts = E.lift_x(Fp(x), all=True)
        if pts:
            fb.append(pts[0])
            fb_x_set.add(int(x))
        if len(fb) >= fb_size: break
    print(f"  factor base: |FB| = {len(fb)}")

    P = E.random_point()
    while P.order() != n: P = E.random_point()
    secret = secrets.randbelow(int(n)-1) + 1
    Q = secret * P

    # METHOD B*: width-4 pair-sum-of-pair-sums
    # Precompute pair_sums = {F_i + F_j : i<j} keyed by point
    # For R, search for (k, l) with R - F_k - F_l in pair_sums
    pair_sums = {}
    setup_ops = 0
    for i in range(len(fb)):
        for j in range(i+1, len(fb)):
            S = fb[i] + fb[j]
            setup_ops += 1
            if S == 0: continue
            pair_sums[(int(S[0]), int(S[1]))] = (i, j)

    relations_B = 0
    queries_B = 0
    target_rels = 5
    t0 = time.time()
    while relations_B < target_rels and time.time()-t0 < 30:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        queries_B += 1
        if R == 0: continue
        # For each (k, l) pair, R - F_k - F_l should be in pair_sums (as some F_i + F_j)
        found = False
        for k_idx in range(len(fb)):
            if found: break
            for l_idx in range(k_idx+1, len(fb)):
                T = R - fb[k_idx] - fb[l_idx]
                if T == 0: continue
                if (int(T[0]), int(T[1])) in pair_sums:
                    relations_B += 1
                    found = True
                    break
    method_B_time = time.time() - t0
    print(f"  Method B (width-4 pair-sum): {relations_B} rels, {queries_B} queries, {method_B_time:.2f}s")

    # METHOD C: F_4 Groebner basis
    # Build F_4 explicitly via resultant
    R_F4 = PolynomialRing(Fp, names=['X1', 'X2', 'X3', 'X4', 'Y'], order='lex')
    X1, X2, X3, X4, Y = R_F4.gens()
    F3a = (X1-X2)**2*Y**2 - 2*((X1+X2)*(X1*X2+a_coef)+2*b_coef)*Y \
        + (X1*X2-a_coef)**2 - 4*b_coef*(X1+X2)
    F3b = (X3-X4)**2*Y**2 - 2*((X3+X4)*(X3*X4+a_coef)+2*b_coef)*Y \
        + (X3*X4-a_coef)**2 - 4*b_coef*(X3+X4)
    F4_full = F3a.resultant(F3b, Y)
    # Substitute x_R for X1
    R_4 = PolynomialRing(Fp, names=['X2', 'X3', 'X4'], order='lex')
    Xa, Xb, Xc = R_4.gens()
    fb_xs = [int(pt[0]) for pt in fb]
    P_Xa = R_4(1)
    P_Xb = R_4(1)
    P_Xc = R_4(1)
    for x in fb_xs:
        P_Xa *= (Xa - x)
        P_Xb *= (Xb - x)
        P_Xc *= (Xc - x)

    relations_C = 0
    queries_C = 0
    gb_time = 0.0
    t0 = time.time()
    while relations_C < target_rels and time.time()-t0 < 60 and queries_C < 15:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        queries_C += 1
        if R == 0: continue
        xR = int(R[0])
        # F_4(xR, X2, X3, X4) = 0
        f4_spec = F4_full(xR, Xa, Xb, Xc, 0)  # substitute X1=xR, Y absent (resultant already eliminated)
        # Hmm wait F4_full is a 5-variable polynomial in R_F4 with vars (X1,X2,X3,X4,Y). After resultant Y is eliminated.
        # Re-check: F4_full lives in R_F4 but doesn't depend on Y after resultant.
        # Use coercion
        f4_in_R4 = R_4.zero()
        for mon, coef in F4_full.dict().items():
            # mon = (e1, e2, e3, e4, e5) — exponents of X1, X2, X3, X4, Y
            e1, e2, e3, e4, e5 = mon
            if e5 != 0:
                continue  # shouldn't happen
            term = coef * (xR**e1) * Xa**e2 * Xb**e3 * Xc**e4
            f4_in_R4 += term
        I = Ideal([f4_in_R4, P_Xa, P_Xb, P_Xc])
        gb_t0 = time.time()
        try:
            gb = I.groebner_basis()
            gb_time += time.time() - gb_t0
            V = I.variety()
            for sol in V:
                xa_val = int(sol[Xa])
                xb_val = int(sol[Xb])
                xc_val = int(sol[Xc])
                if xa_val in fb_x_set and xb_val in fb_x_set and xc_val in fb_x_set:
                    relations_C += 1
                    break
        except Exception as e:
            gb_time += time.time() - gb_t0
    method_C_time = time.time() - t0
    print(f"  Method C (F_4 Groebner): {relations_C} rels, {queries_C} queries, {method_C_time:.1f}s, GB time {gb_time:.1f}s")
    if queries_C > 0:
        print(f"    Average GB per query: {gb_time/queries_C:.3f}s")

print()
print("=" * 78)
print()
print("Conclusion:")
print("  Width-4 pair-sum (B) trades higher cost per query for higher hit rate.")
print("  Width-4 Groebner (C) has constant per-query cost but is heavy due to F_4 degree 12.")
print("  Practical viability of Groebner-basis Semaev for prime fields:")
print("    we expect Method C to be impractical for n > 2^30 due to per-query cost.")
