#!/usr/bin/env sage
"""Groebner-basis-style index calculus benchmark with summation polynomials.

For a fixed x_R = x(R) where R = αP + βQ, the Semaev F_3 equation
  F_3(x_R, X_2, X_3) = 0
defines an algebraic curve of bi-degree (2, 2) in (X_2, X_3). We
search for pairs (X_2, X_3) ∈ FB × FB satisfying this equation
(plus the geometric verification on E).

Three concrete techniques:
  A) Pair-sum hashing (Phase 1 baseline): O(|FB|) per query.
  B) Univariate root-finding (Semaev F_3 standard): for each X_2 in
     FB, solve F_3(x_R, X_2, X_3) = 0 as quadratic in X_3, check if
     any root is in FB. Cost O(|FB|) per query.
  C) Groebner basis on ideal <F_3(x_R, X_2, X_3), prod(X_2 - x_i),
     prod(X_3 - x_j)>: solve all factor-base constraints
     simultaneously. Cost depends on Groebner complexity.

We benchmark (B) and (C) and compare to (A).
"""
import time
import secrets
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, PolynomialRing, Ideal, sage_eval, set_random_seed

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

print("Groebner-basis-style index calculus benchmark")
print("=" * 78)
print()

for bits in [13, 16, 19, 22]:
    E, n, p = find_prime_order_curve(bits)
    if E is None:
        print(f"Could not find curve at {bits} bits")
        continue
    E_short = E
    a_coef = ZZ(E.a4())
    b_coef = ZZ(E.a6())
    print(f"\n--- bits = {bits}, p = {p}, n = {n} ---")

    # Build factor base
    Fp = GF(p)
    R3 = PolynomialRing(Fp, names=['X1', 'X2', 'X3'], order='lex')
    X1, X2, X3 = R3.gens()
    F3 = (X1-X2)**2*X3**2 - 2*((X1+X2)*(X1*X2+a_coef)+2*b_coef)*X3 \
        + (X1*X2-a_coef)**2 - 4*b_coef*(X1+X2)

    fb_size = min(50, max(8, int(float(n)**(1/3))))
    fb = []
    fb_x_set = set()
    for x in range(p):
        pts = E.lift_x(Fp(x), all=True)
        if pts:
            fb.append(pts[0])
            fb_x_set.add(int(x))
        if len(fb) >= fb_size: break
    print(f"  factor base size: {len(fb)}")

    # Random P, Q
    P = E.random_point()
    while P.order() != n: P = E.random_point()
    secret = secrets.randbelow(int(n)-1) + 1
    Q = secret * P

    # METHOD A: pair-sum hashing
    t0 = time.time()
    pair_sums = {}
    for i in range(len(fb)):
        for j in range(i+1, len(fb)):
            S = fb[i] + fb[j]
            if S == 0: continue
            pair_sums[(int(S[0]), int(S[1]))] = (i, j)
    setup_A = time.time() - t0

    relations_A = 0
    queries_A = 0
    target_rels = 5  # just demo
    t0 = time.time()
    while relations_A < target_rels and time.time()-t0 < 5:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        queries_A += 1
        if R == 0: continue
        for k_idx in range(len(fb)):
            T = R - fb[k_idx]
            if T == 0: continue
            if (int(T[0]), int(T[1])) in pair_sums:
                relations_A += 1
                break
    method_A_time = time.time() - t0
    print(f"  Method A (pair-sum): {relations_A} relations, {queries_A} queries, {method_A_time:.2f}s, setup {setup_A:.3f}s")

    # METHOD B: F_3 quadratic root-finding for each x_2 in FB
    R2 = PolynomialRing(Fp, names=['X3'])
    X3_uni = R2.gen()
    relations_B = 0
    queries_B = 0
    t0 = time.time()
    while relations_B < target_rels and time.time()-t0 < 30:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        queries_B += 1
        if R == 0: continue
        xR = int(R[0])
        # Iterate over factor base candidates X_2
        found = False
        for i in range(len(fb)):
            x_i = int(fb[i][0])
            # F_3(xR, x_i, X3) as quadratic in X3
            # Use the explicit formula:
            coef_X3_sq = (xR - x_i)**2 % p
            coef_X3 = (-2*((xR+x_i)*(xR*x_i+a_coef) + 2*b_coef)) % p
            coef_const = ((xR*x_i - a_coef)**2 - 4*b_coef*(xR+x_i)) % p
            poly = coef_X3_sq * X3_uni**2 + coef_X3 * X3_uni + coef_const
            roots = poly.roots()
            for x_3_root, _mult in roots:
                if int(x_3_root) in fb_x_set:
                    relations_B += 1
                    found = True
                    break
            if found: break
    method_B_time = time.time() - t0
    print(f"  Method B (F_3 root): {relations_B} relations, {queries_B} queries, {method_B_time:.2f}s")

    # METHOD C: Groebner basis on ideal
    # I = <F_3(xR, X2, X3), prod_{i}(X2 - x_i), prod_{j}(X3 - x_j)>
    # Solving I gives all (X2, X3) ∈ FB^2 satisfying F_3 = 0
    R_C = PolynomialRing(Fp, names=['X2', 'X3'], order='lex')
    X2c, X3c = R_C.gens()
    # Build pair-of-FB-x product polynomials
    fb_xs = [int(pt[0]) for pt in fb]
    P_fb_X2 = R_C(1)
    P_fb_X3 = R_C(1)
    for x in fb_xs:
        P_fb_X2 *= (X2c - x)
        P_fb_X3 *= (X3c - x)
    relations_C = 0
    queries_C = 0
    groebner_time_total = 0.0
    t0 = time.time()
    while relations_C < target_rels and time.time()-t0 < 30:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        queries_C += 1
        if R == 0: continue
        xR = int(R[0])
        # F_3(xR, X2, X3)
        f3_specialized = (xR - X2c)**2 * X3c**2 \
                       - 2*((xR + X2c)*(xR*X2c + a_coef) + 2*b_coef)*X3c \
                       + (xR*X2c - a_coef)**2 - 4*b_coef*(xR + X2c)
        I = Ideal([f3_specialized, P_fb_X2, P_fb_X3])
        gb_t0 = time.time()
        try:
            gb = I.groebner_basis()
            groebner_time_total += time.time() - gb_t0
            # Solve the ideal (look for variety)
            V = I.variety()
            for sol in V:
                x2_val = int(sol[X2c])
                x3_val = int(sol[X3c])
                if x2_val in fb_x_set and x3_val in fb_x_set:
                    relations_C += 1
                    break
        except Exception as e:
            groebner_time_total += time.time() - gb_t0
            pass
        if queries_C > 20:  # cap to bound runtime
            break
    method_C_time = time.time() - t0
    print(f"  Method C (Groebner): {relations_C} relations, {queries_C} queries, {method_C_time:.2f}s, GB time {groebner_time_total:.2f}s")
    print(f"  Average GB per query: {groebner_time_total/max(queries_C, 1):.4f}s")

print()
print("=" * 78)
print("Interpretation:")
print("  Method A (hash): O(|FB|) per query, fastest in practice")
print("  Method B (root): O(|FB|) per query (same as A asymptotically)")
print("  Method C (GB):   per-query Groebner basis solve")
print()
print("If Method C is competitive or faster, it suggests Groebner-basis")
print("subroutines could form the basis of a new ECDLP attack pipeline.")
print("If Method C is much slower (typical), Method A/B remain the practical choice.")
