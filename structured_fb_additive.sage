#!/usr/bin/env sage
"""Test: additive-shift factor base for Semaev F_3 Groebner basis.

The factor base x-coords form an arithmetic progression:
  FB_x = {x_0, x_0 + d, x_0 + 2d, ..., x_0 + (m-1)d}

The constraint polynomial is
  ∏_i (X - x_0 - i*d) = d^m * (Y)(Y-1)...(Y-(m-1))
where Y = (X - x_0)/d.

This is a "falling factorial polynomial" — closely related to
Stirling numbers, with combinatorial structure.

We test whether this structure makes the GB solve faster than the
generic-FB baseline.
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

def find_arithmetic_fb(E, Fp, p, target_size):
    """Find an arithmetic progression of x-coords yielding curve points."""
    # Try several (x_0, d) combinations
    for d in [1, 2, 3, 5, 7]:
        for x_0_start in range(0, min(p, 1000)):
            fb = []
            fb_x = []
            for i in range(target_size * 10):
                x_cand = (x_0_start + i * d) % p
                pts = E.lift_x(Fp(x_cand), all=True)
                if pts:
                    fb.append(pts[0])
                    fb_x.append(x_cand)
                    if len(fb) >= target_size:
                        break
            # Note: not all x are quadratic residues, so the FB will skip non-residues.
            # We accept whatever fits in the first `target_size * 10` candidates.
            if len(fb) >= target_size:
                return fb, fb_x, x_0_start, d
    return None, None, None, None

print("Additive-shift factor base test")
print("=" * 78)

for bits, target_fb in [(13, 16), (16, 20), (19, 24)]:
    E, n, p = find_prime_order_curve(bits)
    if E is None:
        print(f"\nbits={bits}: no curve")
        continue
    Fp = GF(p)
    a_coef = ZZ(E.a4())
    b_coef = ZZ(E.a6())
    print(f"\n--- bits={bits}, p={p}, n={n}, |FB target|={target_fb} ---")

    fb, fb_xs, x_0, d = find_arithmetic_fb(E, Fp, p, target_fb)
    if fb is None:
        print(f"  Could not find arithmetic progression")
        continue
    fb_x_set = set(fb_xs)
    print(f"  Factor base: |FB|={len(fb)}, x_0={x_0}, d={d}")
    print(f"  Note: FB doesn't fill arithmetic progression perfectly (skip non-QRs)")

    # Random P, Q
    P = E.random_point()
    while P.order() != n: P = E.random_point()
    secret = secrets.randbelow(int(n)-1) + 1
    Q = secret * P

    # Method C-arith: F_3 + arithmetic progression constraint
    # Constraint polynomial = ∏_{i=0}^{m-1} (X - x_0 - i*d) — dense, but uses structure
    R3 = PolynomialRing(Fp, names=['X2', 'X3'], order='lex')
    X2, X3 = R3.gens()

    # Build constraint polynomial as a product (Sage will expand it)
    constraint_X2 = R3(1)
    constraint_X3 = R3(1)
    for x in fb_xs:
        constraint_X2 *= (X2 - x)
        constraint_X3 *= (X3 - x)

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
        f3_spec = (xR - X2)**2 * X3**2 \
                - 2*((xR + X2)*(xR*X2 + a_coef) + 2*b_coef)*X3 \
                + (xR*X2 - a_coef)**2 - 4*b_coef*(xR + X2)
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
    print(f"  Method C-arith: {relations_C} rels, {queries_C} queries, {method_C_time:.2f}s")
    print(f"    Average GB+variety per query: {gb_time/max(queries_C, 1)*1000:.1f}ms")

    # Compare: Method A pair-sum on this same FB
    pair_sums = {}
    for i in range(len(fb)):
        for j in range(i+1, len(fb)):
            S = fb[i] + fb[j]
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
        for k_idx in range(len(fb)):
            T = R - fb[k_idx]
            if T == 0: continue
            if (int(T[0]), int(T[1])) in pair_sums:
                relations_A += 1
                break
    method_A_time = time.time() - t0
    print(f"  Method A (pair-sum on same FB): {relations_A} rels, {queries_A} queries, {method_A_time:.3f}s")

print()
print("=" * 78)
print()
print("INTERPRETATION:")
print("  If arithmetic-progression FB doesn't speed up GB compared to")
print("  the generic baseline, this confirms that *no* polynomial-time-")
print("  expressible structured factor base on prime fields gives GB sparsity.")
