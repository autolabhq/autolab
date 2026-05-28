#!/usr/bin/env sage
"""Parametric Gröbner basis for Semaev F_3.

For a FIXED factor base FB, the ideal
  I(xR) = ⟨F_3(xR, X_2, X_3), ∏(X_2 - x_i), ∏(X_3 - x_i)⟩
is parametrized by xR. The constraints don't change; only F_3
specialization changes per query.

We compute the parametric GB in F_p[xR][X_2, X_3] ONCE, then per
query substitute xR=concrete-value and read off the variety.

If this works, per-query cost is just polynomial evaluation +
root finding — much faster than running F4/F5 per query.

If parametric GB is too expensive to compute, fall back to
characteristic-set decomposition.
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

print("Parametric Gröbner basis approach")
print("=" * 78)

bits = 13
E, n, p = find_prime_order_curve(bits)
Fp = GF(p)
a_coef = ZZ(E.a4())
b_coef = ZZ(E.a6())
print(f"--- bits={bits}, p={p}, n={n} ---")

# Small fixed FB
fb_size = 12
fb = []
fb_x_set = set()
for x in range(p):
    pts = E.lift_x(Fp(x), all=True)
    if pts:
        fb.append(pts[0])
        fb_x_set.add(int(x))
    if len(fb) >= fb_size: break
fb_xs = [int(pt[0]) for pt in fb]
print(f"|FB| = {len(fb)}, x-coords: {fb_xs}")

P = E.random_point()
while P.order() != n: P = E.random_point()
secret = secrets.randbelow(int(n)-1) + 1
Q = secret * P

# ---- Parametric GB computation ----
# Work in F_p[xR, X_2, X_3] with xR as the "first" variable
# We want GB of I = <F_3(xR, X_2, X_3), prod(X_2 - x_i), prod(X_3 - x_i)>
print()
print("=== Approach 1: Sage GB in F_p[xR, X_2, X_3] with lex order (xR < X_3 < X_2) ===")
print()
print("Build the parametric GB once, hope it has manageable size...")

R_param = PolynomialRing(Fp, names=['xR', 'X2', 'X3'], order='lex')
xR_var, X2, X3 = R_param.gens()
F3_param = (xR_var - X2)**2 * X3**2 \
         - 2*((xR_var + X2)*(xR_var*X2 + a_coef) + 2*b_coef)*X3 \
         + (xR_var*X2 - a_coef)**2 - 4*b_coef*(xR_var + X2)

constraint_X2 = R_param(1)
constraint_X3 = R_param(1)
for x in fb_xs:
    constraint_X2 *= (X2 - x)
    constraint_X3 *= (X3 - x)

print(f"F_3 generator: degree {F3_param.total_degree()}, # monomials {len(F3_param.monomials())}")
print(f"X2 constraint: degree {constraint_X2.total_degree()}, # monomials {len(constraint_X2.monomials())}")

I_param = Ideal([F3_param, constraint_X2, constraint_X3])
print()
print("Computing parametric GB...")
t0 = time.time()
try:
    gb_param = I_param.groebner_basis()
    gb_time = time.time() - t0
    print(f"Parametric GB computation time: {gb_time:.2f}s")
    print(f"GB has {len(gb_param)} elements")
    for i, g in enumerate(gb_param):
        deg_xR = g.degree(xR_var)
        deg_X2 = g.degree(X2)
        deg_X3 = g.degree(X3)
        n_mons = len(g.monomials())
        print(f"  GB[{i}]: deg(xR)={deg_xR}, deg(X2)={deg_X2}, deg(X3)={deg_X3}, monomials={n_mons}")
except Exception as e:
    print(f"Parametric GB failed: {e}")
    gb_param = None

# ---- Approach 2: Evaluate parametric GB per query ----
if gb_param is not None:
    print()
    print("=== Approach 2: per-query, substitute xR and find roots ===")
    print()
    relations_pgb = 0
    queries_pgb = 0
    eval_time_total = 0.0
    t0_method = time.time()
    target_rels = 5
    R_eval = PolynomialRing(Fp, names=['X2', 'X3'], order='lex')
    X2e, X3e = R_eval.gens()
    while relations_pgb < target_rels and time.time()-t0_method < 60 and queries_pgb < 30:
        a = secrets.randbelow(int(n))
        b = secrets.randbelow(int(n))
        if b == 0: continue
        R = a*P + b*Q
        queries_pgb += 1
        if R == 0: continue
        xR_val = int(R[0])
        # Evaluate each GB element at xR = xR_val
        eval_t0 = time.time()
        evaluated_gb = []
        for g in gb_param:
            # Substitute xR_var = xR_val into g
            g_dict = g.dict()
            ge = R_eval.zero()
            for mon, coef in g_dict.items():
                e_xR, e_X2, e_X3 = mon
                term = coef * (xR_val**e_xR) * X2e**e_X2 * X3e**e_X3
                ge += term
            if ge != 0:
                evaluated_gb.append(ge)
        # Solve evaluated GB
        I_eval = Ideal(evaluated_gb)
        try:
            V = I_eval.variety()
            eval_time_total += time.time() - eval_t0
            for sol in V:
                x2_val = int(sol[X2e])
                x3_val = int(sol[X3e])
                if x2_val in fb_x_set and x3_val in fb_x_set:
                    relations_pgb += 1
                    break
        except Exception as e:
            eval_time_total += time.time() - eval_t0
    method_pgb_time = time.time() - t0_method
    print(f"Parametric GB method: {relations_pgb} rels, {queries_pgb} queries, {method_pgb_time:.2f}s")
    if queries_pgb > 0:
        print(f"  Per-query evaluate+solve: {eval_time_total/queries_pgb*1000:.2f}ms")

# ---- Compare: per-query fresh GB (Method C baseline) ----
print()
print("=== Baseline: fresh GB per query ===")
print()
R2 = PolynomialRing(Fp, names=['X2', 'X3'], order='lex')
X2b, X3b = R2.gens()
constraint_X2_b = R2(1)
constraint_X3_b = R2(1)
for x in fb_xs:
    constraint_X2_b *= (X2b - x)
    constraint_X3_b *= (X3b - x)

relations_C = 0
queries_C = 0
gb_time_total = 0.0
t0_C = time.time()
while relations_C < 5 and time.time()-t0_C < 60 and queries_C < 30:
    a = secrets.randbelow(int(n))
    b = secrets.randbelow(int(n))
    if b == 0: continue
    R = a*P + b*Q
    queries_C += 1
    if R == 0: continue
    xR_val = int(R[0])
    f3_spec = (xR_val - X2b)**2 * X3b**2 \
            - 2*((xR_val + X2b)*(xR_val*X2b + a_coef) + 2*b_coef)*X3b \
            + (xR_val*X2b - a_coef)**2 - 4*b_coef*(xR_val + X2b)
    I_C = Ideal([f3_spec, constraint_X2_b, constraint_X3_b])
    gb_t0 = time.time()
    try:
        V = I_C.variety()
        gb_time_total += time.time() - gb_t0
        for sol in V:
            x2_val = int(sol[X2b])
            x3_val = int(sol[X3b])
            if x2_val in fb_x_set and x3_val in fb_x_set:
                relations_C += 1
                break
    except Exception as e:
        gb_time_total += time.time() - gb_t0
print(f"Per-query GB: {relations_C} rels, {queries_C} queries, {time.time()-t0_C:.2f}s")
if queries_C > 0:
    print(f"  Per-query GB+variety: {gb_time_total/queries_C*1000:.2f}ms")

print()
print("=" * 78)
print()
print("INTERPRETATION:")
print("If parametric GB is faster, the precomputation amortizes across queries.")
print("If similar speed, the parametric GB doesn't fundamentally simplify the problem.")
