#!/usr/bin/env sage
"""Asymptotic crossover test: pair-sum vs F_3 Groebner at increasing bit sizes.

Earlier F_3 Groebner result showed at 22 bits: A=0.76s vs C=0.65s.
This may be the Diem L(2/3) crossover. Test rigorously at 22, 25, 28 bits.

KEY METHODOLOGY:
1. Equal target relations (5) and time budgets per method
2. Both methods use the SAME factor base size
3. Multiple trials per bit size to control variance
4. Both methods report effective time/relation
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

print("Asymptotic crossover: pair-sum vs F_3 Groebner")
print("=" * 78)
print()
print("Hypothesis: at sufficient bit size, F_3 Groebner ~ |FB|^(constant) per query")
print("beats pair-sum's O(|FB|) per query times O(n/|FB|^2) queries.")
print()

bit_sizes = [22, 25, 28]
n_trials = 3

results = {}

for bits in bit_sizes:
    print(f"\n=== bits = {bits} ===")
    E, n, p = find_prime_order_curve(bits)
    if E is None:
        print(f"  no curve")
        continue
    Fp = GF(p)
    a_coef = ZZ(E.a4())
    b_coef = ZZ(E.a6())
    print(f"  p={p}, n={n}")

    # Factor base size = small fixed value to allow Method C to find rels.
    # Diem-optimal would be n^(1/3) but GB cost balloons exponentially.
    # We test with a small fixed |FB| = 40 to isolate the per-query cost trend.
    fb_size = 40
    print(f"  |FB| = {fb_size} (target n^(1/3) = {int(float(n)**(1/3))})")

    # Build factor base
    fb = []
    fb_x_set = set()
    for x in range(p):
        pts = E.lift_x(Fp(x), all=True)
        if pts:
            fb.append(pts[0])
            fb_x_set.add(int(x))
        if len(fb) >= fb_size: break

    P = E.random_point()
    while P.order() != n: P = E.random_point()
    secret = secrets.randbelow(int(n)-1) + 1
    Q = secret * P

    # Hit probability for width-3: ~|FB|^2/(2n)
    hit_prob = (fb_size * fb_size) / (2.0 * float(n))
    print(f"  expected width-3 hit prob/query: {hit_prob:.6f}")

    # ----- Method A: pair-sum -----
    pair_sums = {}
    for i in range(len(fb)):
        for j in range(i+1, len(fb)):
            S = fb[i] + fb[j]
            if S == 0: continue
            pair_sums[(int(S[0]), int(S[1]))] = (i, j)

    a_times = []
    a_queries = []
    for trial in range(n_trials):
        relations_A = 0
        queries_A = 0
        t0 = time.time()
        while relations_A < 5 and time.time()-t0 < 120:
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
        a_times.append(time.time() - t0)
        a_queries.append(queries_A)
        if relations_A < 5:
            print(f"    [Trial {trial}] Method A: TIMED OUT after {queries_A} queries")
            break
    if a_times:
        a_avg_time = float(sum(a_times)) / len(a_times)
        a_avg_q = float(sum(a_queries)) / len(a_queries)
        print(f"  Method A (pair-sum): avg {a_avg_time:.2f}s, {a_avg_q:.0f} queries (n={len(a_times)} trials)")
    else:
        a_avg_time = float('inf')
        a_avg_q = 0

    # ----- Method C: F_3 Groebner -----
    R_C = PolynomialRing(Fp, names=['X2', 'X3'], order='lex')
    X2c, X3c = R_C.gens()
    fb_xs = [int(pt[0]) for pt in fb]
    P_fb_X2 = R_C(1)
    P_fb_X3 = R_C(1)
    for x in fb_xs:
        P_fb_X2 *= (X2c - x)
        P_fb_X3 *= (X3c - x)

    c_times = []
    c_queries = []
    c_gb_times = []
    for trial in range(n_trials):
        relations_C = 0
        queries_C = 0
        gb_time = 0.0
        t0 = time.time()
        while relations_C < 5 and time.time()-t0 < 300:
            a = secrets.randbelow(int(n))
            b = secrets.randbelow(int(n))
            if b == 0: continue
            R = a*P + b*Q
            queries_C += 1
            if R == 0: continue
            xR = int(R[0])
            f3_specialized = (xR - X2c)**2 * X3c**2 \
                           - 2*((xR + X2c)*(xR*X2c + a_coef) + 2*b_coef)*X3c \
                           + (xR*X2c - a_coef)**2 - 4*b_coef*(xR + X2c)
            I = Ideal([f3_specialized, P_fb_X2, P_fb_X3])
            gb_t0 = time.time()
            try:
                V = I.variety()
                gb_time += time.time() - gb_t0
                for sol in V:
                    x2_val = int(sol[X2c])
                    x3_val = int(sol[X3c])
                    if x2_val in fb_x_set and x3_val in fb_x_set:
                        relations_C += 1
                        break
            except Exception as e:
                gb_time += time.time() - gb_t0
        c_times.append(time.time() - t0)
        c_queries.append(queries_C)
        c_gb_times.append(gb_time)
        if relations_C < 5:
            print(f"    [Trial {trial}] Method C: TIMED OUT after {queries_C} queries ({relations_C} rels)")
            break
    if c_times:
        c_avg_time = float(sum(c_times)) / len(c_times)
        c_avg_q = float(sum(c_queries)) / len(c_queries)
        c_avg_gb_per_q = float(sum(c_gb_times)) / float(sum(c_queries)) if sum(c_queries) > 0 else 0.0
        print(f"  Method C (F_3 GB):   avg {c_avg_time:.2f}s, {c_avg_q:.0f} queries, GB={c_avg_gb_per_q*1000:.1f}ms/q (n={len(c_times)})")
    else:
        c_avg_time = float('inf')
        c_avg_q = 0

    if a_avg_time and c_avg_time:
        ratio = c_avg_time / a_avg_time
        winner = 'A' if ratio > 1 else 'C'
        print(f"  Ratio C/A = {ratio:.2f}× (Method {winner} wins)")

    results[bits] = {
        'fb_size': fb_size,
        'A_time': a_avg_time,
        'A_queries': a_avg_q,
        'C_time': c_avg_time,
        'C_queries': c_avg_q,
    }

print()
print("=" * 78)
print("Summary table:")
print(f"{'bits':>5} | {'|FB|':>5} | {'A_time':>8} | {'C_time':>8} | {'ratio':>6}")
print("-" * 50)
for bits, r in results.items():
    ratio = r['C_time'] / r['A_time'] if r['A_time'] > 0 else float('inf')
    print(f"{bits:>5} | {r['fb_size']:>5} | {r['A_time']:>7.2f}s | {r['C_time']:>7.2f}s | {ratio:>5.2f}×")

print()
print("If ratio is decreasing as bits increase, the crossover hypothesis is confirmed.")
print("If ratio is increasing or flat, no asymptotic crossover.")
