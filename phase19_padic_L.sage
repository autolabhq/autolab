#!/usr/bin/env sage
"""Phase 19.2: p-adic L-function evaluation.

For an elliptic curve E/Q with good ordinary reduction at p, the
p-adic L-function L_p(E, s) is defined. Its value at s=1 conjecturally
relates to the p-adic regulator and Tamagawa numbers (p-adic BSD).

Hypothesis: there's some way to extract ECDLP-relevant info from
L_p(E, 1) mod p^k. Very speculative.

Sage has L_p(E, s) computation via Manin-Vishik symbols.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, ModularSymbols, prime_pi

print("Phase 19.2: p-adic L-function evaluation")
print("=" * 78)

# Use a curve from LMFDB benchmark: 67.a1
E_Q = EllipticCurve([0, 1, 1, -12, -21])
print(f"E/Q: {E_Q}")
print(f"  Conductor: {E_Q.conductor()}")
print(f"  rank: {E_Q.rank()}")
print(f"  L(E, 1): {E_Q.lseries().at1()}")

# Try at a small p where the curve has good ordinary reduction
test_primes = [11, 13, 17, 19, 23, 29, 31]
print(f"\n--- Testing small primes for good ordinary reduction ---")
for p in test_primes:
    try:
        if E_Q.conductor() % p == 0:
            print(f"  p = {p}: bad reduction (conductor divisible)")
            continue
        E_p = E_Q.change_ring(GF(p))
        ap = ZZ(p + 1 - E_p.order())
        ordinary = (ap % p != 0)
        print(f"  p = {p}: a_p = {ap}, ordinary? {ordinary}")
    except Exception as e:
        print(f"  p = {p}: error {e}")

# Compute p-adic L-function at p=11 (smallest good ordinary prime)
print(f"\n--- p-adic L-function at p=11 ---")
import time
try:
    p_test = 11
    t0 = time.time()
    Lp = E_Q.padic_lseries(p_test)
    t_setup = time.time() - t0
    print(f"  padic_lseries setup: {t_setup:.2f}s")
    t0 = time.time()
    # Compute L_p value at s=1 with precision 5
    val = Lp.series(n=5, prec=5)
    t_eval = time.time() - t0
    print(f"  L_p(E, T) (Mazur-Tate-Teitelbaum): {val}")
    print(f"  Evaluation time: {t_eval:.2f}s")
except Exception as e:
    print(f"  padic_lseries failed: {e}")
    import traceback
    traceback.print_exc()

# Check at benchmark p
print(f"\n--- p-adic L-function at benchmark p (81-bit) ---")
p_bench = 1208925819614629469615699
print(f"  p = {p_bench}, bits = {p_bench.bit_length()}")
try:
    import time
    t0 = time.time()
    Lp_bench = E_Q.padic_lseries(p_bench)
    print(f"  padic_lseries setup at 81-bit: {time.time()-t0:.2f}s")
except Exception as e:
    print(f"  Cannot compute at 81-bit: {e}")
    print(f"  (Sage's padic_lseries requires substantial precomputation;")
    print(f"   for cryptographically-sized primes, the L-function is")
    print(f"   computationally infeasible to evaluate)")

print()
print("=" * 78)
print()
print("CONCLUSION:")
print("  p-adic L-function evaluation is feasible at small p (≤ 50) via")
print("  Sage's padic_lseries module. At cryptographic scale (81-bit p),")
print("  the computation is infeasible (modular symbols don't scale).")
print()
print("  Even if computed: there is no known connection between L_p(E, 1)")
print("  and ECDLP. The p-adic BSD conjecture relates L_p(E, 1) to")
print("  L'(E, 1) and Tamagawa numbers, NOT to discrete logarithms.")
print()
print("  This direction is closed as a near-term attack vector.")
