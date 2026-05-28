#!/usr/bin/env sage
"""Phase 21.1 validation: Pollard rho scaling at 30-60 bits.

Measures actual Pollard rho wall time at increasing bit sizes,
fits an exponential model, and projects to 80 bits. Then compares
to the C-extension projected throughput.
"""
import time
import secrets
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, set_random_seed, discrete_log

set_random_seed(int(42))

print("Phase 21.1: Pollard rho scaling validation 30-50 bits")
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

def pollard_rho_naive(E, P, Q, n, max_iter=10**8):
    """Sage's discrete_log (Pohlig-Hellman + BSGS/rho)."""
    t0 = time.time()
    try:
        k = discrete_log(Q, P, n, operation='+')
        elapsed = time.time() - t0
        return k, elapsed
    except Exception as e:
        return None, time.time() - t0

results = []
for bits in [20, 25, 30, 35, 40]:
    E, n, p = find_prime_order_curve(bits)
    if E is None:
        print(f"  bits={bits}: no curve")
        continue
    print(f"\n--- bits={bits}, p={p}, n={n} ---")

    n_trials = 3
    trial_times = []
    for trial in range(n_trials):
        P = E.random_point()
        while P.order() != n: P = E.random_point()
        secret = secrets.randbelow(int(n) - 1) + 1
        Q = secret * P

        k, elapsed = pollard_rho_naive(E, P, Q, n)
        if k is not None and k == secret:
            trial_times.append(elapsed)
            # print(f"  [Trial {trial}] solved in {elapsed:.4f}s (correct)")
        else:
            print(f"  [Trial {trial}] FAILED")
            break

    if trial_times:
        mean_time = sum(trial_times) / len(trial_times)
        print(f"  mean wall-time: {mean_time:.4f}s ({len(trial_times)} trials)")
        results.append((bits, n, mean_time))

print()
print("=" * 78)
print()
print("SCALING ANALYSIS:")
print()
import math
if len(results) >= 3:
    print(f"{'bits':>5} | {'n':>15} | {'time(s)':>10} | {'time/sqrt(n)':>15}")
    for bits, n, t in results:
        norm = t / math.sqrt(float(n))
        print(f"{bits:>5} | {n:>15} | {t:>9.4f}s | {norm:>15.6e}")

    # Fit slope: log(time) vs bits
    if len(results) >= 2:
        x = [bits for bits, _, _ in results]
        y = [math.log2(t) for _, _, t in results]
        n_pts = len(x)
        mean_x = sum(x) / n_pts
        mean_y = sum(y) / n_pts
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den = sum((xi - mean_x)**2 for xi in x)
        slope = num / den if den > 0 else 0
        intercept = mean_y - slope * mean_x
        print(f"\nLinear fit log2(time) = {slope:.4f} * bits + {intercept:.4f}")
        print(f"  (Pollard rho theory predicts slope ≈ 0.5 = log2(sqrt(2)))")

        # Project to 80 bits
        proj_80 = 2**(slope * 80 + intercept)
        print(f"\nProjected Sage Pollard rho time at 80 bits: {proj_80:.0f}s = {proj_80/3600:.1f}h = {proj_80/86400:.1f}d")

        # With C speedup (~63x at 4 threads):
        proj_80_c = proj_80 / 63
        print(f"Projected C-extension time at 80 bits (63x speedup): {proj_80_c:.0f}s = {proj_80_c/3600:.1f}h")

        # With C + multi-target (6 targets, sqrt(6) ≈ 2.45x):
        proj_80_c_mt = proj_80_c / 2.45
        print(f"Projected with multi-target boost: {proj_80_c_mt:.0f}s = {proj_80_c_mt/3600:.1f}h")

        budget_seconds = int(2 * 4 * 3600)  # 2 CPUs * 4 hours
        print(f"\nAutoLab budget: 2 CPUs × 4 hours = {budget_seconds}s ({float(budget_seconds)/3600.0:.0f}h CPU-time)")
        if proj_80_c_mt < budget_seconds:
            print(f"*** 80-bit ECDLP FITS IN BUDGET with C + multi-target ***")
        else:
            print(f"Short by factor {proj_80_c_mt / budget_seconds:.2f}x")

print()
print("=" * 78)
