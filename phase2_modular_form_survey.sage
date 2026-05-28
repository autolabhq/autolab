#!/usr/bin/env sage
"""Phase 2 sub-experiment 1: a_p factorization survey for the 4 LMFDB curves.

For each of 67.a1, 21175.bc1, 23232.cr1, 114224.v1, compute a_p for primes
p up to a bound, and look for:
  - primes p where |a_p| is unusually small (near-anomalous reduction)
  - primes p where a_p factors into small primes (smooth Pohlig-Hellman)
  - primes p where the curve has rank > 0 in some sense (related to BSD)
  - primes p where t^2 - 4p has small fundamental discriminant (CM-like)

If any such primes exist within the AutoLab benchmark constraint (p < 12000),
they would constitute a real cryptanalytic exploit.
"""
import sys
import json
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, ZZ, QQ, prime_range, gcd, fundamental_discriminant

with open('/Volumes/Volume/autolab/tasks/ecdlp_index_calculus/environment/lmfdb_curves.json') as f:
    records = json.load(f)['records']

print("Phase 2 sub-experiment 1: a_p factorization survey")
print("=" * 70)

# For each precomputed-target curve, run survey
findings = {}
for r in records:
    if not r.get('precomputed_targets'): continue
    label = r['label']
    ainvs = r['ainvs']
    E = EllipticCurve(QQ, ainvs)
    conductor = E.conductor()
    print(f"\n--- {label} ---")
    print(f"  ainvs = {ainvs}, conductor = {conductor}")

    interesting = {
        'near_anomalous': [],     # |a_p| < log(p)
        'smooth_trace': [],       # |a_p| factors into primes < 100
        'small_fund_disc': [],    # t^2 - 4p has small fundamental discriminant
        'mw_lift_candidate': [],  # specific p where reduced Mordell-Weil generator has small order
    }

    # Iterate over many primes
    primes_to_check = list(prime_range(5, 10000))
    for p in primes_to_check:
        if conductor % p == 0: continue  # bad reduction
        try:
            a_p = E.ap(p)
        except Exception:
            continue

        # Near-anomalous: |a_p| < 5
        if abs(a_p) < 5:
            n = p + 1 - a_p
            interesting['near_anomalous'].append((p, a_p, n))

        # Smooth trace: factor |a_p|
        if a_p != 0 and abs(a_p) > 1:
            t_abs = abs(a_p)
            t_factors = []
            d = 2
            temp = t_abs
            while d * d <= temp:
                if temp % d == 0:
                    while temp % d == 0:
                        temp //= d
                    t_factors.append(d)
                d += 1
            if temp > 1: t_factors.append(temp)
            if t_factors and max(t_factors) <= 50:
                interesting['smooth_trace'].append((p, a_p, t_factors))

        # Small fundamental discriminant for End(E_p) ⊗ Q
        disc = a_p*a_p - 4*p
        # disc < 0 always (Hasse). Compute fundamental disc
        f_disc = fundamental_discriminant(disc)
        if abs(f_disc) < 1000:
            interesting['small_fund_disc'].append((p, a_p, disc, f_disc))

    # Report
    print(f"  Primes checked: {len(primes_to_check)}")
    for category, items in interesting.items():
        print(f"  {category}: {len(items)} hits")
        for item in items[:5]:
            print(f"    {item}")

    findings[label] = interesting

# Cross-curve analysis
print()
print("=" * 70)
print("Cross-curve analysis:")

# Check if any p appears in interesting categories for MULTIPLE curves
all_categories = ['near_anomalous', 'smooth_trace', 'small_fund_disc']
for cat in all_categories:
    primes_per_curve = {}
    for label, finds in findings.items():
        primes_per_curve[label] = set(item[0] for item in finds[cat])
    # Find overlaps
    all_labels = list(primes_per_curve.keys())
    if len(all_labels) >= 2:
        common = primes_per_curve[all_labels[0]]
        for label in all_labels[1:]:
            common = common & primes_per_curve[label]
        print(f"  {cat}: {len(common)} primes shared across ALL {len(all_labels)} curves")
        if common:
            print(f"    Common primes: {sorted(common)[:10]}")

# Summary
print()
print("Summary of Phase 2 findings:")
for label in findings:
    total = sum(len(items) for items in findings[label].values())
    print(f"  {label}: {total} total 'interesting' primes out of {len(primes_to_check)}")

print()
print("Conclusion: Phase 2 sub-experiment 1 complete.")
print("If any 'interesting' primes lie in the AutoLab range (p < 12000) AND would")
print("give a concrete attack on the verifier's challenge for that (label, p),")
print("we would have a Phase 2 success.")
