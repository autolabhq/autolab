#!/usr/bin/env sage
"""Phase 20.1 sanity check: isogeny walk at small bits to verify mechanism."""
import time
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, set_random_seed

set_random_seed(int(42))

print("Phase 20.1 sanity check: isogeny walk at small bits")
print("=" * 78)

# Try several small primes
for bits in [10, 12, 14, 16, 18, 20, 25, 30]:
    print(f"\n--- bits={bits} ---")
    p_found = None
    for offset in range(2000):
        p = (1 << bits) + 13 + 2*offset
        if not ZZ(p).is_prime(): continue
        try:
            E = EllipticCurve(GF(p), [3, 5])
            n = ZZ(E.order())
            if n.is_prime():
                p_found = (p, E, n)
                break
        except Exception:
            continue
    if p_found is None:
        print(f"  no curve")
        continue
    p, E, n = p_found
    print(f"  p={p}, n={n}")

    # Walk
    t0 = time.time()
    try:
        isogs_2 = E.isogenies_prime_degree(2)
        isogs_3 = E.isogenies_prime_degree(3)
        isogs_5 = E.isogenies_prime_degree(5)
        elapsed = time.time() - t0
        print(f"  isogenies_prime_degree took {elapsed:.2f}s")
        print(f"  #2-isogenies: {len(isogs_2)}, #3-isogenies: {len(isogs_3)}, #5-isogenies: {len(isogs_5)}")
        if isogs_2:
            E_2 = isogs_2[0].codomain()
            print(f"    2-isogeny destination j-inv: {E_2.j_invariant()}")
            print(f"    new curve order: {E_2.order()}, preserved? {E_2.order() == n}")
    except Exception as e:
        print(f"  ERROR: {e}")
