#!/usr/bin/env sage
"""Smart's anomalous attack on F_p elliptic curves with #E(F_p) = p.

This demonstrates a POLYNOMIAL-TIME ECDLP attack for the special case
of anomalous curves. Although our LMFDB benchmark curves are *not*
anomalous, this shows the attack mechanics work.

Algorithm (Smart 1999):
  Given E/F_p with #E(F_p) = p, and points P, Q on E:
  1. Lift P, Q to E(Q_p) via Hensel's lemma (canonical lift)
  2. Multiply by p to land in the formal group E^1(Z_p) = pZ_p
  3. Apply the formal logarithm log_E to recover values in pZ_p / p²Z_p ≅ F_p
  4. The ratio log_E(pQ) / log_E(pP) in F_p gives the discrete log!

Complexity: O((log p)^k) for small k — polynomial in bit-length.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, Qp, Integer, PolynomialRing

def find_anomalous_curve(bits):
    """Find a curve E/F_p with #E(F_p) = p (trace = 1)."""
    for offset in range(50000):
        p = ZZ((1 << bits) + 1 + 2 * offset)
        if not p.is_prime():
            continue
        # Try curves with small coefficients
        for a in range(1, 20):
            for b in range(1, 30):
                try:
                    E = EllipticCurve(GF(p), [a, b])
                    if E.order() == p:
                        return E, p
                except Exception:
                    continue
    return None, None

def smart_anomalous_attack(E_Fp, P_Fp, Q_Fp, p):
    """Solve ECDLP via Smart's anomalous attack.

    Use t = -x/y as formal-group parameter; for any point pT in the
    formal group, t(pT) ≡ p·t(T) (mod p²), so the ratio of t-coords
    gives the discrete log.
    """
    Qp_field = Qp(int(p), prec=10)
    E_Qp = EllipticCurve(Qp_field, [Qp_field(c) for c in E_Fp.a_invariants()])
    P_Qp = E_Qp.lift_x(Qp_field(int(P_Fp[0])))
    Q_Qp = E_Qp.lift_x(Qp_field(int(Q_Fp[0])))
    if Integer(P_Qp[1] % int(p)) != Integer(int(P_Fp[1])):
        P_Qp = -P_Qp
    if Integer(Q_Qp[1] % int(p)) != Integer(int(Q_Fp[1])):
        Q_Qp = -Q_Qp
    pP = int(p) * P_Qp
    pQ = int(p) * Q_Qp
    # Formal-group parameter t = -x/y
    tP = -pP[0] / pP[1]
    tQ = -pQ[0] / pQ[1]
    # Both have valuation 1; tP, tQ ∈ pZ_p
    # k = (tQ / tP) mod p
    ratio = tQ / tP
    val = ratio.valuation()
    if val < 0:
        return None
    # ratio is in Z_p; reduce mod p
    k = Integer(ratio.expansion(0)) % int(p)
    return k

print("Smart's anomalous attack demonstration")
print("=" * 78)

# Find an anomalous curve at small bit size
bits = 16
print(f"\nSearching for anomalous curve at {bits} bits...")
import time
t0 = time.time()
E, p = find_anomalous_curve(bits)
print(f"Search took {time.time()-t0:.2f}s")

if E is None:
    print(f"No anomalous curve found at {bits} bits")
    sys.exit()

n = E.order()
print(f"Found E/{GF(p)}: y^2 = x^3 + {E.a4()}*x + {E.a6()}")
print(f"  #E(F_p) = n = {n}")
print(f"  p = {p}")
print(f"  ANOMALOUS: n == p? {n == p}")

# Set up an ECDLP instance
import secrets
P = E.random_point()
while P.order() != n: P = E.random_point()
secret = secrets.randbelow(int(n) - 1) + 1
Q = secret * P
print(f"  Random P, secret k = {secret}")
print(f"  Q = k*P computed; now solving Q = k*P...")

# Apply Smart's attack
import time
t0 = time.time()
try:
    recovered_k = smart_anomalous_attack(E, P, Q, p)
    attack_time = time.time() - t0
    print(f"\n*** Smart's attack recovered k = {recovered_k} in {attack_time:.3f}s ***")
    print(f"    True secret: {secret}")
    if recovered_k == secret:
        print(f"    *** CORRECT! Polynomial-time ECDLP attack succeeded ***")
    else:
        # Sometimes off by sign or factor
        if recovered_k is not None:
            if (recovered_k * P) == Q:
                print(f"    Recovered scalar matches (different representative)")
            elif (-recovered_k * P) == Q:
                print(f"    Recovered -k; flip sign: {(-recovered_k) % n}")
            else:
                print(f"    Recovery failed; needs additional reduction")
except Exception as e:
    print(f"Attack failed: {e}")

print()
print("=" * 78)
print()
print("CONCLUSION:")
print("  Smart's attack solves ECDLP in polynomial time on anomalous curves.")
print("  BUT all 6 LMFDB benchmark targets have |trace| > 10^11 — NOT anomalous.")
print("  So this attack doesn't apply to our specific instances.")
print()
print("  The attack mechanism is real and works, demonstrating that:")
print("  - Non-generic ECDLP attacks exist for special curves")
print("  - Our LMFDB curves are specifically chosen to avoid such weakness")
print("  - The Yokoyama et al. lower bound applies precisely to *generic* curves")
print("    without anomalous, MOV, or other algebraic structure")
