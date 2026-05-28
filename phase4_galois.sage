#!/usr/bin/env sage
"""Phase 4: Galois representation non-surjectivity check.

For each of the 4 LMFDB curves, check the primes ℓ where the mod-ℓ
Galois representation ρ_{E, ℓ}: Gal(Q-bar/Q) → GL_2(F_ℓ) is NOT
surjective. By Serre's open-image theorem, the set of such ℓ is finite
for any non-CM elliptic curve over Q.

For these specific ℓ, the mod-ℓ Galois action has a smaller image
(typically Borel, normalizer of split/nonsplit Cartan, or exceptional).
This can give constraints on k mod ℓ for the ECDLP at any p of good
reduction.

If we find any ℓ with non-trivial constraint that applies to our 80-bit
precomputed primes, that's a partial cryptanalytic exploit.
"""
import sys
import json
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, QQ, ZZ, prime_range

with open('/Volumes/Volume/autolab/tasks/ecdlp_index_calculus/environment/lmfdb_curves.json') as f:
    records = json.load(f)['records']

print("Phase 4: Mod-ℓ Galois representation surjectivity analysis")
print("=" * 70)

for r in records:
    if not r.get('precomputed_targets'): continue
    label = r['label']
    ainvs = r['ainvs']
    E = EllipticCurve(QQ, ainvs)
    print(f"\n--- {label} (ainvs={ainvs}, conductor={E.conductor()}) ---")

    # Sage's E.galois_representation() gives detailed analysis
    galrep = E.galois_representation()
    # Find non-surjective primes
    non_surj_primes = galrep.non_surjective()
    print(f"  Non-surjective primes ℓ (mod-ℓ Galois rep image ≠ GL_2(F_ℓ)): {non_surj_primes}")

    # For each non-surjective ℓ, get image type
    for ell in non_surj_primes:
        try:
            img_type = galrep.image_type(ell)
            print(f"    ℓ={ell}: image type = {img_type}")
        except Exception as e:
            print(f"    ℓ={ell}: image_type error = {e}")
    # Also check is_surjective for first few primes
    sample_primes = [2, 3, 5, 7, 11, 13]
    for ell in sample_primes:
        if ell in non_surj_primes: continue
        is_surj = galrep.is_surjective(ell)
        if not is_surj:
            print(f"    ℓ={ell}: NOT surjective (additional info)")

print()
print("=" * 70)
print("Implication analysis:")
print()
print("For a non-surjective ℓ, the mod-ℓ Galois rep image is some proper")
print("subgroup G ⊂ GL_2(F_ℓ). The image determines:")
print("  - whether E has an ℓ-isogeny over Q (Borel image)")
print("  - whether E[ℓ] has a rational subgroup of order ℓ")
print("  - whether E has CM by an order related to ℓ (Cartan image)")
print()
print("CONSTRAINT ON ECDLP: Even with non-surjective ρ_{E,ℓ}, the action")
print("of Frobenius π_p on E[ℓ](F_p̄) is determined by the IMAGE GROUP, but")
print("not by the SPECIFIC ELEMENT (Frobenius lives in a conjugacy class).")
print("For our ECDLP, k mod ℓ is unconstrained by ρ_{E,ℓ} alone.")
print()
print("The exception: if E has a Q-rational ℓ-isogeny, then E[ℓ] = C ⊕ C'")
print("with C an ℓ-cyclic isogeny kernel. Reducing mod p gives a known")
print("ℓ-subgroup of E(F_p). If this subgroup intersects ⟨P⟩ nontrivially,")
print("we get constraints on k mod ℓ.")
print()
print("Phase 4 conclusion: if any ℓ is in non_surj_primes AND has Borel")
print("image type AND ℓ divides the order of P, we have a Pohlig-Hellman")
print("attack on the ℓ-component of k.")
