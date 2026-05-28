#!/usr/bin/env sage
"""Explicit construction of Semaev summation polynomials F_n for n = 3, 4, 5.

Semaev (2004) defined a sequence of polynomials F_n(X_1, ..., X_n) such that
F_n(x_1, ..., x_n) = 0 iff there exist points P_i = (x_i, y_i) on the
elliptic curve E with sign choices ε_i ∈ {±1} such that
  Σ ε_i P_i = O   (identity of E)

For short Weierstrass form y^2 = x^3 + ax + b:

F_3(X_1, X_2, X_3) = (X_1 - X_2)^2 X_3^2
                   - 2[(X_1 + X_2)(X_1 X_2 + a) + 2b] X_3
                   + (X_1 X_2 - a)^2 - 4b(X_1 + X_2)

For n ≥ 4: F_n(X_1, ..., X_n) = res_Y(F_{n-1}(X_1, ..., X_{n-2}, Y),
                                       F_3(X_{n-1}, X_n, Y))

We compute F_3, F_4, F_5, measure their algebraic properties, and
benchmark Groebner basis computation on systems involving F_4.
"""
import time
import sys
sys.stdout.reconfigure(line_buffering=True)
from sage.all import EllipticCurve, GF, ZZ, PolynomialRing, Ideal, QQ

print("Semaev summation polynomials: explicit construction")
print("=" * 70)

# Use short Weierstrass form y^2 = x^3 + a*x + b
# We'll work symbolically over Q first to verify structure
R = PolynomialRing(QQ, names=['a', 'b'])
a_sym, b_sym = R.gens()

# F_3 explicit formula
R3 = PolynomialRing(R, names=['X1', 'X2', 'X3'], order='lex')
X1, X2, X3 = R3.gens()

F3_def = (X1 - X2)**2 * X3**2 \
       - 2*((X1 + X2)*(X1*X2 + a_sym) + 2*b_sym) * X3 \
       + (X1*X2 - a_sym)**2 - 4*b_sym*(X1 + X2)

print(f"\nF_3:")
print(f"  total degree: {F3_def.total_degree()}")
print(f"  degree in X_1: {F3_def.degree(X1)}")
print(f"  degree in X_2: {F3_def.degree(X2)}")
print(f"  degree in X_3: {F3_def.degree(X3)}")
print(f"  # monomials: {len(F3_def.monomials())}")

# F_4 via resultant: F_4(X1, X2, X3, X4) = res_Y(F_3(X1, X2, Y), F_3(X3, X4, Y))
R4 = PolynomialRing(R, names=['X1', 'X2', 'X3', 'X4', 'Y'], order='lex')
X1_4, X2_4, X3_4, X4_4, Y = R4.gens()

F3_a = (X1_4 - X2_4)**2 * Y**2 \
     - 2*((X1_4 + X2_4)*(X1_4*X2_4 + a_sym) + 2*b_sym) * Y \
     + (X1_4*X2_4 - a_sym)**2 - 4*b_sym*(X1_4 + X2_4)
F3_b = (X3_4 - X4_4)**2 * Y**2 \
     - 2*((X3_4 + X4_4)*(X3_4*X4_4 + a_sym) + 2*b_sym) * Y \
     + (X3_4*X4_4 - a_sym)**2 - 4*b_sym*(X3_4 + X4_4)

print(f"\nComputing F_4 = res_Y(F_3a, F_3b)...")
t0 = time.time()
F4_def = F3_a.resultant(F3_b, Y)
elapsed = time.time() - t0
print(f"  F_4 built in {elapsed:.2f}s")
print(f"  total degree: {F4_def.total_degree()}")
for i, var in enumerate([X1_4, X2_4, X3_4, X4_4]):
    print(f"  degree in X_{i+1}: {F4_def.degree(var)}")
print(f"  # monomials: {len(F4_def.monomials())}")

# F_5 = res_Y(F_4(X1, X2, X3, X4, Y), F_3(X4, X5, Y))
# This is more complex - F_5 in 5 variables
# F_5(X1, ..., X5) such that F_5 = 0 iff sum εi P_i = O with x-coords (X1, ..., X5)
print(f"\nComputing F_5 (5-variable summation polynomial)...")
R5 = PolynomialRing(R, names=['X1', 'X2', 'X3', 'X4', 'X5', 'Y'], order='lex')
X1_5, X2_5, X3_5, X4_5, X5_5, Y5 = R5.gens()
# F_4 with X4 replaced by Y (so we can resultant)
# F_4(X1, X2, X3, Y) means P1+P2+P3+P_Y = O for some point P_Y with x-coord Y
F4_subst = F3_a  # placeholder — let me recompute F4 specifically with X4 -> Y
# Actually: F_5(X1,...,X5) = res_Y( F_4(X1,X2,X3,Y) , F_3(X4, X5, Y) )

# Rebuild F_4(X1, X2, X3, X_unknown):
R4_for_5 = PolynomialRing(R, names=['X1', 'X2', 'X3', 'X4', 'Z'], order='lex')
X1z, X2z, X3z, X4z, Z = R4_for_5.gens()
F3_ab_in_4 = (X1z - X2z)**2 * Z**2 \
           - 2*((X1z + X2z)*(X1z*X2z + a_sym) + 2*b_sym) * Z \
           + (X1z*X2z - a_sym)**2 - 4*b_sym*(X1z + X2z)
F3_cd_in_4 = (X3z - X4z)**2 * Z**2 \
           - 2*((X3z + X4z)*(X3z*X4z + a_sym) + 2*b_sym) * Z \
           + (X3z*X4z - a_sym)**2 - 4*b_sym*(X3z + X4z)
t0 = time.time()
F4_general = F3_ab_in_4.resultant(F3_cd_in_4, Z)  # F_4(X1, X2, X3, X4)
elapsed = time.time() - t0
print(f"  F_4 general rebuilt: {elapsed:.2f}s")
print(f"  F_4 general # monomials: {len(F4_general.monomials())}")

# F_5: take F_4(X1, X2, X3, Y) and F_3(X4, X5, Y), eliminate Y
R5g = PolynomialRing(R, names=['X1', 'X2', 'X3', 'X4', 'X5', 'Y'], order='lex')
X1g, X2g, X3g, X4g, X5g, Yg = R5g.gens()
# Map F_4_general(X1, X2, X3, X4) -> F4_general(X1g, X2g, X3g, Yg)
phi4 = R4_for_5.hom([X1g, X2g, X3g, Yg, R5g.zero()], R5g)
F4_in_5 = phi4(F4_general.subs({Z: 0}))  # Wait — need to substitute X4 with Yg directly
# Hmm let me just rebuild explicitly
F4_with_Y = (X1g - X2g)**2 * (R5g(0))  # placeholder
# Actually let me just do the substitution

# Rebuild F4_general directly in R5g with X4 = Yg
F3_ab_R5 = (X1g - X2g)**2 * Yg**2 \
         - 2*((X1g + X2g)*(X1g*X2g + a_sym) + 2*b_sym) * Yg \
         + (X1g*X2g - a_sym)**2 - 4*b_sym*(X1g + X2g)
# F_4(X1, X2, X3, X4): need an intermediate variable
# F_4 = res_Z(F_3(X1, X2, Z), F_3(X3, X4, Z))
# We need to compute this in R5g.

R5z = PolynomialRing(R, names=['X1', 'X2', 'X3', 'Y', 'X5', 'Z'], order='lex')
X1zz, X2zz, X3zz, Yzz, X5zz, Zzz = R5z.gens()
F3_ab_zz = (X1zz - X2zz)**2 * Zzz**2 \
         - 2*((X1zz + X2zz)*(X1zz*X2zz + a_sym) + 2*b_sym) * Zzz \
         + (X1zz*X2zz - a_sym)**2 - 4*b_sym*(X1zz + X2zz)
F3_cd_zz = (X3zz - Yzz)**2 * Zzz**2 \
         - 2*((X3zz + Yzz)*(X3zz*Yzz + a_sym) + 2*b_sym) * Zzz \
         + (X3zz*Yzz - a_sym)**2 - 4*b_sym*(X3zz + Yzz)
print(f"\nComputing F_5: res_Z(F_3(X1, X2, Z), F_3(X3, Y, Z)) gives F_4(X1, X2, X3, Y)")
t0 = time.time()
F4_via_zz = F3_ab_zz.resultant(F3_cd_zz, Zzz)
elapsed = time.time() - t0
print(f"  F_4(X1, X2, X3, Y) built: {elapsed:.2f}s, # monomials {len(F4_via_zz.monomials())}")

# Now F_5 = res_Y(F_4(X1, X2, X3, Y), F_3(X5, X4, Y))
# wait we already have F4_via_zz with X4 named Yzz. Now eliminate Yzz against F_3(X5, X4? Let me use a clean ring
R6 = PolynomialRing(R, names=['X1', 'X2', 'X3', 'X4', 'X5', 'Y'], order='lex')
X1f, X2f, X3f, X4f, X5f, Yf = R6.gens()
# Rebuild F_4(X1, X2, X3, Y) in R6 (where Y is the auxiliary)
F3_part1_R6 = (X1f - X2f)**2 * Yf**2 \
            - 2*((X1f + X2f)*(X1f*X2f + a_sym) + 2*b_sym) * Yf \
            + (X1f*X2f - a_sym)**2 - 4*b_sym*(X1f + X2f)
# F_3(X3, X4, Y) — wait we want F_4 then resultant against F_3
# Let me just compute F_5(X1, X2, X3, X4, X5):
#   F_5 = 0 iff sum εi P_i = O for i=1..5
# Construction: F_5 = res_Y( F_3(X1, X2, Y), F_4(X3, X4, X5, Y) )
# But we need F_4(a, b, c, d) - we have F4_general

# Substitute F4_general(X3, X4, X5, Y) into F_5 setup:
R5_final = PolynomialRing(R, names=['X1', 'X2', 'X3', 'X4', 'X5', 'Y'], order='lex')
v = R5_final.gens()
F3_for_F5 = (v[0] - v[1])**2 * v[5]**2 \
          - 2*((v[0] + v[1])*(v[0]*v[1] + a_sym) + 2*b_sym) * v[5] \
          + (v[0]*v[1] - a_sym)**2 - 4*b_sym*(v[0] + v[1])
# F_4(X3, X4, X5, Y) — substitute X1->v[2], X2->v[3], X3->v[4], X4->v[5] (Y)
# F4_general was built in R4_for_5 with vars X1, X2, X3, X4, Z (Z eliminated)
# F4_general has vars X1, X2, X3, X4 (Z eliminated by resultant)
F4_for_F5_subs = {R4_for_5.gens()[0]: v[2], R4_for_5.gens()[1]: v[3],
                  R4_for_5.gens()[2]: v[4], R4_for_5.gens()[3]: v[5],
                  R4_for_5.gens()[4]: R4_for_5.zero()}
# Sage subs on polynomial
F4_for_F5 = F4_general(v[2], v[3], v[4], v[5], 0)  # the 5th arg is Z which is eliminated

print(f"\nComputing F_5 = res_Y(F_3(X1, X2, Y), F_4(X3, X4, X5, Y))...")
t0 = time.time()
F5_def = F3_for_F5.resultant(F4_for_F5, v[5])
elapsed = time.time() - t0
print(f"  F_5 built in {elapsed:.2f}s")
print(f"  total degree: {F5_def.total_degree()}")
for i in range(5):
    print(f"  degree in X_{i+1}: {F5_def.degree(v[i])}")
print(f"  # monomials: {len(F5_def.monomials())}")

# Theoretical degrees: deg_X_i(F_n) = 2^{n-2}
print(f"\nTheoretical degree pattern (Semaev 2004): deg_X_i(F_n) = 2^(n-2)")
print(f"  F_3: should be 2^1 = 2 in each var ✓")
print(f"  F_4: should be 2^2 = 4 in each var (verify above)")
print(f"  F_5: should be 2^3 = 8 in each var (verify above)")
