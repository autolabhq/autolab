# P1553 M6 S3 determinantal transfer noncommutativity gate R173

Date: 2026-08-01

## Scope

R173 tests the most compact literal realization of the factored self-`S3`
interface left open by R172. It rewrites each `S3(X,Z,u)` as the determinant of
an affine `2 by 2` matrix pencil, multiplies those pencils in both target
orders, and charges the represented transfer before any outer resultant.

The determinant identity is exact. The naive matrix transfer is order
sensitive and its represented body is quadratic in the target count. This is
a closure of that transfer, not a general resultant lower bound, an ECDLP
algorithm, or a Pollard-rho or Shoup improvement.

## Determinantal Form

Let

```text
s1 = X + Z + u,
s2 = XZ + Xu + Zu,
s3 = XZu.
```

Then

```text
S3(X,Z,u) = (s2-a)^2 - 4 s1 (s3+b)
           = det([[s2-a, 2(s3+b)], [2s1, s2-a]]).
```

Writing the determinant matrix as `M_u(Z)=A(X,u)+Z B(X,u)` gives

```text
A = [[Xu-a, 2b],       B = [[X+u, 2Xu],
     [2(X+u), Xu-a]]        [2,   X+u]].
```

The controls verify the evaluated determinant in 68,326 rows.

## Order Obstruction

The coefficient matrices satisfy

```text
[A,B] = diag(delta,-delta),
delta = 4(b-Xu(X+u)).
```

Consequently,

```text
M_u(z1)M_u(z2)-M_u(z2)M_u(z1) = (z2-z1)[A,B].
```

All 1,675,890 tested distinct-root pairs are noncommuting, with zero tested
points on the exceptional `delta=0` locus. All 118 target-factor pairs are
also noncommuting. Reversing the target order changes two of four matrix
entries in every control.

The determinant remains exact in either order:

```text
det(product_j M_{u_j}(X,Z))
  = product_j S3(X,Z,u_j)
  = Res_Y(W(Y),S3(X,Z,Y)).
```

Both orders replay all six R172 reverse resultants exactly. Thus the scalar
determinant is symmetric while the proposed transfer state is not.

## Discriminant Split

As a quadratic in `Z`,

```text
disc_Z S3(X,Z,u) = 16 (X^3+aX+b)(u^3+au+b).
```

The controls verify this identity in 1,486 rows. On the selected and target
curve divisors, the square root is `4 V(X) V_T(u)`. Diagonalizing the pencil
therefore recovers the plus and target-sign-conjugate branches already charged
in R172; it does not create a third compact branch.

## Cost Boundary

```text
factored pencil input state:                 B^(5/4)
represented ordered 2 by 2 transfer body:   B^(5/2)
represented scalar determinant body:         B^(5/2)
standard outer root/factor grid:              B^(7/2)
represented aggregate output:                 B^(9/4)
rho proxy:                                    B^(5/2)
```

The six represented matrix products reserve 1,448 coefficient slots and use
1,418 of them, with coefficient-rank sum 184. These finite density and rank
measurements receive no asymptotic or lower-bound credit. The slot geometry
itself is `4(N+1)^2`, so materializing this transfer is not strictly below rho.

## Admission

Admit the elementary-symmetric determinant, affine pencil, commutator,
discriminant, order-dependence, and exact determinant replay.

Close only the naive order-independent `2 by 2` matrix transfer and its
represented product. Preserve custom commutative, transposed, implicit, and
arbitrary-squarefree-modulus algorithms as open. Do not admit a factored
self-`S3` resultant algorithm, a generic-prime coordinate-family algorithm, a
complete attack, or a Pollard-rho or Shoup improvement.

Disposition:

```text
ADMIT_ELEMENTARY_SYMMETRIC_S3_DETERMINANT_AND_SEPARABLE_DISCRIMINANT__EXACT_2X2_AFFINE_PENCIL__GENERIC_PENCIL_AND_TARGET_FACTOR_NONCOMMUTATIVITY__ORDERED_TRANSFER_DETERMINANT_REPLAYS_R172__REPRESENTED_MATRIX_BODY_N2_B5O2_AT_RHO__DIAGONALIZATION_RETURNS_R172_SIGN_SPLIT__NAIVE_2X2_SYMMETRIC_TRANSFER_CLOSED__CUSTOM_IMPLICIT_RESULTANT_OPEN__NO_GENERAL_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Use the known square root `4 V(X) V_T(u)` directly. Formulate the two diagonal
target-sign branches as compact divisor pushforwards modulo arbitrary
squarefree `U`, then test a transposed modular-composition or multipoint
implementation with all setup charged and total work strictly below
`B^(5/2)`. Reject any route that exposes `N^2` coefficients, visits `nN` pairs,
uses target-dependent preprocessing, or assumes a unit-cost norm, resultant,
root, or multipoint oracle.
