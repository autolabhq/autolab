# P1553 M6 lambda-zero Fitting/target-norm deduplication gate R170

Date: 2026-08-01

## Scope

R170 asks whether the zero specialization of R169's scalar resolvent creates a
new fraction-free primitive. It does not: after the explicit R167 correction
units, the lambda-zero Fitting support is exactly the existing aggregate
Kummer target norm.

The equivalence is exact. The finite coefficient-density observations receive
no asymptotic attack or lower-bound credit.

## Exact identity

Let `A=F_p[X]/(U)` and, after the public target-equality split, define

```text
c_j(P) = U(x(T_j-P))
C(P)   = product_j c_j(P) in A.
```

Then the candidate factor is

```text
G = gcd(U,C).
```

R169's scalar resolvent has constant term

```text
chi_P(0) = product_(Q in S union -S) h(Q+P).
```

R167 proves componentwise that, after multiplying by the explicit auxiliary
correction and dividing by the unit `h(P)^(2n)`, this value equals `C(P)`.
Therefore its zero-th Fitting support and the direct target norm have exactly
the same roots. Fraction-free specialization preserves the roots but does not
construct the aggregate element for free.

## Finite controls

Six controls over three curves and two seeds reconstruct every target factor
in the monomial basis of `A`, multiply the factors modulo `U`, and verify:

- every interpolated target factor is exact;
- the coefficient-ring product equals the direct target product;
- the corrected lambda-zero value equals that product at every endpoint;
- `gcd(U,C)` has exactly the 140 R167 candidate roots;
- every target factor has degree `n-1` and 100% coefficient density;
- every aggregate `C` has degree `n-1` and 100% coefficient density.

The controls contain 1,486 represented target-factor coefficient slots. The
density result denies sparsity credit only to this standard representation; it
is not a circuit lower bound.

No DLP, root, count, marginal, rank, source, norm, Fitting, or subresultant
oracle is consumed.

## Cost boundary

At the campaign caps:

```text
compact target-divisor SLP state:       B^(5/4)
N represented elements of A:            B^(7/2)
standard fraction-free product/PRS:      B^(7/2)
one represented aggregate element C:    B^(9/4)
swapped n-by-n Fitting matrix:           B^(9/2)
expected candidate factor:               B^(3/4)
signed candidate verification:           B^2
rho proxy:                               B^(5/2)
```

The final aggregate element is small enough, but every standard route to it is
above rho. The remaining question is whether the generalized Miller SLP can
stream the target norm directly into `A` in softly `O(n+N)` work without
materializing the dense `nN` input body.

## Literature and deduplication

R164 already isolates the output-sensitive aggregate target norm. R167 gives
the compact target-divisor witness and exact reciprocity swap. R169 gives the
candidate-safe scalar resolvent. R170 proves that lambda-zero Fitting is the
same hard aggregate constructor up to units, so it is not a separate lane.

Prokofev and Zabrodin prove determinant, inverse, product, and factorization
identities for complex sigma-function elliptic Cauchy matrices. Their
factorization uses full square matrices and does not provide a finite-field
Weierstrass-coordinate, candidate-safe, output-sensitive norm algorithm for an
arbitrary degree-`N` target divisor. Polewise use would retain `nN` state.

## Admission

Sixteen of twenty-four obligations pass. Admit the exact lambda-zero/target-
norm equivalence, all six coefficient-ring replays, full-density controls, and
the `B^(7/2)` standard represented cost.

Do not admit an SLP-streaming norm constructor, custom algebraic elliptic
operator, deterministic hash-to-curve transfer, generic-prime algorithm,
Pollard-rho or Shoup improvement, or ECDLP breakthrough.

Disposition:

```text
ADMIT_LAMBDA_ZERO_FITTING_EQUALS_R167_DIRECT_TARGET_NORM_UP_TO_UNITS__SIX_COEFFICIENT_RING_REPLAYS__STANDARD_DENSE_FACTORS_NN_B7O2__SWAPPED_FITTING_N2_B9O2__ELLIPTIC_CAUCHY_FACTORIZATION_NOT_A_FINITE_FIELD_OUTPUT_SENSITIVE_CONSTRUCTOR__SLP_STREAMING_NORM_OPEN__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next action

Construct or refute an SLP-streaming output-sensitive elliptic target norm
modulo `U` from the compact generalized Miller witness, below `B^(5/2)` and
preferably `B^(9/4+o(1))`. It must emit the aggregate element or candidate gcd
without `N` dense quotient-ring elements, an `nN` coefficient body, or an
`n^2` Fitting matrix.
