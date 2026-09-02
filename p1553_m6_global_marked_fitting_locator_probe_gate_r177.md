# P1553 M6 global marked Fitting-locator gate R177

Date: 2026-08-01

## Scope

R177 replaces the R175 balanced subset-query tree by one globally marked
lambda-adic norm. Its first nonzero lambda coefficient is the characteristic
polynomial of the left coordinate on the signed-incidence kernel, so one gcd
with the squarefree selected-divisor polynomial returns every distinct R176
candidate root.

The identity is exact and eliminates adaptive subset queries at the algebraic
interface. The standard pair algebra still has dimension `n^2`, and the finite
interpolation used to verify the identity is much more expensive. This is not
an output-sensitive constructor, an arithmetic-circuit lower bound, an ECDLP
algorithm, or a Pollard-rho or Shoup improvement.

## Global Marked Norm

Let `T_D` be the reduced pair algebra of `D x D`. Let `K` be multiplication by

```text
k(P,Q) = h(P+Q),
```

where `h` is the R176 principal signed-incidence witness, and let `X_1` be
multiplication by `x(P)`. Define

```text
F(A, lambda)
  = det(K + lambda (A I - X_1))
  = product_(P,Q) (h(P+Q) + lambda (A-x(P))).
```

If `M` is the number of signed incidence pairs, then

```text
ord_lambda F = M
```

and

```text
[lambda^M] F
  = pdet(K) product_(h(P+Q)=0) (A-x(P)).
```

Here `pdet(K)` is the product of the nonzero diagonal values on the complement
of `ker K`. It is a field unit, so monic normalization gives exactly

```text
det(A I - X_1 | ker K).
```

The roots are the signed candidate left coordinates, with their pair-incidence
multiplicities. Since the selected-divisor polynomial `U` is squarefree,

```text
gcd(U, monic([lambda^M]F))
```

returns exactly the distinct R176 candidates. No subset query, candidate
inversion, or opposite-sign verification branch is used.

## Exact Controls

The six R176 controls have total pair-algebra dimension `8,922`. Their kernel
dimensions sum to `241`, exactly the R176 signed-incidence count. The recovered
marker polynomial has total degree `241`, maximum candidate multiplicity four,
and its gcd with the six squarefree selected-divisor polynomials returns all
`140` R176 roots with total candidate-factor degree `140`.

For each control, the finite verifier evaluates the full pair product as a
univariate polynomial in `lambda`, truncated through degree `M`, at `M+1`
marker values. It verifies that every coefficient below `M` vanishes,
interpolates coefficient `M` in `A`, and compares it exactly with
`pdet(K) product(A-x(P))` over the zero pairs.

```text
marker interpolation samples:       247
truncated lambda updates:     28,788,282
```

These are finite correctness controls and receive no asymptotic credit.

## Cost Boundary

```text
selected divisor degree n:                    B^(9/4)
target witness degree N:                      B^(5/4)
candidate marker degree M:                    B^(3/4)
pair-algebra dimension n^2:                   B^(9/2)
generic pair-algebra element state:           B^(9/2)
full bivariate marked-norm coefficient body:  B^9
explicit symbolic lambda truncation state:    B^(3/2)
explicit pair scan:                           B^(9/2)
explicit marker interpolation work n^2 M^2:   B^6
conditional output-sensitive Fitting total:   B^(9/4)
R163 label/backpointer postprocessing:         B^2
rho proxy:                                    B^(5/2)
```

The low-degree output itself fits below rho. The missing step is constructing
that output from compact `U,V,h`; neither output size nor the exact finite
factorization supplies the constructor. Standard pair-algebra, generic matrix
pencil, full bivariate determinant, and explicit truncated interpolation
routes remain above rho. These represented-route costs are not arithmetic-
circuit lower bounds.

## Open Primitive

The surviving primitive is a fraction-free output-sensitive marked Fitting or
subresultant operator accepting compact `U,V,h`. It must compute

```text
M = dim ker K
det(A I - X_1 | ker K)
```

in softly `O(n+N+M)` total work without constructing the `n^2` pair algebra,
a generic `n^2` matrix pencil, the full `lambda,A` determinant body, or the
`M^2` marker-interpolation grid. With R163 candidate output and label recovery,
this would retain the conditional `B^(9/4)` envelope.

## Admission

Admit the global marked norm identity, exact lambda valuation, lowest-
coefficient product formula, restricted-kernel characteristic polynomial,
candidate gcd locator, all six controls, and the removal of subset queries from
the algebraic interface.

Do not admit an output-sensitive marked Fitting constructor, an arithmetic-
circuit lower bound, deterministic hash-to-curve transfer, a generic-prime
coordinate-family algorithm, a complete ECDLP attack, or a Pollard-rho or Shoup
improvement.

Disposition:

```text
ADMIT_GLOBAL_MARKED_LAMBDA_NORM__KERNEL_NULLITY_EQUALS_241_SIGNED_INCIDENCES__LOWEST_COEFFICIENT_IS_RESTRICTED_X1_CHARACTERISTIC_POLYNOMIAL__DEGREE_241_WITH_MULTIPLICITY__GCD_U_RETURNS_140_R176_ROOTS__NO_SUBSET_QUERIES__STANDARD_PAIR_ALGEBRA_N2_B9O2__FULL_BODY_N4_B9__EXPLICIT_TRUNCATION_B6__OUTPUT_SENSITIVE_MARKED_FITTING_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute a fraction-free output-sensitive marked Fitting or
subresultant operator that accepts compact `U,V` and the degree-`N` principal
target witness `h`, computes `M=dim ker K`, and emits
`det(AI-X_1|ker K)` in softly `O(n+N+M)` total work. Reject `n^2` pair
enumeration or tensor state, the full lambda/A determinant body, `M^2` marker
interpolation, generic `n^2` matrix pencils, candidate inversions, and unit-cost
Fitting, kernel, resultant, root, count, marginal, rank, source, or generic
locator oracles.
