# P1553 M6 signed-C3 divisor translation/gcd gate R161

Date: 2026-08-01

## Scope

R161 instantiates the first coordinate-specific source-locator interface after
R160. It represents the signed C3 endpoint set by an elliptic divisor
`(U(X),V(X))`, translates that divisor by each target in `F_p[X]/U`, and
extracts positive-C6 source candidates by polynomial gcd.

This is an exact algebraic interface and finite correctness gate. It does not
supply the target-batched modular-composition primitive needed for an attack.

## Semantic deduplication

R117 already charges independent translated-divisor evaluation. R161 does not
claim that cost calculation as new. Its delta is narrower:

- an exact signed `U,V` quotient representation for the C3 endpoints;
- explicit elliptic target-translation formulas in the quotient algebra;
- an exact source-membership gcd biconditional;
- a degree-at-most-20 source-factor theorem for unique positive-C6 targets;
- direct treatment of roots where the translation denominator vanishes.

R160 requires any surviving locator to inspect coordinates. These formulas
supply such an operation, but not an inside-cap batched realization.

## Signed divisor and translation

For the `n=binomial(d+2,3)=B^(9/4)` unordered C3 endpoints with distinct
x-coordinates, define

```text
U(X) = product_P (X-x(P))
V(x(P)) = y(P), deg(V) < n.
```

Then `U` divides `V^2-(X^3+aX+b)`. For a target `T=(u,v)` and a formal
endpoint `P=(X,V(X))`, outside the roots `u=X`, compute in `F_p[X]/U`

```text
lambda = (v+V)/(u-X)
phi    = lambda^2-u-X
psi    = lambda(u-phi)-v.
```

The point `(phi,psi)` is exactly `T-P`. Therefore a regular endpoint belongs
to a positive decomposition `T=P+P'` with `P'` in C3 exactly when

```text
U(phi(P)) = 0
psi(P)-V(phi(P)) = 0.
```

The split polynomial is consequently

```text
gcd(U, U(phi), psi-V(phi)).
```

Roots of `gcd(U,u-X)` are removed before inversion and checked directly.

## Constant output degree

If the target has one positive-C6 coefficient source and the C3 x-map is
injective, every split root represents a size-three coefficient submultiset
of its six occurrences. Distinct coefficient submultisets are images of the
`binomial(6,3)=20` occurrence subsets. The source gcd therefore has degree at
most 20. Factoring this constant-degree split polynomial and looking up a C3
source costs `B^(o(1))` after the gcd is available.

This theorem bounds answer size. It does not make the degree-`B^(9/4)`
compositions cheap.

Primary source for the summation-polynomial setting:

```text
Igor Semaev, Summation polynomials and the discrete logarithm problem on
elliptic curves, IACR ePrint 2004/031.
https://eprint.iacr.org/2004/031
local sha256 991f85d58ab68551a229266d03c2f88a5fc42e81b2a5f8f4432937bcceff16df
```

The signed-divisor quotient/gcd formulation's novelty is unverified.

## Fully charged cost

The persistent `U,V` state and C3 source dictionary have degree/size
`B^(9/4)`. Even granting optimistic quasi-linear modular composition, a
regular target needs two degree-`B^(9/4)` compositions and two gcds. Across
the R159 batch of `B^(5/4)` targets, independent evaluation costs

```text
B^(9/4) * B^(5/4) = B^(7/2),
```

which exceeds Pollard rho. The required complete-batch cap remains
`B^(5/4+o(1))`. Finite dictionary scans receive no asymptotic attack credit.

## Finite controls

The producer runs six controls over three public prime-order curve families
and two fixed seeds. Every control:

- constructs the exact C3 divisor with injective x-coordinates;
- verifies `U | V^2-(X^3+aX+b)`;
- recovers every preregistered unique positive-C6 relation/descent source;
- observes a maximum positive source-gcd degree at most 20;
- rejects a verifier-selected empty target;
- exercises and verifies the denominator-exception split;
- consumes no candidate DLP, root, count, marginal, rank, or source oracle.

The observed maximum gcd degrees are `6, 6, 14, 10, 14, 14`.

## Admission

Eighteen of twenty-six obligations pass. Admit:

- the signed C3 `U,V` divisor theorem on the stated injective chart;
- the exact target-translation formulas and source gcd biconditional;
- the denominator-exception split;
- the degree-at-most-20 source-factor theorem;
- all six finite positive, empty, and exceptional controls.

Do not admit:

- target-batched many-inner modular composition;
- a target-batched gcd/source adjoint;
- deterministic hash-to-curve pseudorandomness;
- an unconditional total attack cost;
- a generic-prime coordinate-family algorithm;
- a Pollard-rho or Shoup improvement;
- an ECDLP breakthrough.

Disposition:

```text
ADMIT_SIGNED_C3_U_V_TRANSLATION_GCD_INTERFACE_AND_DEGREE_20_SOURCE_FACTOR__INDEPENDENT_BATCH_B7O2__TARGET_BATCHED_COMPOSITION_GCD_ADJOINT_OPEN__NO_RHO__NO_SHOUP_IMPROVEMENT__NO_BREAKTHROUGH
```

## Next action

Construct or rule out, under an explicit arithmetic model, a many-inner
target-batched modular-composition and gcd source adjoint for one frozen
degree-`B^(9/4)` signed divisor. It must process all `B^(5/4)` targets in
`B^(5/4+o(1))` work, return the degree-at-most-20 source factors, handle
`u-X` exceptional roots, and avoid materializing per-target degree-`B^(9/4)`
states before replaying factor logs and identical descent.
