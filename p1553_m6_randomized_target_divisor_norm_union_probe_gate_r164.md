# P1553 M6 randomized target-divisor norm union gate R164

Date: 2026-08-01

## Scope

R164 replaces R163's exact Fermat projector with one randomized linear
residual per public target. It proves a one-sided reduction from the aggregate
union to one regular-branch target norm, makes the result exact by direct
verification, and removes denominator incidents from the remaining hard
constructor.

The finite producer evaluates every endpoint-target pair. That work receives
no asymptotic attack credit.

## Collision-safe target algebra

Give the `N` public targets distinct field labels `z_j` and set

```text
W(Z) = product_j (Z-z_j).
```

Target coordinates and a degree-below-`N` randomizer polynomial `R(Z)` are
interpolated in `F_p[Z]/W`. The evaluation map is a vector-space isomorphism,
so a uniform `R` gives independent uniform values `r_j=R(z_j)`. Labels remain
distinct even if public targets share an x-coordinate.

## Randomized same-target residual

For a regular pair `(P,T_j)`, let `a_j(P),b_j(P)` be R161's signed x/y
membership residuals and define

```text
c_j(P) = a_j(P) + r_j b_j(P).
```

Every true same-target match has `c_j(P)=0` for every randomizer. A regular
nonmatch has false cancellation probability at most `1/p`: exactly one value
of `r_j` when `b_j(P)` is nonzero, and none when it is zero.

In the split tensor algebra, assign value one to the incidence branch
`x(P)=x(T_j)` and use `c_j(P)` elsewhere. Then

```text
C(P) = product_j c_j(P)
G_random = gcd(U,C)
```

contains every regular true union root and has no false negatives. It is the
regular-branch norm, equivalently `Res_Z(W,c)` after the incidence component
is split.

For `n=B^(9/4)`, `N=B^(5/4)`, and `p=B^5`, a union bound gives

```text
Pr[any regular false root] <= nN/p = B^(-3/2).
```

Factor `G_random`, scan the public targets for each root, and retain only
direct group-law matches. This makes the result exact with expected `B^2`
verification work. A deliberately forced cancellation `r=-a/b` produces a
false root in the finite control, and verification removes it exactly.

## Incidence branch

All possible denominator incidents are listed by

```text
J = gcd(U, product_j (X-x(T_j))).
```

Signed C3 x-injectivity implies at most one left endpoint per target x, hence
at most `N` incidence pairs. Build `J` in softly `O(n+N)` work, hash targets by
x-coordinate, and check only matching buckets. This costs
`B^(9/4+o(1))`, below rho, and preserves the positive tangent control
`P+(-2P)=-P` together with its opposite regular orientation.

## Standard norm cost and dynamic evaluation

Once all `c_j` are represented independently in `F_p[X]/U`, a product tree or
standard coefficient-ring resultant uses softly `O(nN)=B^(7/2)` base-field
work, one exponent above Pollard rho. This is a charge for the standard
represented route, not an arithmetic-circuit lower bound.

Dahan's dynamic-evaluation algorithm correctly splits quotient rings when a
subresultant coefficient is noninvertible or nilpotent. Its core remains a
represented subresultant sequence, and the paper does not supply an
output-sensitive simultaneous norm for this family of elliptic translations.
It therefore receives no below-rho constructor credit here.

Primary source:

- https://arxiv.org/abs/2010.14775

## Deduplication

R163 supplies the exact union semantics and `B^2` label/source-backpointer
verification. R164 changes the projector and isolates the incidence branch;
it does not claim a fast norm constructor.

R107 closes standard explicit canonical `5A+5C` resultant bodies at its
`B^(13/5)` interface. R164 instead targets an unlabeled C3 endpoint union in
a separate target-label algebra and does not reopen R107's multiplicity
interface.

## Admission

Twenty of twenty-seven obligations pass. Admit the collision-safe label
algebra, randomized linear residual, one-sided `nN/p` error bound, exact
verification, and incidence constructor.

Do not admit an output-sensitive regular norm constructor, deterministic
hash-to-curve transfer, unconditional generic-prime algorithm, Pollard-rho or
Shoup improvement, or ECDLP breakthrough.

Disposition:

```text
ADMIT_RANDOM_LINEAR_TARGET_RESIDUAL_WITH_NO_FALSE_NEGATIVES__FALSE_UNION_PROBABILITY_B_MINUS3O2__FORCED_FALSE_ROOT_VERIFIED_AWAY__INCIDENCE_BRANCH_B9O4__STANDARD_NORM_B7O2_OVER_RHO__OUTPUT_SENSITIVE_ELLIPTIC_TRANSLATION_NORM_OPEN__NO_RHO__NO_SHOUP_IMPROVEMENT__NO_BREAKTHROUGH
```

## Next action

Construct or refute the regular-branch norm
`gcd(U,Norm_target(a+R*b))` below `B^(5/2)`, preferably in
`B^(9/4+o(1))`, by exploiting that every target map is a translation of one
elliptic divisor. The constructor may return an unlabeled factor and must not
materialize the `n` by `N` residual table.
