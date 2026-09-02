# P1553 M6 Kummer x-translate and signed-verification gate R166

Date: 2026-08-01

## Scope

R166 tests the deterministic `r=0` specialization of R165. It removes the
random scalar and the `V`-dependent y residual from the fixed translated
function. The signed divisor polynomial `V`, or its equivalent y side table,
remains necessary to represent the selected sign of `P` during translation
and is reused for exact post-verification.

The finite producer explicitly evaluates endpoint-target pairs. Those tables
prove the identities below but receive no asymptotic attack credit.

## Kummer relaxation

Let `S` be the selected signed C3 endpoint set and

```text
U(X) = product_(P in S) (X-x(P)),  n=deg(U).
```

The elliptic function

```text
f_0(Q) = U(x(Q))
```

has a pole of order `2n` at `O` and zero divisor

```text
S + (-S).
```

Consequently, for an affine translated point `Q=T_j-P`,

```text
U(x(T_j-P)) = 0
```

exactly when `T_j-P` is a selected signed C3 endpoint or its negative. The
first orientation is a true positive-C6 split. The second is an x-only
false-sign branch. No true signed match is lost.

When `P=T_j`, the translated point is the pole `O`, which is outside the
affine C3 set; that factor is assigned semantic value one. The global group
translation also handles x-incidence and tangent cases without a separate
membership oracle.

The required candidate factor is therefore

```text
G_0 = gcd(U, product_j U(x(T_j-P)) mod U).
```

Every root of `G_0` is checked against the signed y-coordinate dictionary.
This removes all opposite-sign-only roots exactly.

## False-sign density

For a planted positive target, write

```text
T = v.C,  ||v||_1=6,
P = p.C,  ||p||_1=3,
Q = q.C,  ||q||_1=3,
```

with nonnegative coefficient vectors. A false-sign equation `T=P-Q` is

```text
(v-p+q).C = O.
```

The integer coefficient sum of `v-p+q` is six, so the form cannot be zero.
Its l1 norm is at most twelve. For independent uniform cyclic C labels and
prime subgroup order above that coefficient bound, each fixed form vanishes
with probability exactly `1/q`. A union bound over `n^2 K` planted pairs gives

```text
E[planted opposite-sign pairs] <= n^2 K/q = B^(1/4+o(1)).
```

For independent uniform unplanted targets, each fixed equality to `P-Q` also
has probability `1/q`. Across the full target cap,

```text
E[unplanted opposite-sign pairs] <= n^2 N/q = B^(3/4+o(1)).
```

The true orientation has the same expected scale on unplanted targets, while
planted targets have at most 20 occurrence splits per unique source. Thus the
expected candidate-root count is `B^(3/4+o(1))`. Markov gives a
`B^(3/4+epsilon)` cap except with probability `B^(-epsilon)`.

Scanning all `N=B^(5/4)` target labels for every candidate costs

```text
B^(3/4) B^(5/4) = B^2,
```

strictly below the `B^(5/2)` rho proxy. This statement is proved only in the
independent uniform cyclic-label/target model. Transfer to the deterministic
hash-to-curve sampler remains open.

## Finite controls

The six inherited target batches contain:

```text
true signed endpoint-target pairs: 241
opposite-sign pairs:                  0
opposite-sign-only roots:             0
```

All six Kummer candidate factors equal the union of the true and opposite
orientations. Signed verification returns the exact R163 aggregate union.
The zero observed false-sign count is recorded without probability credit.

Six additional controls use 4,096 deterministic pseudorandom nonzero subgroup
scalars per curve/seed. They observe:

```text
true-orientation pairs:      100
opposite-orientation pairs:   89
combined iid expectation per orientation: about 85
```

These are density calibrations, not a proof about hash-to-curve points. A
deliberate `T=P-Q` control creates six opposite-sign candidate roots and no
true roots; signed verification removes all six.

## Degree and cost boundary

The standard represented translate product has zero and pole divisor degrees

```text
2nN = B^(7/2),
```

one exponent above rho. Removing `r` and the y residual from the fixed function
sharpens the interface but does not eliminate the signed-divisor input or
compress this product. No explicit divisor, `n` by `N` value table, unit-cost
norm, 3SUM oracle, root oracle, or source oracle receives credit.

The open primitive is an output-sensitive arbitrary-target Kummer translate
product remainder modulo `U`, preferably in `B^(9/4+o(1))` work and in all
cases strictly below `B^(5/2)` total work.

## Deduplication

R165 supplies a randomized signed elliptic function. R166 specializes to the
deterministic x-only function and fully charges its opposite-sign branch.

R158 concerns collisions among nonnegative l1-six sources. R166's forms
`v-p+q` are different signed forms; no deterministic sampler transfer is
inherited from R158.

R152 explicitly warns that x-only Semaev elimination introduces sign
branches. R166 retains that boundary and adds an exact output verifier plus an
iid-model density theorem.

R161 enforces the selected sign with `U,V` during each target composition.
R166 still uses `V` to translate the selected signed divisor, but the fixed
function and product zero test use `U` alone; exact sign filtering moves to the
output-sized candidate set.

R148 closes standard static 3SUM indexing tradeoffs at the campaign caps.
R166 asks for a coordinate-specific Kummer remainder operator, not a generic
unit-cost 3SUM data structure.

## Admission

Twenty-four of thirty-one obligations pass. Admit the Kummer zero-divisor
identity, exact candidate biconditional, nonzero planted signed forms, iid
false-sign density bound, expected `B^2` verifier, and exact finite controls.

Do not admit an output-sensitive Kummer translate-product constructor,
deterministic hash-to-curve transfer, unconditional generic-prime algorithm,
Pollard-rho or Shoup improvement, or ECDLP breakthrough.

Disposition:

```text
ADMIT_KUMMER_X_RELAXATION__ZERO_DIVISOR_S_PLUS_NEGATIVE_S__TRUE_MATCHES_RETAINED__OPPOSITE_SIGN_BRANCH_VERIFIED_AWAY__IID_FALSE_PAIR_OUTPUT_B3O4__EXPECTED_SIGNED_VERIFIER_B2__EXPLICIT_TRANSLATE_DIVISOR_B7O2__OUTPUT_SENSITIVE_KUMMER_REMAINDER_AND_HASH_TRANSFER_OPEN__NO_RHO__NO_SHOUP_IMPROVEMENT__NO_BREAKTHROUGH
```

## Next action

Construct or refute `gcd(U,product_j U(x(T_j-P)))` below `B^(5/2)`, preferably
in `B^(9/4+o(1))`, without expanding the degree-`2nN` divisor or the `n` by
`N` value table. Separately prove the signed-difference density transfer for
the deterministic hash-to-curve sampler.
