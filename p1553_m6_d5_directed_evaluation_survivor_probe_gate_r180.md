# P1553 M6 D5 directed-evaluation survivor gate R180

Date: 2026-08-01

## Scope

R180 tests whether D5 splitting or directed panoramic evaluation turns the
factored R174/R178 signed target stream into a softly `O(n+N)` constructor in
the squarefree product algebra

```text
A = F[X]/(U),  deg U = n.
```

The literal interface does not cross the rho boundary. D5 can retire candidate
components as soon as a target-norm factor becomes a nonunit, and directed
evaluation can avoid repeated split recomputation, but every noncandidate
component survives every one of the `N` target factors. A one-shot monogenic
finite-field modular-composition compiler remains open. This is not an
arithmetic-circuit lower bound, an ECDLP algorithm, or a Pollard-rho or Shoup
improvement.

## Exact Decomposition

For each retained target `T_j`, R180 constructs the R174 signed deflated-line
norm

```text
g_j mod U.
```

All six controls verify

```text
gcd(U, product_j g_j) = G_1,
```

with the same candidate-factor hash as both R174 and R178. They also verify that
the union of the zero roots of the individual `g_j` is exactly the root set of
`G_1`.

The finite transcript contains:

```text
selected divisor degree sum:              202
target factor count sum:                    40
candidate degree sum:                      140
noncandidate survivor degree sum:           62
target-factor zero incidences:             241
materialized factor-residue slots:       1,486
finite signed pair evaluations:          68,326
```

The materialized factors and pair evaluations are controls only and receive no
asymptotic credit.

## Optimal Splitting

R180 exhaustively optimizes the target order in each finite fixture. A factor is
tested only on the current survivor modulus; a gcd split is recorded only when
the factor is an actual nonunit. The natural order costs `869` component-factor
visits, while the exact optimum costs `762`.

Let `c=deg G_1`. Every root of `U/G_1` is a unit for every target factor, so it
cannot be retired at any step. Consequently every order satisfies

```text
component-factor visits >= (n-c)N.
```

The six finite lower bounds total `466`, and every optimum respects them. This
is a lower bound for the literal factor-by-factor computation tree only.

At campaign scale:

```text
n:                                      B^(9/4)
N target factors:                       B^(5/4)
c = deg G_1:                            B^(3/4)
n-c:                                    Theta(B^(9/4))
literal factor stream:                  B^(7/2)
successive D5 zero tests:               B^(7/2)
directed evaluation of the same tree:   B^(7/2)
standard D5 half-GCD:                   B^(9/2)
rho proxy:                              B^(5/2)
```

## Literature Boundary

Dahan, Moreno Maza, Schost, and Xie give quasi-linear multiplication,
quasi-inversion, splitting, and half-GCD machinery over direct products of
fields. Their bound preserves the polynomial-degree factor in a polynomial
GCD over the product algebra.

Van der Hoeven and Lecerf remove the repeated-splitting overhead through
directed evaluation. Their one-parameter Theorem 4.2 still charges the
computation tree's `tau_mul` multiplications and `tau_div` zero tests or
inversions by the cost of arithmetic in the degree-`n` algebra. For this tree,
those counts retain the `N` factor.

The cited 2026 generic algebraic modular-composition bound `O(n^1.343)` becomes
`B^3.02175` at `n=B^(9/4)`, still above rho. Finite-field near-linear bit
complexity is not closed: it could become relevant if the elliptic
two-parameter kernel is first compiled into one univariate composition without
emitting `N` residue elements.

## Admission

Admit the target-factor decomposition, all six R174/R178 candidate replays, the
optimal early-split transcripts, the noncandidate-survivor invariant, and the
published D5 and directed-evaluation cost specializations.

Do not infer a lower bound against one-shot monogenic, bivariate
modular-composition, transposed, or custom arithmetic-circuit constructions. No
generic-prime transfer, factor logs, target descent, complete attack, or Shoup
improvement is supplied.

Disposition:

```text
ADMIT_D5_DIRECTED_EVALUATION_INTERFACE_SCOPE__SIX_R174_R178_TARGET_FACTOR_CONTROLS_REPLAYED__OPTIMAL_EARLY_NONUNIT_SPLITS_EXACT__EVERY_NONCANDIDATE_COMPONENT_SURVIVES_ALL_N_FACTORS__LITERAL_FACTOR_STREAM_AND_DIRECTED_EVALUATION_B7O2__STANDARD_D5_HALF_GCD_B9O2__GENERIC_ALGEBRAIC_MODCOMP_B3O02175__ONE_SHOT_MONOGENIC_FINITE_FIELD_FOLD_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute one one-shot monogenic compiler for the signed elliptic
translate product. From compact `U,V` and the degree-`N` target Miller SLP,
derive `H,a` with `C_h=H(a) mod U`, or a bounded-bidegree `G(X,a(X)) mod U`, in
softly `O(n+N)` preprocessing without emitting `N` residue elements. Apply a
charged finite-field modular-composition algorithm and verify `G_1` on held-out
divisors. Reject a disguised `N`-step product-algebra tree, `nN` coefficients,
`n^2` pair state, candidate inversions, and unit-cost composition, norm,
resultant, root, count, marginal, rank, or source oracles.
