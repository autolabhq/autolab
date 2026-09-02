# P1553 M6 A6-batched C3 pair-sum source-locator gate R116

## Claim boundary

R116 proves an exact interface reduction and rejects several bound
implementations at the frozen R115 exponents. It does not construct an
inside-cap relation locator, prove relation rank, solve factor logs, perform
target descent, improve Pollard rho, beat Shoup's generic lower bound, or
claim an ECDLP breakthrough.

Classification:

```text
M6_SELF_CONVOLUTION_EXACTLY_REDUCES_TO_A6_BATCH_OF_C3_PAIR_SUM_QUERIES__C3_STATE_B9O4_AND_BATCH_B1O2__AVERAGE_QUERY_ALLOWANCE_B3O4__DINUR_GOLOVNEV_ONLINE_BOUNDARY_SETUP_B39O8_AND_INTEGER_RESIDUE_TRANSFER_ABSENT__PREPROCESSED_UNIVERSE_N2_SETUP_FAIL__STANDARD_TRANSLATED_DIVISOR_BATCH_B11O4__PAIR_RESULTANT_B9O2__DENSE_FOURIER_B5__FINITE_PROJECTIVE_COUNTS_AND_SOURCES_EXACT__TRANSPOSED_COEFFICIENT_FUNCTIONAL_RANK_LOGS_DESCENT_OPEN
```

## Exact reduction

At the R115 vertex,

```text
N = B^(5+o(1))
|A| = B^(1/12+o(1))
|C| = B^(3/4+o(1))
F = A+C
m = 6.
```

Let `mu_X` denote the occurrence measure of a deck or endpoint list. Then
associativity and commutativity give the exact measure identity

```text
mu_F^*6
  = (mu_A^*3 * mu_C^*3) * (mu_A^*3 * mu_C^*3)
  = mu_A^*6 * (mu_C^*3 * mu_C^*3).               (1)
```

Writing `A6=mu_A^*6` and `C3=mu_C^*3`, the coefficient at a fresh target
`R` is

```text
sum_a6 A6(a6) (C3*C3)(R-a6).                     (2)
```

The exact sizes in `B` exponents are

```text
A3: 1/4
A6 target batch: 1/2
C3 persistent occurrence list: 9/4
C3+C3 occurrence body: 9/2
A6*C3*C3 full source body: 5.
```

The setup cap is saturated by `C3`. The fresh cap `5/4` leaves average
work `3/4` for each of the `B^(1/2)` correlated targets `R-a6`.

A positive source consists of one ordered `A6` backpointer and two ordered
`C3` backpointers. Concatenating the C sources and pairing positions returns
six coupled factors `F_i=A_i+C_i`.

## Finite projective controls

The producer runs three controls on the existing R82 prime-order projective
curve families:

| control | A deck | C deck | positive targets | empty targets |
| --- | ---: | ---: | ---: | ---: |
| `p98561...u2_v3` | 2 | 3 | 1 | 0 |
| labeled semantic subdeck | 1 | 2 | 2 | 1 |
| `p3148097...u3_v5` | 3 | 5 | 3 | 0 |

For every tested target, including the empty target, the exact integer
coefficients agree in all three forms:

```text
A6*C6
A6*(C3*C3)
(A3*C3)*(A3*C3).
```

Every positive result returns and replays:

- one direct `A6+C6` source;
- one `A6` plus two `C3` sources;
- two `3F` sources;
- the same target as twelve atoms and as six coupled factors;
- five nonprojective `S3` chain checks, all zero.

The sparse subdeck exists only to exercise an exact no-source branch. No
finite enumeration or subdeck receives asymptotic runtime credit.

## Current indexing tradeoffs

Put

```text
n = |C3| = B^(9/4+o(1))
Q = |A6| = B^(1/2+o(1)).
```

Dinur and Golovnev's Theorem 5.1 specializes for equal lists to

```text
S = soft-O(n^(5/2-delta))
T = soft-O(n^delta)
0 <= delta <= 1.
```

The whole batch fits fresh work only if

```text
1/2 + (9/4)delta <= 5/4,
delta <= 1/3.                                     (3)
```

At this online-compatible endpoint,

```text
S = B^((9/4)(5/2-1/3)) = B^(39/8)
T = B^(3/4)
Q*T = B^(5/4).                                   (4)
```

The setup cap would require `delta>=3/2`, outside the theorem. The paper's
efficient preprocessing is `soft-O(n^2)=B^(9/2+o(1))`, also over cap.

The older Fiat--Naor 3SUM-indexing curve has setup `B^(17/4)` at the same
online boundary. Storing the list and scanning costs `B^(11/4)` for the
batch; storing every pair sum costs `B^(9/2)` setup.

Kirkpatrick, Kuszmaul, Mathialagan, and Vassilevska Williams prove for
preprocessed 3SUM with unknown universes

```text
preprocessing n^2
space n^(2-2*epsilon/3)
query n^(3/2+epsilon)
0 <= epsilon <= 1/2.
```

Even its minimum-space endpoint is `B^(15/4)` with `B^(9/2)`
preprocessing. Its model is also a stronger, different query involving
subsets and an unknown list, not the single point challenge in (2).

These are current-algorithm controls. R116 does not infer a polynomial
data-structure lower bound.

## Prime-order transfer boundary

The bound indexing constructions use additive integer residues and, in the
2026 preprocessed-universe route, small-modulus FFT convolution. A homomorphism
from the prime-order group of order `q` to a proper residue group whose order
is not divisible by `q` is trivial. Public elliptic-curve coordinate encodings
are not addition-compatible.

Discrete-log labels would restore the needed additive residues, but those
labels are the unavailable object. The finite point-hash controls therefore
receive semantic credit only, not transfer credit for the integer algorithms.

## Explicit algebraic routes

| route | charged exponent in `B` | result |
| --- | ---: | --- |
| store explicit `C3` endpoint divisor | `9/4` state | fits setup, no pair locator |
| translate/intersect it for one target | `9/4` fresh | over fresh cap |
| translate/intersect for all `A6` targets | `11/4` fresh | over rho and fresh cap |
| materialize `C3+C3` resultant/Chow body | `9/2` degree/state | over setup |
| materialize `C3+C3+A6` incidence body | `5` occurrence degree | group-order body |
| sparse group-algebra square | `9/2` output | over setup |
| dense group Fourier transform | `5` modes | over setup |
| balanced `3F | 3F` join | `5/2` work/state | exactly rho |

This closes only standard explicit coefficient, translated-divisor,
output-materialized resultant, sparse group-algebra, and dense Fourier
realizations. It is not a lower bound against an implicit arithmetic circuit
or transposed FFE operator.

## Summation-polynomial boundary

The factor-level relation uses `S7`, with degree 32 in each variable and
total degree 192. Expanding six `F=A+C` factors gives `S13`, with degree 2048
in each variable and total degree 24576.

These fixed degrees have zero `B` exponent, but they do not evaluate (2),
avoid the `C3+C3` body, or return source backpointers. R116 preserves exactly
one representation-specific opening:

```text
a jointly transposed target-batched elliptic coefficient functional
with a source adjoint.
```

## Admission

Nine of sixteen obligations pass:

- twelve immutable source bindings;
- exact R115 vertex and endpoint exponents;
- exact finite three-form coefficient identity;
- positive and empty source semantics;
- six coupled factor-source replay;
- exact current-indexing exponent substitution;
- current bound indexing routes rejected at the frozen caps;
- standard explicit algebraic costs charged.

The missing obligations are the implicit coefficient functional, known-RHS
rank, factor logs, identical target descent, complete source-to-target cost,
Shoup improvement, and breakthrough.

Disposition:

```text
ADMIT_EXACT_A6_BATCHED_C3_PAIR_SUM_REDUCTION_AND_FINITE_SOURCE_REPLAY_ONLY__REJECT_CURRENT_BOUND_INDEXING_AND_STANDARD_EXPLICIT_DIVISOR_RESULTANT_GROUP_ALGEBRA_ROUTES_AT_FROZEN_CAPS__PRESERVE_TARGET_BATCHED_TRANSPOSED_ELLIPTIC_COEFFICIENT_FUNCTIONAL__NO_LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one exact target-batched elliptic coefficient functional
for (2). It may store the `B^(9/4)` `C3` endpoint divisor and enumerate the
`B^(1/2)` `A6` target batch, but it must:

- use at most `B^(5/4+o(1))` fresh work and workspace;
- never emit the `B^(9/2)` pair-sum body or `B^(11/4)` translation batch;
- use no DLP labels, small-residue homomorphism, or determinant oracle;
- return one `A6` and two `C3` occurrence backpointers on every positive
  projective branch;
- use the same operator for relation collection and target descent;
- pass known-RHS rank and independently verified factor-log gates.

## Primary sources

- Dinur and Golovnev, *Improved Time-Space Tradeoffs for
  3SUM-Indexing*, `arXiv:2512.04258v2`,
  <https://arxiv.org/abs/2512.04258>.
- Kirkpatrick, Kuszmaul, Mathialagan, and Vassilevska Williams,
  *Preprocessed 3SUM for Unknown Universes with Subquadratic Space*,
  `arXiv:2602.11363v1`, <https://arxiv.org/abs/2602.11363>.
- Semaev, *Summation polynomials and the discrete logarithm problem on
  elliptic curves*, <https://eprint.iacr.org/2004/031>.
