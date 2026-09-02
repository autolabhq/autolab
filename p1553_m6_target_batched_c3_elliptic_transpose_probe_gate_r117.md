# P1553 M6 target-batched C3 elliptic-transpose gate R117

## Claim boundary

R117 proves a prime-cyclotomic rank theorem for the R116 six-factor count,
charges a regular rational target-section representation and two current
`k=7` indexing routes, and verifies the conclusions on finite projective
controls. It does not prove a base-field nonlinear circuit or data-structure
lower bound. It supplies no inside-cap source locator, known-RHS rank, factor
logs, target descent, Pollard-rho improvement, Shoup improvement, or ECDLP
breakthrough.

Classification:

```text
PRIME_CYCLOTOMIC_THEOREM_FORCES_FULL_B5_CHARACTERISTIC_ZERO_TRANSLATION_RANK_FOR_SIX_FACTOR_COUNT__CANONICAL_SOURCE_SUPPORT_LEAVES_1_MINUS_1O518400_ZERO_FRACTION_AND_REGULAR_SECTION_POLE_DEGREE_B5__FINITE_R82_AUXILIARY_FIELD_DFT_AND_SPARSE_FULL_RANK_CONTROLS_EXACT__ORIGINAL_C_DECK_K7_INDEX_AT_ONLINE_DELTA1_NEEDS_B33O8_STATE__GENERAL_BASE_FIELD_NONLINEAR_DATA_STRUCTURE_AND_CIRCUIT_LOWER_BOUND_NOT_CLAIMED__VALUE_SENSITIVE_SOURCE_LOCATOR_RANK_LOGS_DESCENT_OPEN
```

## Frozen interface

R116 reduces the six-factor relation coefficient to

```text
sum_a6 mu_A6(a6) (mu_C3*mu_C3)(R-a6),            (1)
```

with the following exponents:

```text
q = B^(5+o(1))
|A| = B^(1/12+o(1))
|C| = B^(3/4+o(1))
A6 target batch = B^(1/2+o(1))
C3 persistent state = B^(9/4+o(1))
fresh batch cap = B^(5/4+o(1)).
```

Thus the average allowance for each of the `B^(1/2)` translated targets is
`B^(3/4+o(1))`. A positive answer must return one `A6` and two `C3`
occurrence backpointers.

## Prime-cyclotomic rank theorem

Let `D` be a nonempty proper subset of `Z/qZ`, where `q` is prime, and let
`zeta` be a primitive complex `q`th root. For every nonzero frequency `r`,

```text
sum_(d in D) zeta^(r d) != 0.                    (2)
```

Indeed, if the proper `0/1` deck polynomial vanished at `zeta^r`, it would
be divisible over the rationals by

```text
Phi_q(X) = 1 + X + ... + X^(q-1).
```

Both polynomials have degree below `q`; divisibility would force all `q`
coefficients of the deck polynomial to be equal, contradicting that `D` is
nonempty and proper. The zero-frequency coefficient is `|D|`, so it is also
nonzero.

For the ordered six-plus-six endpoint count

```text
h = mu_A^*6 * mu_C^*6,
hat(h)(r) = hat(mu_A)(r)^6 hat(mu_C)(r)^6.        (3)
```

Every characteristic-zero Fourier mode is nonzero. The circulant matrix of
all target translations therefore has rank `q`, corresponding to
`B^(5+o(1))` exact characteristic-zero linear state.

This theorem concerns universal exact linear shift-equivariant
representations in characteristic zero. It is not a lower bound for
nonlinear algorithms, branching data structures, base-field arithmetic
circuits, or target-specialized source locators.

## Sparse support and rational sections

The number of canonical unordered six-`A`, six-`C` sources is

```text
K = binom(|A|+5,6) binom(|C|+5,6)
  = (1/(6!)^2 + o(1)) q
  = (1/518400 + o(1)) q.                         (4)
```

Consequently the Boolean existence function is zero on at least
`(1-1/518400+o(1))q` targets. A nonzero rational function on the elliptic
curve that is defined at the tested targets and agrees with this Boolean
function has at least that many rational zeros. Equality of zero- and
pole-divisor degrees forces pole degree `Omega(q)=B^(5+o(1))`.

This rejects materializing one such regular rational target section inside
the setup cap. Pole degree is not straight-line-program size: repeated
squaring and other nonlinear circuits can produce high degree succinctly.
No arithmetic-circuit lower bound is claimed.

## Finite controls

R117 uses the first R82 prime-order projective subgroup:

```text
field prime p = 98561
subgroup order q = 16427
cofactor = 6.
```

Scalar labels are computed only by the verifier. DFT checks are performed in
the auxiliary prime field `F_98563`, which contains a `q`th root of unity.

| control | ordered sources | canonical sources | support | zero targets | DFT rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| `u1_v2` | 64 | 7 | 7 | 16420 | 16427 |
| `u2_v3` | 46656 | 196 | 196 | 16231 | 16427 |

In both controls:

- direct DFT coefficients equal the product of the sixth powers of the two
  deck DFTs at every frequency;
- no target-count frequency vanishes;
- ordered and canonical source sets have identical endpoint support;
- the translation orbit has full rank despite sparse endpoint support.

These finite labels, DFTs, and enumerations receive no asymptotic algorithmic
credit.

## Current `k=7` indexing costs

Viewing a query as a six-sum over the original `C` deck gives `k=7` and

```text
n = |C| = B^(3/4+o(1)).
```

Dinur and Golovnev's bound theorem gives

```text
S = soft-O(n^(k-1/2-delta))
T = soft-O(n^delta)
0 <= delta <= 1.
```

The largest allowed `delta=1` is required to attain per-target query
`B^(3/4+o(1))`. It yields

```text
S = n^(11/2) = B^(33/8+o(1))
|A6| T = B^(1/2+3/4+o(1)) = B^(5/4+o(1)).
```

The online batch fits exactly, but setup exceeds `B^(9/4+o(1))`. The trivial
exact-source route that stores all five-`C` sums has state
`B^(15/4+o(1))` and the same online cost, so it also misses setup. These are
current-algorithm exclusions, not unconditional indexing lower bounds, and
integer-residue algorithms still do not transfer through public
prime-order elliptic-curve coordinates without unavailable DLP labels.

## Admission

Ten of sixteen obligations pass:

- twelve immutable source bindings;
- inherited exact R116 source semantics;
- exact finite convolution/DFT factorization;
- exact finite full translation rank, including sparse support;
- complete prime-cyclotomic nonvanishing proof;
- canonical source-support bound;
- regular-section zero/pole-degree bound;
- charged original-deck `k=7` indexing cost;
- explicit exclusion of general circuit and data-structure lower bounds.

The nonlinear value-sensitive source locator, known-RHS rank, factor logs,
identical target descent, Shoup improvement, and breakthrough remain open.

Disposition:

```text
REJECT_UNIVERSAL_CHARACTERISTIC_ZERO_LINEAR_SHIFT_SKETCH_AND_SINGLE_REGULAR_RATIONAL_TARGET_SECTION_ONLY__FINITE_AUXILIARY_DFT_FULL_RANK_AND_SOURCE_SUPPORT_CONTROLS_EXACT__REJECT_BOUND_K7_INTEGER_INDEX_AND_STANDARD_TRANSLATED_DIVISOR_AT_FROZEN_CAPS__PRESERVE_NONLINEAR_VALUE_SENSITIVE_SIX_C_SOURCE_LOCATOR__NO_LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one nonlinear, value-sensitive six-`C` source locator on
the original `C` deck. It may retain the `B^(9/4+o(1))` `C3` preprocessing
and spend `B^(3/4+o(1))` per `A6` target, but it must:

- avoid a universal linear shift sketch and a single regular target section;
- avoid the `B^(33/8)` bound `k=7` index and `B^(11/4)` translated-divisor
  batch;
- freeze every branch, `S7`/FFE remainder or subresultant dimension, and
  false-positive verification cost;
- return one exact six-`C` source, then compose it with the `A6` source;
- use the identical operator for relation collection and target descent.

## Primary sources

- Dinur and Golovnev, *Improved Time-Space Tradeoffs for
  3SUM-Indexing*, `arXiv:2512.04258v2`,
  <https://arxiv.org/abs/2512.04258>.
- Semaev, *Summation polynomials and the discrete logarithm problem on
  elliptic curves*, <https://eprint.iacr.org/2004/031>.
