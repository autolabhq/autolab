# P1553 M6 output-sensitive nonlinear C5 source-index gate R119

## Claim boundary

R119 proves that the canonical five-sum endpoint support of an iid random
prime-cyclic deck is asymptotically occurrence-sized in the frozen campaign
regime. It also gives exact source and empty-query controls on all eight R82
hash decks. This closes output-linear endpoint dictionaries, radical/source
selectors, and image compilers at the setup cap. It does not prove a lower
bound for a sub-output nonlinear membership circuit or data structure.
It supplies no inside-cap locator, relation rank, factor logs, target descent,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
IID_RANDOM_PRIME_CYCLIC_C5_SUPPORT_IS_B15O4_WITH_HIGH_PROBABILITY_BY_PAIR_COLLISION_FIRST_MOMENT__ALL_EIGHT_R82_HASH_DECKS_HAVE_CANONICAL_C5_ENDPOINT_INJECTIVITY_AND_EXACT_SOURCE_EMPTY_REPLAY__OUTPUT_LINEAR_ENDPOINT_DICTIONARY_RADICAL_SELECTOR_AND_IMAGE_COMPILERS_EXCEED_B9O4_SETUP__CURRENT_KSUM_INDEXING_AND_KNOWN_LOWER_BOUNDS_DO_NOT_CLOSE_SUBOUTPUT_NONLINEAR_INDEX__IMPLICIT_POLYLOG_C5_MEMBERSHIP_SOURCE_CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Random-deck support theorem

Let `q>5` be prime and let `C_1,...,C_n` be independent uniform elements of
`Z/qZ`. A canonical five-source is a multiplicity vector

```text
a in Z_{\ge 0}^n,  sum_i a_i = 5.
```

There are

```text
M = binom(n+4,5)
```

such sources. For two distinct sources `a,b`, put `d=a-b`. Since every
coordinate of `d` has absolute value at most five and `q>5`, `d` is nonzero
modulo `q`. Choose a coordinate with `d_i != 0` and condition on all other
deck labels. Then

```text
sum_j d_j C_j
```

is uniform in `Z/qZ`, so the two sources collide with probability exactly
`1/q`. Therefore the expected number of colliding unordered source pairs is

```text
E X = binom(M,2)/q.                              (1)
```

If an endpoint fiber has size `f`, it contributes deficiency `f-1` and
collision count `binom(f,2)`, with `f-1 <= binom(f,2)`. Summing fibers gives

```text
M - |5C| <= X.                                  (2)
```

Markov's inequality and (1)-(2) yield

```text
Pr[|5C| < (1-epsilon)M]
    <= (M-1)/(2 epsilon q).                     (3)
```

In the campaign substitution

```text
n = B^(3/4+o(1)),
q = B^(5+o(1)),
M = B^(15/4+o(1)),
E X = B^(5/2+o(1)),
M/q = B^(-5/4+o(1)).
```

Thus `|5C|=(1-o(1))M` with probability `1-o(1)`. This is an iid
random-deck theorem. The R82 deck is filtered through an elliptic hash
construction, and no asymptotic transfer theorem to that process is claimed.

## Exact finite controls

An exhaustive control uses `q=11`, `n=3`, all `11^3=1331` label assignments,
and all `M=21` canonical five-sources. Across the `210` distinct source
pairs, the observed total collision count is

```text
11^2 * 210 = 25410,
```

exactly the `1/11` pair-collision prediction. Inequality (2) holds on every
assignment.

Eight projective controls use all four R82 prime-order subgroup families and
both offsets. Their deck sizes are `3,5,6,7`; their canonical five-source
counts are respectively `21,126,252,462`. Every endpoint map is injective,
every selected five-source replays as the same fixed-sign projective sum, and
an absent target is rejected exactly. Candidate scalar labels are not used.
These finite enumerations receive no asymptotic credit.

## Output-linear cost boundary

At the R118 interface, setup is at most `B^(9/4+o(1))` and each arbitrary
five-`C` membership/source query must use only polylogarithmic field work.
For a random deck:

- an explicit endpoint membership/source dictionary has
  `B^(15/4+o(1))` entries;
- the radical endpoint polynomial and any output-per-endpoint source
  selector have `B^(15/4+o(1))` coefficients;
- any compiler that emits the full radical image or trace has
  `B^(15/4+o(1))` output words;
- a universal characteristic-zero shift representation has `B^(5+o(1))`
  state by the inherited prime-cyclotomic rank theorem;
- a regular Boolean section has pole degree `B^(5+o(1))` if materialized;
- the current bound `k=6` indexing route has `B^(33/8+o(1))` state at zero
  polynomial query exponent.

These are charged realizations, not a general lower bound. In particular,
output support size alone does not lower-bound arithmetic-circuit size,
cell-probe space, RAM space, or query time.

## Preserved interface

The surviving object is a sub-output implicit nonlinear index:

```text
setup: B^(9/4+o(1)) state or less,
query: polylogarithmic field work,
input: an arbitrary projective target and the scalar-blind C deck,
output: exact empty or five C occurrence backpointers.
```

No unconditional theorem used by R119 excludes such an index with
`n^3=B^(9/4)` state. Static `kSUM` indexing lower bounds of the strength
needed here are not supplied, and no general circuit lower bound is claimed.

## Admission

Ten of seventeen obligations pass:

- twelve immutable source bindings;
- inherited exact R118 one-atom reduction;
- iid random-deck collision and support theorem;
- exact campaign exponent substitution;
- exhaustive iid finite collision control;
- eight R82 projective support controls;
- exact endpoint injectivity on those controls;
- exact source and empty-query replay;
- charged output-linear representations;
- explicit preservation of the implicit-index and general-lower-bound gaps.

The implicit membership circuit, implicit source recovery, known-RHS relation
rank, factor logs, identical target descent, Shoup improvement, and
breakthrough remain open.

Disposition:

```text
ADMIT_IID_RANDOM_DECK_C5_SUPPORT_THEOREM_AND_FINITE_R82_SOURCE_CONTROLS_ONLY__REJECT_OUTPUT_LINEAR_ENDPOINT_DICTIONARY_RADICAL_SELECTOR_AND_IMAGE_COMPILERS_AT_FROZEN_SETUP__PRESERVE_SUBOUTPUT_IMPLICIT_NONLINEAR_C5_MEMBERSHIP_SOURCE_CIRCUIT__NO_LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one sub-output implicit nonlinear `C5` membership/source
circuit with `B^(9/4+o(1))` state and polylogarithmic exact query. Freeze the
target computation graph, every `S6` or fixed-sign group-law branch and
remainder dimension, exact empty certification, and reverse five-source
recovery. The route receives no unit-cost `kSUM`, resultant, gcd, character,
DLP, root, or source oracle and must compose with R118 before receiving rank,
log, descent, memory, or asymptotic credit.

## Primary sources

- Dinur and Golovnev, *Improved Time-Space Tradeoffs for
  3SUM-Indexing*, <https://arxiv.org/abs/2512.04258>.
- Bhargava, Ghosh, Guo, Kumar, and Umans, *Fast Multivariate Multipoint
  Evaluation Over All Finite Fields*, <https://arxiv.org/abs/2205.00342>.
- Moroz and Schost, *A Fast Algorithm for Computing the Truncated
  Resultant*, <https://arxiv.org/abs/1609.04259>.
- Semaev, *Summation polynomials and the discrete logarithm problem on
  elliptic curves*, <https://eprint.iacr.org/2004/031>.
