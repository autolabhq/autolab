# P1553 S4 centered-carry rank probe R71

## Classification

- Owner: existing P1553 R7/R8 and IDEA-049/IDEA-198 frontier; no P1554.
- Evidence: exact deterministic four-family computation with two target types.
- Status: `REJECT_FROZEN_CANONICAL_S4_LIFT`.
- Labels: `exact-toy`, `representation-bound`, `scalar-blind-source`,
  `novelty-unverified`.
- Cryptanalytic result: no relation campaign, branch-complete root locator,
  factor-log solve, scalar-blind descent, Shoup-bound improvement, or ECDLP
  breakthrough.

R8 proved that constant modular CP rank does not universally imply a
low-rank canonical bounded remainder or carry, but it deliberately left the
actual centered elliptic tensors unresolved. R71 evaluates the first elliptic
predecessor directly. It freezes three legal x-class decks and a target, forms

```text
F_R(x_1,x_2,x_3)
  = Res_z(S_3(x_1,x_2,z), S_3(x_3,x(R),z))
  = S_4(x_1,x_2,x_3,x(R)),

R_R = ctr_p(F_R),
K_R = (F_R-R_R)/p,
```

and measures mode-flattening ranks of the raw lift, centered remainder, and
carry on every prefix of size `3` through `8`.

The frozen result is negative for this representation. Every mode of every
carry tensor has full row rank on every tested prefix, for all four curves and
both an independent hash target and a forced-positive target. The same is true
of every centered remainder. At size eight, the raw integer lift has exact
mode rank five, while the centered remainder and carry have mode rank eight.

This rejects the canonical target-specific `S4`, `k=1` centered-carry
precursor on the named legal grids. It is not an asymptotic theorem for `S6`,
another lift, a shared-correlation algorithm, or a non-CP
unit-or-zero-divisor interface.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R8 integer-valued quotient gate | `78c856187bb43adcd97d0f02f2259c7299874ab93a03954dacb4dd1b8b007ed9` |
| R7 tensorized small-root gate | `b36870eeb0c7c0a53e6d1714d623629c522f4c58cd308a072e33ea6046a06615` |
| R70 four-family curve report | `9e58c6178eb18b7535c59e117ad942465d6c7853890291dec2c7c0abdb4ffd89` |

## Frozen construction

The four short-Weierstrass families are

| Family | `p` | `A` | `B` | prime subgroup `q` | cofactor |
|---|---:|---:|---:|---:|---:|
| `p193_a2_b3_q103_h2` | 193 | 2 | 3 | 103 | 2 |
| `p257_a1_b7_q281_h1` | 257 | 1 | 7 | 281 | 1 |
| `p337_a1_b3_q163_h2` | 337 | 1 | 3 | 163 | 2 |
| `p449_a1_b3_q463_h1` | 449 | 1 | 3 | 463 | 1 |

For each family, three domain-separated SHA-256 x-coordinate streams are
mapped to canonical square roots and publicly cofactor-cleared. Duplicate
x-classes are discarded. No scalar label or discrete logarithm is consumed.
Each deck has eight points and supplies frozen prefixes of sizes `3,...,8`.

Two targets are tested:

1. `blind_hash_target`, from an independent SHA-256 x-coordinate stream; and
2. `forced_positive_target`, the negative sum of one frozen point from each
   deck, with the witness recorded before the rank computation.

The second target guarantees that the positive branch is exercised. The first
retains an independent target control.

## Exact S4 replay

For

```text
S_3(x,y,z) = a_2 z^2 + a_1 z + a_0,

a_2 = (x-y)^2,
a_1 = -2((x+y)(xy+A)+2B),
a_0 = (xy-A)^2-4B(x+y),
```

R71 computes the quadratic resultant in two independent exact forms:

```text
Res = (a_2 f_0-a_0 f_2)^2
      -(a_2 f_1-a_1 f_2)(a_1 f_0-a_0 f_1)
```

and the determinant of the `4 x 4` Sylvester matrix by fraction-free Bareiss
elimination. All `8 * 8^3 = 4096` target-tuples agree between the two forms.

For each tuple, the run also enumerates the eight normalized sign choices in

```text
epsilon_1 P_1 + epsilon_2 P_2 + epsilon_3 P_3 + R = O.
```

The group relation exists exactly when the resultant vanishes modulo `p`.
There are zero predicate mismatches. Root counts on the full grids are:

| Family | blind target | forced-positive target |
|---|---:|---:|
| `p193_a2_b3_q103_h2` | 44 | 51 |
| `p257_a1_b7_q281_h1` | 13 | 15 |
| `p337_a1_b3_q163_h2` | 29 | 27 |
| `p449_a1_b3_q463_h1` | 3 | 8 |

Every carry division is exact, and every recorded forced witness verifies.

## Rank certificate

For each three-way value tensor and each mode, R71 flattens the tensor into a
`B x B^2` matrix and row-reduces it modulo the distinct auxiliary primes
`1000003` and `1000033`.

If a matrix has rank `r` modulo either prime, an `r x r` minor is nonzero as an
integer. Therefore its rational matrix rank is at least `r`, and the rational
CP rank of the value tensor is at least `r`. Evaluating a separated
coefficient representation on the legal grid is factorwise, so this value
rank lower-bounds any exact rational CP representation agreeing on that grid.
Invertible factorwise rational label-basis transforms preserve the same
flattening ranks.

Canonical recentering is nonlinear over the rationals. R71 measures after that
operation rather than transporting rank through it.

The aggregate certificate is:

```text
families                              4
target instances                     8
prefix instances                     48
mode flattenings per tensor kind     144

carry modes full under both primes   144 / 144
remainder modes full under both      144 / 144
largest carry instances all-full       8 / 8
largest remainder instances all-full   8 / 8
```

Every carry and remainder mode rank follows the exact prefix sequence

```text
B =       3  4  5  6  7  8
rank =    3  4  5  6  7  8.
```

At `B=8`, every raw-lift mode has modular rank five under both auxiliary
primes. Because fixed-target `S4` has degree four in each variable, its raw
value tensor has rational mode rank at most five. Thus the raw rank is exactly
five, while the centered remainder and carry have certified rational mode
rank eight.

A constant all-ones tensor is replayed at every prefix and has mode rank one,
so the measurement pipeline preserves the positive low-rank control.

## Consequence for R7/R8

For every frozen instance, any exact rational CP representation of the
canonical carry value tensor has rank at least `B`. This is incompatible with
treating that tensor as the tiny-rank cancellation object required by R7's
naive pairwise CP-Gram path, whose asymptotic accounting requires total
represented rank

```text
A <= B^(1/8+o(1)).
```

The finite probe does not prove an asymptotic contradiction: hidden constants,
different deck families, and the transition to `S6` remain unbounded by these
small instances. It does establish that the hoped-for low rank is absent from
the first actual elliptic predecessor, not merely from R8's synthetic modular
multiplication control.

The surviving R8 exceptions are now narrower:

- an `S6`-specific cancellation absent from the `S4` predecessor;
- a noncanonical bounded lift with fully charged integer height;
- a shared-correlation or non-CP implicit module; or
- an exact unit-or-zero-divisor router that meets the direct setup, state,
  online, workspace, and source-recovery caps without representing this carry.

No such object is supplied here.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_s4_centered_carry_rank_probe_r71.py` | `1b22d8bb7ab19c686c7945dd7c0e13a1575031173ce8b262ee39fb3799d6931b` |
| `p1553_s4_centered_carry_rank_probe_report_r71.json` | `6ee62d61f6c567a5a65947b6b47026182bd961565dde3c1aee540cf3cbe5b72f` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_s4_centered_carry_rank_probe_r71.py` | `30ba5c3561cf7a4dac4d17d5fb87f34e18931f63376e8ba2146032f5d88af556` |

Targeted replay:

```text
Ran 3 tests in 0.140s
OK
```

## Disposition

```text
REJECT_FROZEN_CANONICAL_S4_K1_CENTERED_CARRY__FOUR_GENERIC_PRIME_FIELD_FAMILIES__THREE_SCALAR_BLIND_LEGAL_X_CLASS_DECKS__BLIND_AND_FORCED_POSITIVE_TARGETS__EXACT_QUADRATIC_RESULTANT_EQUALS_BAREISS_SYLVESTER_DETERMINANT__S4_ZERO_IFF_SIGNED_GROUP_RELATION_ON_ALL_4096_TUPLES__RAW_B8_MODE_RANK_EXACTLY_5__CENTERED_REMAINDER_AND_CARRY_MODE_RANK_FULL_B_FOR_EVERY_B3_THROUGH_B8_PREFIX__BOTH_AUXILIARY_PRIMES__RATIONAL_CP_RANK_AT_LEAST_B_ON_EVERY_FROZEN_CARRY__R7_TINY_CP_CANCELLATION_NOT_OBSERVED__FINITE_REPRESENTATION_BOUND_ONLY__NO_S6_ASYMPTOTIC_THEOREM__NONCANONICAL_AND_NON_CP_INTERFACES_OPEN__NO_UNIT_OR_ZERO_DIVISOR_ROUTER__NO_FACTOR_LOG_SOLVE__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Stop allocating experiments to the canonical `k=1` centered-carry CP route.
For the live R8 exception, specify one exact branch-complete
unit-or-zero-divisor interface for the five-label `S6` predicate that either
proves an `S6`-specific cancellation absent from this `S4` predecessor or
avoids explicit centered-carry representation entirely. Before any run, expose
its target-independent state, target specialization, exact empty-fiber test,
positive-child preservation, occurrence-labelled source recovery, integer
heights, and direct setup/online/workspace costs. Reject it at the interface
stage if any standard `B^3` or `B^5` object is hidden.
