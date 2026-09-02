# P1553 target-translated frequency-orbit probe R77

## Classification

- Owner: existing P1553 R3 Query2P1 and R76 subset-incidence frontier; no
  P1554 and no new idea ID.
- Evidence: exact finite group algebra and finite-field Fourier replay on four
  prime-order toy groups.
- Status: `REJECT_UNIVERSAL_LINEAR_TARGET_TRANSLATION_FREQUENCY_SKETCH_ONLY`.
- Labels: `exact-finite`, `linear-representation-bound`,
  `scalar-label-diagnostic`, `novelty-unverified`.
- Cryptanalytic result: no admitted source, relation campaign, factor-log
  solve, fresh-target descent, Shoup improvement, or ECDLP breakthrough.

R76 converts exact S6 endpoint incidence into an alternating inner product of
prefix and target-plus-pair subset frequencies. R77 asks whether the singleton
part can use one universal low-dimensional linear sketch for every target
translation. It grants the algorithm scalar labels and exact group characters
as a diagnostic advantage. Every frozen pair-query translation orbit still
has full ambient group rank.

This closes only universal exact linear shift-equivariant sketches. Nonlinear
target-specialized nested resultants and implicit circuits remain open.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R70 multiplicative-x report | `9e58c6178eb18b7535c59e117ad942465d6c7853890291dec2c7c0abdb4ffd89` |
| R76 subset-incidence report | `de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93` |
| R31 registry containing R3 Query2P1 gate | `0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f` |
| Bound R3 Query2P1 gate | `b2ee5934e295ab1f0d6b43452898e520d0cb18e718a8f5865694b25909b0df5e` |

## Exact reduction

Work temporarily in a prime-order cyclic group `G` of order `q`, and orient
each x-coordinate occurrence by including both point signs. Let

```text
f_i(P) = multiplicity of P in signed deck i.       (1)
```

For group convolution `*`, define the pair query kernel

```text
g = f_4 * f_5.                                     (2)
```

The signed five-list branch count for target `R` is exactly

```text
(f_1 * f_2 * f_3 * f_4 * f_5)(-R).                (3)
```

This is the singleton endpoint contribution underlying the R76 Kummer
incidence formula. Degenerate tuples with multiple common endpoints still
need R76's higher-subset correction.

## Linear orbit-rank lemma

A universal linear translation sketch must support inner products against
every translate of `g`. The circulant matrix whose rows are those translates
is diagonalized by the `q` group characters. Therefore

```text
rank{translations of g}
  = #{characters chi : Fourier(g)(chi) != 0}.      (4)
```

R77 verifies (4) against direct finite-field row reduction on an exact
seven-element control. It then computes every Fourier coefficient in a prime
field containing a primitive q-th root of unity. The Fourier modulus exceeds
the maximum branch mass, and integer convolution mass is checked separately.

## Frozen replay

The four R70 prime-order subgroups have orders

```text
103, 281, 163, 463.
```

For each group, R77 tests:

```text
multiplicative-x deck reused in all five positions,
five independently hashed decks,
B = 2,3,4,6,8.
```

All 40 instances satisfy:

```text
exact DFT(convolution) = product of deck DFTs,
integer branch mass = (2B)^5,
pair-query translation-orbit rank = q,
fivefold correlation Fourier support = q.         (5)
```

The target sequences contain 1,801 exact zero targets and 8,299 positive
targets in aggregate. Thus sparse or zero outcomes in some instances do not
create a missing character mode in the universal pair-query orbit.

These are exact finite controls, not an asymptotic theorem that every deck has
full Fourier support.

## Character boundary

R77 deliberately computes scalar labels by walking a toy generator. That is
not available to the proposed fresh-target algorithm. For prime `q`, any
nontrivial scalar-additive character representation is injective and
transports the target DLP to its character image. R3 already records that
integer 3SUM/kSUM indexing transplants need precisely such an unavailable map.

Even granting those labels, (5) requires all `q` character modes for a
universal exact linear translation sketch. That is ambient-group state, not
advice within the `B^(9/4+o(1))` cap. Without the grant, evaluating the modes
on a fresh target is itself unsupplied.

This is a representation result, not an unconditional data-structure or
arithmetic-circuit lower bound.

## R3 deduplication

R3 already rejects standard scalar-additive indexing, shifted pair-divisor
resultants, dynamic splitting, and target-label norms at their charged
representations. R77 adds an executable exact orbit-rank control for the
universal linear frequency-sketch subcase. It does not warrant a new idea ID.

## Scope

R77 does not address nonlinear target specialization, implicit
summation-polynomial resultants, Las Vegas data structures, support-restricted
advice, or value-sensitive circuits. It also omits R76's higher-subset
multiple-root correction from the Fourier reduction. Those routes remain
outside the rejection.

No factor-base relation matrix, independent rank, factor logs, or identical
fresh-target descent is supplied.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_target_translated_frequency_orbit_probe_r77.py` | `b5e863090aa036504d3c9fd7d47878e370c2c812f7f4e75b9b1d8d0ad972b335` |
| `p1553_target_translated_frequency_orbit_probe_report_r77.json` | `73f66184fa53a2d43397a915c4249a41c5687cbcd1d5d16cc3e4ff47cf254787` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_target_translated_frequency_orbit_probe_r77.py` | `165b60456efdc5af76308d696189badf3ebebb0938e0e4596ee85ac0c8da6398` |

Targeted replay:

```text
Ran 3 tests in 0.029s
OK
families=4
pair_orbits_full=True
convolution_exact=True
lane_admitted=False
```

## Disposition

```text
REJECT_UNIVERSAL_LINEAR_TARGET_TRANSLATION_FREQUENCY_SKETCH_ONLY__FOUR_PRIME_ORDER_TOY_GROUPS__MULTIPLICATIVE_AND_HASH_DECKS__B2_3_4_6_8__FULL_PAIR_QUERY_CHARACTER_SUPPORT__EXACT_CONVOLUTION__SCALAR_LABELS_GRANTED_ONLY_AS_DIAGNOSTIC__R3_BOUNDARY_SHARPENED__NONLINEAR_TARGET_SPECIALIZATION_OPEN__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Freeze one nonlinear target-specialized nested-resultant scalar functional for
the actual S4 deck factors. It must avoid group characters, `B^3` prefix
values, and `B^2` suffix materialization, while replaying R76 exact counts,
multiple-root correction, blind zero, one source, dyadic children, and the
direct caps.
