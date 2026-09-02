# P1553 5A5C Nonlocal Moment/Hankel Translation Probe Gate R90

## Claim boundary

No generic-prime-field ECDLP algorithm, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, or target descent is claimed.

Classification:

```text
NONLOCAL_MOMENT_UPDATE_EXACT__HANKEL_PADE_ORDER_C5_B3_OVER_CAP
```

## Frozen candidate

For five C decks, freeze the target-independent exponential-moment channels

```text
G(z)   = product_i sum_j exp(c_ij z) mod z^K
G_i(z) = (sum_j (j+1) exp(c_ij z))
         product_(h!=i) sum_j exp(c_hj z) mod z^K.
```

Deck updates are nonlinear truncated-series products. A target translation is
exactly

```text
G_target(z) = exp(target*z) G(-z)
```

for the norm and every marker channel. Newton identities or Hankel/Padé
rational reconstruction decode the endpoint annihilator and the five marker
deformation polynomials.

## Exact controls

The deck-product identity and binomial target-translation law match direct
moments through order 23 over F_65537.

Radix decks of sizes 2, 3, and 4 have respectively 32, 243, and 1024 distinct
five-deck endpoints. Berlekamp-Massey returns these exact linear complexities
for the norm channel and each of the five fixed-marker channels:

```text
deck size 2: [32, 32, 32, 32, 32, 32]
deck size 3: [243, 243, 243, 243, 243, 243]
deck size 4: [1024, 1024, 1024, 1024, 1024, 1024]
```

Newton reconstruction returns the monic annihilator with every expected root.
On the size-two control, the full six-channel state recovers all 32 five-slot
source-marker tuples exactly.

The universal Newton sharpness control uses `X^32-3` and `X^32-5`. Their first
31 power sums agree, while the degree-32 power sum and norm at zero differ.
This sharpness statement is for the universal moment grammar; it is not
asserted to be a five-deck realization.

## Cost gate

With `C=B^(3/5)`, the five-deck endpoint count and exact Hankel/Prony order are

```text
C^5 = B^3.
```

Even granting quasi-linear series multiplication and rational reconstruction,
the exact state, deck update, and full-state translation have exponent 3.
They exceed the `B^(9/4)` setup/state cap and the `B^(5/4)` fresh-work cap.
Truncating below full order preserves a compact nonlinear update but loses the
exact norm/marker decoder.

This closes only the frozen exponential-moment, Newton, and Hankel/Padé
grammar. It is not a lower bound against non-moment arithmetic circuits,
source-reporting sum indices, representation-changing FFE identities, or all
nonlinear data structures.

## Admission

Eight of seventeen obligations pass. Missing obligations include a subcap
exact state and fresh source return, actual Semaev projective exceptional
charts, a signed source biconditional, known-RHS rank, factor logs, identical
descent, a generic-prime algorithm, and a Shoup improvement.

Disposition:

```text
REJECT_NONLOCAL_EXPONENTIAL_MOMENT_HANKEL_PADE_ONLY__DECK_UPDATE_AND_TRANSLATION_EXACT__NORM_AND_FIVE_MARKER_HANKEL_RANK_C5__NEWTON_AND_FULL_MARKER_SOURCE_REPLAY_EXACT_ONLY_WITH_B3_STATE__NO_PROJECTIVE_SOURCE_BICONDITIONAL__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Instantiate or refute one unequal-list subfunction-inversion index for the
`B^2` five-A endpoints against the `B^3` five-C endpoints. Derive setup,
query, reporting, and memory exponents from an explicit finite-field
5SUM-indexing construction; require `B^(9/4)` setup and `B^(5/4)` fresh
source return without moments, endpoint tables, verifier scalars, or omitted
exceptional branches.

## Literature controls

- Norton, *The Berlekamp-Massey Algorithm via Minimal Polynomials*:
  <https://arxiv.org/abs/1001.1597>.
- Berthomieu and Faugere, *Polynomial-Division-Based Algorithms for Computing
  Linear Recurrence Relations*: <https://arxiv.org/abs/2107.02582>.
- Dinur and Golovnev, *Improved Time-Space Tradeoffs for 3SUM-Indexing*:
  <https://arxiv.org/abs/2512.04258>.
