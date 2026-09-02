# P1553 5A5C Unequal-List Subfunction Inversion Gate R91

## Claim boundary

No generic-prime-field ECDLP algorithm, Pollard-rho improvement, Shoup-bound
improvement, factor-log solve, or target descent is claimed.

Classification:

```text
UNEQUAL_LIST_SUBFUNCTION_THEOREM_EXACT__BEST_ONLINE_COMPATIBLE_SETUP_B4P4__INTEGER_RESIDUE_MAP_NO_GENERIC_GROUP_TRANSFER
```

## Bound theorem

The locally bound source is Dinur and Golovnev,
*Improved Time-Space Tradeoffs for 3SUM-Indexing*,
`arXiv:2512.04258v2`, dated 2026-04-23:

```text
e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c
```

For integer lists of lengths `n<=m`, Theorem 5.1 gives

```text
S = soft-O(n^(3/2-delta) m)
T = soft-O(n^delta)
0 <= delta <= 1.
```

The output is one exact source pair or bottom. The auxiliary data includes
the explicit sorted input lists and a copy of the large list sorted by its
residue modulo the sampled prime.

## Intended 5A+5C substitution

For the R82 split,

```text
n = B^2
m = B^3
S = B^(6-2*delta)
T = B^(2*delta).
```

The setup cap would require

```text
delta >= 15/8,
```

outside the theorem's range. The online cap requires

```text
delta <= 5/8.
```

At the best online-compatible point `delta=5/8`, the exact exponents are

```text
S = B^(19/4) = B^4.75
T = B^(5/4).
```

The explicit large-list auxiliary alone has `B^3` words and already exceeds
the `B^(9/4)` setup/state cap.

## All deck partitions

R91 enumerates every nonempty bipartition of five `B^(2/5)` A decks and five
`B^(3/5)` C decks, identifying complementary partitions. There are eleven
distinct small-side exponents.

The best theorem point that respects the online cap uses a small side of
exponent `6/5`, `delta=1`, and a large side of exponent `19/5`:

```text
S = B^(22/5) = B^4.4
T = B^(6/5).
```

No partition meets the setup cap, and no partition's explicit large-list
auxiliary fits the setup cap.

The balanced kSUM theorem applied optimistically to a factor-base list of
size `B` and a five-factor query has `k=6`. Its best paper-range point is

```text
S = B^(9/2)
T = B.
```

It also fails setup even when scalar labels are granted.

## Source and transfer controls

A finite explicit-integer control verifies the paper's `f_d`, residue maps,
and exact source translator: every present target returns one verified source,
and an absent target is rejected after exact verification. Direct scanning is
used only as a semantic oracle and receives no runtime credit.

The transfer to a generic prime-order elliptic group is absent. The paper's
maps `y mod q` and `y mod p` act on additive integer labels. Any homomorphism
from a prime-order group to a smaller residue group is trivial, while a public
point encoding is not addition-compatible. DLP labels would make the maps
compatible but are unavailable to the candidate.

## Scope

This closes direct applications of the bound unequal-list and balanced
kSUM-indexing theorems to explicit 5A+5C endpoint lists. It does not lower-bound
a compact elliptic subfunction decomposition, a non-Fiat-Naor data structure,
or a representation-changing summation-polynomial/FFE identity.

Nine of twenty obligations pass. Rank, factor logs, identical descent, generic
prime coverage, exceptional projective source replay, and a Shoup improvement
remain absent.

Disposition:

```text
REJECT_BOUND_UNEQUAL_LIST_AND_BALANCED_KSUM_INDEX_THEOREMS_ONLY__INTENDED_SPLIT_B4P75_SETUP_UNDER_ONLINE_CAP__BEST_TEN_DECK_SPLIT_B4P4__EXPLICIT_LARGE_LIST_OVER_CAP__INTEGER_SOURCE_REPORTING_CONTROL_EXACT__NO_ADDITIVE_RESIDUE_MAP_ON_GENERIC_PRIME_GROUP_ENCODINGS__COMPACT_ELLIPTIC_SUBFUNCTION_MAP_OPEN__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one compact elliptic analogue of the subfunction theorem.
Public `MAP1`, `MAP2`, `f_d`, and `TR` must act directly on compact `D_A,D_C`
and one target, reduce inversion to one subfunction, fit `B^(9/4)` setup and
`B^(5/4)` fresh source return, and use no DLP labels, proper quotient, explicit
endpoint list, verifier oracle, or omitted projective branch.

## Primary reference

- Dinur and Golovnev, *Improved Time-Space Tradeoffs for 3SUM-Indexing*:
  <https://arxiv.org/abs/2512.04258v2>.
