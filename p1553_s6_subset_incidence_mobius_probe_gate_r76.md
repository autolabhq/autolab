# P1553 S6 subset-incidence Mobius probe R76

## Classification

- Owner: existing P1553 R2/R9-R10 finite-deck endpoint frontier; no P1554.
- Evidence: exact group-law occurrence counting on four standardized
  prime-order curves.
- Status: `REJECT_EXACT_S6_SUBSET_INCIDENCE_MOBIUS_CONSTRUCTION_ONLY`.
- Labels: `exact-finite`, `multiplicity-preserving`, `scalar-blind-source`,
  `representation-bound`, `novelty-unverified`.
- Cryptanalytic result: no admitted relation campaign, factor-log solve,
  fresh-target descent, Shoup-bound improvement, or ECDLP breakthrough.

R74 represented each three-point prefix and target-plus-pair suffix by its
at-most-four signed Kummer endpoints. It measured exact intersections through
an oracle signature grid and charged the literal grid as `B^5`. R76 replaces
that grid with an explicit exact occurrence-count formula and source
backpointers. The construction costs `B^3` prefix work/state and `B^2`
fresh-target suffix work, so it remains outside both direct caps.

This sharpens the old finite-deck R2 existence-bit boundary. It is not a new
idea ID and does not reopen the standard `2+2+1` endpoint route.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R74 S6 residual-support report | `1558482f504bd5e05b112464ee7c6734dd30ee518789735ac7abc8c305a5f740` |
| R73 resultant-multiplicity report | `00f750c15644acdaea32bbbbe9b407071cd3bf6a5cf2268ce075c55bd9a29915` |
| R31 registry containing R2 finite-deck gate | `0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f` |
| Bound R2 finite-deck gate | `55acc1457e7fd5a740da57c2c1db957374c7c18561c67b1748176dc8c61fcda5` |

## Exact identity

Let `A_i` be the signed endpoint set of prefix occurrence `i`, and let `C_j`
be the endpoint set of suffix occurrence `j`. For every nonempty endpoint
subset `S`, define occurrence histograms

```text
h_P(S) = #{i : S subset A_i},
h_Q(S) = #{j : S subset C_j}.                    (1)
```

Every set has at most four atoms, including a separate identity atom, so one
occurrence contributes at most `2^4-1=15` keys. Then

```text
N = sum_(S nonempty) (-1)^(|S|+1) h_P(S) h_Q(S)  (2)
```

is exactly the number of prefix-suffix occurrence pairs whose endpoint sets
intersect. Indeed, one pair with intersection `I` contributes

```text
sum_(empty != S subset I) (-1)^(|S|+1)
  = 1  if I is nonempty,
  = 0  otherwise.                                (3)
```

Equation (2) counts tuple occurrences, not distinct endpoint roots. Duplicate
prefix or suffix records remain separate through their histogram
multiplicities. Storing one occurrence backpointer for each singleton key
returns a concrete source whenever `N>0`.

## Multiple-root control

The frozen synthetic control uses two duplicate left sets `{1,2}` and right
sets `{1,2}` and `{2,3}`. There are exactly four intersecting occurrence
pairs. Summing only common singleton incidences gives six because pairs
sharing both endpoints are counted twice. Equation (2) returns exactly four.

Thus a product gcd degree or singleton endpoint count is insufficient as a
tuple count when two factors share multiple roots. The subset correction is
necessary and exact.

## Frozen replay

The R72 public SHA-256 decks are reused on secp256k1, P-256, P-384, and P-521
at `B=6,10,14,18`, with one blind and one forced-positive target. No scalar
label or discrete logarithm is consumed.

Across all 32 target instances:

```text
Mobius tuple count = direct deduplicated endpoint incidence count,
blind target count = 0,
forced target count = 1,
recovered forced source verifies by exact signed group law,
every dyadic parent count = sum of its two child counts.       (4)
```

At `B=18`, every possible subset contribution is distinct:

```text
prefix distinct keys = prefix contributions = 15 * 18^3 = 87,480,
suffix distinct keys = suffix contributions = 15 * 18^2 = 4,860.          (5)
```

This is finite evidence on the frozen families, not an asymptotic
distinctness theorem for arbitrary decks.

## Cost ledger

Constructing (1) for all prefix triples costs

```text
prefix work/state       B^(3+o(1)),               (6)
```

and building the target-specialized suffix histogram and looking up its keys
costs

```text
fresh-target work       B^(2+o(1)).               (7)
```

All dyadic first-coordinate children can reuse the identity, but explicitly
materializing their histograms costs `B^(3+o(1))` with a suppressed logarithmic
factor. R76 therefore improves the literal `B^5` pair grid to `B^3` setup and
`B^2` query work, while missing

```text
setup/state cap         B^(9/4+o(1)),
fresh-target cap        B^(5/4+o(1)).              (8)
```

The exact zero certificate and source replay do not repair those exponents.
No relation-density estimate, independent relation matrix, factor-log solve,
or identical fresh-target descent is supplied.

## R2 deduplication

The archived R2 gate already states that a subset-stable existence bit is
enough for logarithmic source replay, two source-labelled pair indexes fit
`B^2` setup, and standard `2+2+1` point or norm queries restore `B^3` or
`B^4` traffic. R76's increment is the explicit multiplicity-preserving count,
the multiple-common-root correction, and executable source/child controls at
the same rejected boundary.

No new idea ID is warranted. The representation-sensitive Query2P1 exception
remains the same open family.

## Scope

R76 closes only the explicit endpoint-subset histogram representation. It
does not prove a lower bound against implicit, transposed, algebraic,
randomized, or target-translated data structures. In particular, it does not
exclude a succinct prefix frequency oracle or a target update that avoids
enumerating all `B^2` suffix pairs.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_s6_subset_incidence_mobius_probe_r76.py` | `3e1887461a0355bfd026dd35bdef3aad43d9d2313a21745902ab66c463435314` |
| `p1553_s6_subset_incidence_mobius_probe_report_r76.json` | `de41d1618bc71c46f700bfead0ed72c5ac0b29a89da3b5c32534a15314ef4c93` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_s6_subset_incidence_mobius_probe_r76.py` | `a3f69763285d1d04205b16271d347989dac518f47a53b91ee7f8eddad38aaf2f` |

Targeted replay:

```text
Ran 3 tests in 0.349s
OK
families=4
counts_exact=True
sources_verified=True
lane_admitted=False
```

## Disposition

```text
REJECT_EXACT_S6_SUBSET_INCIDENCE_MOBIUS_CONSTRUCTION_ONLY__FOUR_STANDARD_CURVES__B6_10_14_18__EXACT_TUPLE_COUNTS__MULTIPLE_COMMON_ROOTS_CORRECTED__BLIND_ZERO__FORCED_SOURCE__DYADIC_CHILDREN__PREFIX_STATE_B3__FRESH_TARGET_B2__R2_BOUNDARY_SHARPENED_NOT_NEW_IDEA__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct a target-translated subset-frequency oracle with prefix advice at
most `B^(9/4+o(1))` and fresh-target query work/workspace at most
`B^(5/4+o(1))`, without enumerating either the `B^3` prefix triples or `B^2`
target suffixes. Bind the existing Query2P1 gates and require exact counts, a
source, blind zero, and dyadic children.
