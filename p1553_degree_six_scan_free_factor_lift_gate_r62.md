# P1553 degree-six scan-free factor-lift gate R62

## Classification

- Owner: existing P1553/IDEA-195 primitive product-pencil frontier; no new
  idea ID.
- Evidence: exact affine preimage solve, rank-one Groebner lift, cubic
  intersection factorization, and R60 witness replay; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_POSITIVE_SCAN_FREE_FACTOR_LIFT_TOY_GATE`.
- Labels: `toy`, `exact-scan-free-factor-lift`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: conditional on completed module setup, the R61
  fixed-class tensor, and the R60 pencil, R62 recovers the `t=29` parameter,
  both factor sections, and all six source curve points without an online
  `F_p`, `E(F_p)`, or subgroup enumeration.
  It does not charge source-pencil generation, restricted factor-base density,
  projected scalar logs, relation rank, linear algebra, or target descent. It
  is not a Shoup-bound improvement or ECDLP breakthrough.

## Inputs

| Input | SHA-256 |
|---|---|
| R61 hypersurface-locator gate | `a737e50f7b587ccae70edf988ca9687f4706778db549b3ca58432463017fa688` |
| R61 exact locator report | `f91213021bd350b77cf1ac7d5cb8f2f9ae63b1265a5081d4ed58a3da07aa1f64` |
| R60 deterministic witness report | `b15ac1b6cb5df18b4fb5fdcad4d0982fb8c122777dfc6009cbd5318fc28dd364` |

## Rank-one lift

For each rational product-image point on the R60 pencil, R62 solves

```text
mu(z)=w,  z in F_193^9.                            (1)
```

The rank-six map leaves an affine three-space `z=z_0+s_0k_0+s_1k_1+s_2k_2`.
The nine `2 x 2` minors of the reshaped `3 x 3` tensor impose rank one. For
the finite roots `0,29,33` and the product point at infinity, the lexicographic
Groebner bases are three independent linear equations, hence each fiber has a
unique rational rank-one point.

At `t=29`, the recovered translated factor lines are exactly

```text
(1,81,24), (1,136,11),                            (2)
```

matching R60. Their curve-intersection cubics factor as three linear factors
each. Undoing the fixed translations recovers exactly R60's two source
triples with curve-point indices

```text
(47,103,107), (119,134,150).                      (3)
```

The comparison in (3) uses the coordinates pinned in the R60 report; R62 does
not rebuild or scan the 206-point curve catalog.

## Root classification without scans

The marked roots `t=0` and `t=infinity` also lift to two split cubics. The
extra rational hypersurface root `t=33` lifts uniquely to lines

```text
(1,103,43), (1,103,117),                          (4)
```

but both intersection cubics are irreducible over `F_193`, so it yields no
rational source point. This reproduces R61's verifier classification by
bounded-degree factorization rather than curve-point enumeration.

After setup, the online conditional recovery consists of one constant-size rank-one
system and two cubic factorizations. The latter use expected `O(log p)` field
operations at fixed degree, with field-operation bit costs separate. Public
`[104]` projection then maps each recovered rational point into the
order-103 subgroup using scalar multiplication; it does not reveal the
projected scalar logarithm.

The executable setup is not scan-free: importing the legacy R44 dependency
constructs a forward scalar-labelled table of all `N` subgroup embeddings,
and R62 reconstructs its three factor-basis lines from pinned labelled blocks.
This is `Theta(N)` setup, can be inverted by a linear lookup, and is neither
charged nor amortized here. No inverse DLP is invoked by the conditional lift
itself. Pinning the basis lines directly and removing the import-time table is
still required before making a broad end-to-end no-enumeration claim.

## Cost boundary

R62 does not pay for how the marked source products were selected. R60 chose
them from scalar-labelled subgroup triples, so treating those known labels as
free would circularly assume the logarithms that relation collection is meant
to recover. A usable algorithm still needs a public point factor base,
scan-free or charged source sampling, and a measured probability that all
recovered atoms land in that base.

No relation matrix, independent-rank receipt, factor-base logarithms, fresh
target action, or scalar-blind descent is present.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_POSITIVE_SCAN_FREE_FACTOR_LIFT_TOY_GATE
FOUR_RATIONAL_PRODUCT_IMAGE_POINTS_HAVE_UNIQUE_RANK_ONE_LIFTS
T29_LINES_AND_SIX_SOURCE_POINTS_RECOVERED_EXACTLY
T33_TWO_IRREDUCIBLE_CUBICS_CLASSIFIED_WITHOUT_CURVE_SCAN
NO_FIELD_CURVE_OR_SUBGROUP_ENUMERATION_AFTER_UNCHARGED_THETA_N_SETUP
SOURCE_PENCIL_GENERATION_AND_FACTOR_BASE_DENSITY_UNCHARGED
PROJECTED_SCALAR_LOGS_RANK_LINEAR_ALGEBRA_DESCENT_ABSENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: replace the scalar-labelled source triples with a
public point factor base of size `B`, run the full locator-and-lift pipeline,
and charge the empirical and combinatorial probability that all six recovered
atoms lie in the base.
