# Independent P1553 projector-trace router R9 red-team transcript

Reviewer: `019f7ca1-e1be-7941-97be-290005295446`
Record: coordinator transcription of the independent theorem-only response
Evidence: read-only review; no run

## Terminal verdict

```text
REVISE_SCOPED_THEOREM__WITHIN_COLOR_X_CLASS_UNIQUENESS_REQUIRED__RESTRICTED_ROOT_COUNT_AT_MOST_16_B4_AND_FROZEN_32_B4_BOUND_CONSERVATIVE__EXACT_TRACE_COUNT_ONLY_WHEN_BOUND_LT_P__ONE_PLUS_FIVE_LOG_B_COUNT_CALLS_RETURN_ONE_SOURCE__COUNT_IMPLIES_ROUTER_NOT_CONVERSE__TARGET_FRESH_CP_FACTORS_REQUIRE_A_LE_B1_OVER_4__TARGET_INDEPENDENT_FACTORS_CONDITIONALLY_ALLOW_A_LE_B5_OVER_4__PROJECTOR_RANK_AT_MOST_ROOT_COUNT__SPARSE_FIBERS_DEFEAT_LARGE_RANK_OBSTRUCTION__FIRST_CARRY_AFFINE_ONLY_FOR_K1_AND_LIFT_DEPENDENT__LITERAL_FERMAT_EXPANSION_OVER_CAP_FOR_R_R_GE_2_WITHOUT_RANK_LOWER_BOUND__STANDARD_BALANCED_TRIPLE_TABLE_OR_SEARCH_B3__EXACT_IMPLICIT_COUNT_CONTRACTION_UNSUPPLIED__NO_NEW_IDEA__NO_RUN__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Accepted root-count theorem

Assume every colored deck contains each global x-class at most once. Repeated
global x-classes in different colors remain separate occurrences. After target
sign normalization, fixing four labels leaves `2^4` sign choices. Every choice
determines at most one legal fifth x-class, so

```text
z_(R,I) <= 16 min_j product_(i!=j) |I_i|.
```

The producer's frozen `32 B^4` bound is conservative. If one color contains
multiple occurrences of the same global x-class, the maximum within-color
multiplicity must multiply the bound and the finite-size threshold must be
rechecked.

In the split quotient, multiplication by

```text
chi_(R,I)=1-F_R^(p-1)
```

is diagonal with one on each occurrence-labelled x-root. Its algebra trace is
therefore `z_(R,I) mod p`. Under the explicit threshold `32 B^4<p`, the
canonical field representative is the exact integer count.

The trace counts x-tuples, not sign patterns. The R6 biconditional guarantees
at least one valid sign pattern for every counted tuple; constant sign
enumeration and complete projective verification recover it at the leaf.

## Accepted branch replay

Query the full count once. At each dyadic split, query one child. If its count
is positive retain it; otherwise retain the complementary child, whose count
is positive by disjoint additivity. This requires at most

```text
1 + sum_i ceil(log_2 |I_i|) <= 1 + 5 ceil(log_2 B)
```

exact count calls. The leaf retains occurrence backpointers; repeated global
columns are aggregated only after sign verification.

Exact counting therefore implies empty-or-one-source Query2P1 with
`B^(o(1))` replay overhead. The converse is not established because a router
that returns one source need not recover the full multiplicity.

## Accepted representation bounds

In CRT delta coordinates,

```text
rank_CP(chi_(R,I)) <= z_(R,I),
rank_CP(F_R^(p-1)) <= z_(R,I)+1.
```

These are output-rank upper bounds. Empty and singleton fibers have projector
rank zero and one, so a universal large-rank projector obstruction is false on
the useful sparse fibers. A source-faithful delta decomposition already
contains the sources and is not a source-free constructor.

If an exact `A`-term target-fresh CP projector is supplied, factor construction
and interval sums cost `A B^(1+o(1))`, requiring `A<=B^(1/4+o(1))` under the
online cap. If all factor vectors and interval sums are target-independent and
only the `A` weights specialize, setup/state costs `A B^(1+o(1))` and target
replay costs `A B^(o(1))`; both direct caps conditionally allow
`A<=B^(5/4+o(1))`.

For

```text
K_1=(tilde(F)-res_p(F))/p,
```

changing the lift to `tilde(F)+pL` gives `K'_1=K_1+L`. This affine formula is
asserted only for the first carry. Higher powers remain lift- and
schedule-specific. No unrestricted finite-field circuit or elliptic
projector lower bound follows.

## Accepted constructor boundary

Let `r_R` be the number of nonzero separated terms after target
specialization. Literal expansion of `F_R^(p-1)` has

```text
binomial(p+r_R-2,r_R-1)
```

composition-indexed terms with nonzero multinomial scalars modulo `p`. For
fixed `r_R>=2`, this is `p^(r_R-1+o(1))`. Products may coincide or cancel, so
this rejects only literal uncompressed expansion. The `r_R=1` case and every
recompressed circuit remain outside that conclusion.

The standard balanced route stores the `B^2` occurrence-labelled pair
dictionary, then materializes or searches `B^3` target-fresh triple
evaluations for the nonlinear zero match. This gives `B^4` relation
collection and `B^3` blind descent. The separated triple formula itself has
small factor storage, so this is a named standard-route cost, not a lower
bound on every implicit orthogonality or trace contraction.

## Exactly one next action

Derive or refute one exact balanced `2|3` contraction computing only
`C_(R,I)` on the full box and every queried dyadic child, without materializing
or source-supplying the `B^3` triple table, within `B^(9/4+o(1))` setup/state
and `B^(5/4+o(1))` fresh-target time/workspace. Charge the initial count, one
child per level including zero children, target specialization, exact field
and bit complexity, occurrence backpointers, leaf sign verification,
repeated-column aggregation, relation density and rank, factor logs, and
identical blind-masked descent. A negative result closes only the explicit
contraction representation it proves.
