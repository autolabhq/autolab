# Independent P1553 NET red-team transcript

Task: `TASK-20260719-P1553-NET-RT-R1`
Reviewer agent: `019f7c5d-37bc-7190-8b41-46b328379798`
Record: coordinator transcription of the independent final response
Evidence: theorem-only, read-only, no run

## Terminal verdict

```text
REVISE_SCOPED_NEGATIVE__RAW_INDEX_CORRECTED__EXACT_ZERO_SUPPORT_ONLY_ON_NONDEGENERATE_CHART__GAUGE_SUPPORT_INVARIANT_RAW_UNITS_NOT_INVARIANT__STANDARD_MIXED_SEEDS_B4_OR_FIXED_LABEL_B2__DIAGONAL_PRODUCT_NONCLOSURE_SCOPED_TO_STANDARD_COMPONENTWISE_RECURRENCES__NO_RELATION_ALL_UNIT__RANK_LOGS_DESCENT_UNSUPPLIED__P1513_P1540_P1551_P1516_MERGE__NO_GENERAL_LOWER_BOUND__NO_SHOUP_CLAIM__NO_P1554__NO_RUN__NO_BREAKTHROUGH
```

## Reconstructed corrections

The raw producer interface `Psi_(j,1)(T,q)` fails at `j=0,1` because
Stange normalizes `Psi_(0,1)` and `Psi_(1,1)` to one. Put

```text
q_(u,v)=A_0+u+v-R,
Q_(u,v)=q_(u,v)-2T.
```

Then

```text
Psi_(j+2,1)(T,Q_(u,v))=0
iff (j+2)T+Q_(u,v)=O
iff u+v+A_j=R
```

on the nondegenerate chart. Identity, `Q=+/-T`, tangent, vertical,
infinity, repeated, and nonreduced cases require complete projective branches
and saturated masks.

Under independent nonzero quadratic net gauges, individual zeros, aggregate
zero support, fifth-label gcd support, and interval zero/nonzero decisions are
invariant. Raw nonzero products and raw interval unit values are not. Any
value-sensitive algorithm must construct and charge its normalization.

The raw product of component net terms is not proved equal to R4's
key-difference resultant. The strongest charitable replacement is only the
same-zero-support scalar `r_R(t_j)`.

## Recurrence gate

For each component, write the standard three recurrence summands as

```text
A_q+B_q+C_q=0.
```

These equations do not imply

```text
product_q A_q + product_q B_q + product_q C_q = 0.
```

Multiplication introduces all mixed choices. This rejects only the standard
componentwise diagonal-product closure. It does not prove generic monomial
independence and does not rule out an unknown aggregate arithmetic or Boolean
circuit.

## Cost gate

```text
pair-index setup                       B^2,
individual mixed pair-pair seeds       B^4,
one fixed-label resultant surrogate    B^2,
all B fifth labels                     B^3,
optimistic supplied recurrence         B^(5/2),
required total online cap              B^(5/4).
```

The one-label surrogate already exceeds the cap and returns neither all fifth
labels nor pair-source witnesses. On the no-relation control all components
and intervals are units, so no dynamic split or early zero removes negative
replay.

## Conditional path

The earlier `lambda=0.45`, `mu=0.40` ledger remains conditional on the
missing constructor, relation and target density, `Theta(B)` independent
rank, factor-log completion, identical scalar-blind descent, collision and
failure accounting, and bit costs.

P1540 owns direct net, QRT, EDS, Fourier, and index-location controls. P1513
owns translated products and common-factor/source replay; P1551 owns
represented endpoint access; P1516 owns the pair indexes and missing router.
No new owner or P1554 is justified.

## Provenance objection preserved

The reviewer called historical receipt hashes stale after comparing them with
the later shared-file state. That objection is preserved for coordinator
adjudication; this transcript does not silently promote it.

## Exactly one next action

Under existing P1553/P1513/P1551/P1516 ownership, require one theorem-only
oracle-free construction of `r_R mod g_I` or exact dyadic zero/unit
decisions within total `B^(5/4+o(1))), including complete charts, target
freshness, positive and negative replay, and exact source recovery; otherwise
preserve the scoped exception unchanged.
