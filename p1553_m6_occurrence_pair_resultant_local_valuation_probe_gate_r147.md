# P1553 M6 occurrence-pair resultant local-valuation gate R147

## Claim boundary

R147 gives an exact algebraic representation of the ordered six-factor
fiber count as a local valuation. It retains every ordered `C3` occurrence
as a repeated root, pairs two such occurrences through the Cayley group law,
and proves that the local valuation of the resulting implicit pair resultant
is exactly the ordered `C6` count.

The full pair resultant is never materialized. Applying the
Moroz-Schost truncated-resultant bound independently to each target or to
the standard componentwise direct product still exceeds the R115 caps.
This charges a standard route; it is not a lower bound for shared
multi-target circuits or data structures.

It claims no shared valuation index, atom marginals, factor logs, identical
descent, Pollard-rho improvement, Shoup improvement, or breakthrough.

Classification:

```text
ORDERED_C6_COUNT_IS_LOCAL_VALUATION_OF_OCCURRENCE_PAIR_RESULTANT__REPEATED_C3_DIVISOR_DEGREE_B9O4__FULL_PAIR_RESULTANT_NEVER_MATERIALIZED__MOROZ_SCHOST_LOCAL_TRUNCATION_SOFT_O_DK__ONE_LOCAL_QUERY_B9O4__A6_BATCH_B11O4__FULL_RELATION_STREAM_B7O2__SHARED_TRANSPOSED_VALUATION_MARKER_OPERATOR_OPEN__NO_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Exact identity

Let

```text
P_occ(X) = product over ordered C3 occurrences
           (X - cayley_parameter(sum occurrence)).
```

Repeated endpoints remain repeated roots, so

```text
deg(P_occ) = |C|^3 = B^(9/4+o(1)).
```

Pair two occurrence roots through the Cayley composition `oplus`. Up to
nonzero chart units, the implicit pair resultant is

```text
R(T) = product over ordered C3 occurrence pairs
       (T - (x oplus y)).
```

Its formal degree is

```text
|C|^6 = B^(9/2+o(1)),
```

but it is not materialized. At a target parameter `tau`, every local Cayley
factor is simple in `T`, hence

```text
ord_(T=tau) R(T)
  = number of ordered C3 occurrence pairs summing to tau
  = ordered C6 fiber count.
```

No root extraction or discrete logarithm is required by this identity.

## Exact controls

The verifier independently constructs the repeated occurrence divisor for
all four R82 curve families and both offsets. The occurrence degrees are

```text
27, 27, 125, 125, 216, 216, 343, 343,
```

while the squarefree support degrees are

```text
10, 10, 35, 35, 56, 56, 84, 84.
```

Every repeated-root multiplicity is exact. On each control, twelve sampled
positive targets have local valuation equal to the direct ordered `C6`
count, and six sampled empty targets have valuation zero. Finite
enumeration consumes no candidate root or DLP oracle and receives no
asymptotic credit.

## Truncated-resultant charge

Moroz and Schost compute a degree-`d` bivariate resultant truncated to order
`k` in soft-`O(d k)` base-field operations, including singular expansion
points, in characteristic zero or characteristic at least `k`.

Primary source:

```text
https://arxiv.org/abs/1609.04259
references/moroz_schost_truncated_resultant_1609.04259.pdf
sha256 160c68cfbb413ca27352a064cbf2d27f7ad4ed6a210c3d6ead2770e00204b709
```

For bounded local valuation order, applying that algorithm to the occurrence
divisor costs

```text
one local target:                 B^(9/4+o(1))
one known-target A6 batch:        B^(11/4+o(1))
complete B^(5/4) relation stream: B^(7/2+o(1))
                                  = N^(7/10+o(1)).
```

The occurrence divisor fits the `B^(9/4)` setup cap, but one independent
query already exceeds the `B^(5/4)` fresh-work cap. The standard A6 batch
and complete relation stream both exceed Pollard rho.

These are upper-bound charges for componentwise or direct-product use of
the cited algorithm. They do not prove a lower bound for a shared
transposed multi-target valuation circuit, modular-composition structure,
RAM, or cell-probe model.

## Admission

Fifteen of twenty-five obligations pass. The repeated occurrence divisor,
local pair-resultant count identity, source-bound replay, eight finite
controls, and standard truncated-resultant charges are admitted.

No inside-cap shared multi-target valuation index, offline/online atom
marginal transpose, generic integer lift, structured rank theorem,
factor-log solve, identical target descent, or generic-prime algorithm is
admitted.

Disposition:

```text
ADMIT_EXACT_LOCAL_VALUATION_COUNT_IDENTITY__ADMIT_STANDARD_COMPONENTWISE_TRUNCATED_RESULTANT_NEGATIVE__NO_SHARED_MULTI_TARGET_INDEX__NO_MARGINALS__NO_LOGS__NO_DESCENT__NO_GENERIC_TRANSFER__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Do not apply local truncated resultants independently. Construct one
genuinely shared transposed multi-target valuation-and-marker operator for
the occurrence divisor. It must process the complete `B^(5/4)` target
stream after `B^(9/4)` setup without a `B^(9/4)` factor per target, emit
exact integer counts and `B^(3/4)` `A/C` marginals, and replay R144 rank,
factor logs, and shifted descent. It may use no DLP, root, resultant,
valuation, count, marginal, rank, or source oracle.
