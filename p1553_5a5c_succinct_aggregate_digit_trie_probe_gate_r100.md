# P1553 5A/5C Succinct Aggregate Digit-Trie Gate R100

Date: 2026-07-29

Status: `SCOPED_NEGATIVE_WITH_STRUCTURED_POSITIVE_CONTROL`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can an exact aggregate digit trie support membership and one occurrence for
every `D`-message source set using less than `D` persistent state?

R100 freezes the universal family of all `D`-element subsets of `F_p`. It
does not assume that the actual R84 `3A+2C` endpoint image is universal.

## Information theorem

There are

```text
binomial(p,D)
```

possible source sets. Two different sets must have different persistent
states: querying any target in their symmetric difference requires different
exact membership answers. Source return is at least as strong.

Therefore a state encoded in `F_p` words needs at least

```text
log_p binomial(p,D)
  >= D(1-log_p D)
```

words. Under

```text
p = Theta(B^5),
D = Theta(B^(12/5)),
```

this is at least

```text
(13/25)D = Theta(B^(12/5)).
```

The setup/state cap is `B^(9/4)`, so the universal exact index is over cap.

R100 exactly counts `binomial(p,D)` states and the minimum field-word capacity
for `(p,D)=(17,4),(31,5),(61,7),(127,10)`.

## Explicit tries and occurrences

An explicit radix trie has `D` terminal records. A compressed binary Patricia
trie with `D>1` distinct leaves has `D` leaves and `D-1` branch nodes.
Duplicate values can share a value leaf, but complete occurrence return still
requires all occurrence payloads.

The duplicate control counts two copies of value `2`, retains eight occurrence
payload words, and returns one exact source.

## Structured positive control

The interval family

```text
S=[lower,lower+D)
```

has a two-word summary. Canonical integer comparison gives exact membership,
bottom, and source index. This is an explicit counterexample to applying the
universal information bound to a structured family.

R100 therefore does not claim a lower bound for the actual `3A+2C` endpoint
image. Its entropy, short generator, and leaf-free merge law remain open.

## Admission

Passed obligations: `16/31`

Lane admitted: `false`

Missing gates include an actual-image entropy or compression theorem, a
leaf-free merge law, actual 5A/5C integer and source replay, projective
infinity, proper-subsum, tangent and multiplicity branches, known-RHS rank,
factor logs, identical descent, a generic-prime family algorithm, and a
complete Shoup comparison.

## Exactly one next action

Prove or refute one actual-image entropy and merge theorem for the R84
`3A+2C` side directly from `D_A,D_C`. Freeze the public parameter family and
endpoint-key map before outcomes; either derive a leaf-free summary below
`B^(9/4)` with `B^(5/4)` exact query/source return, or prove that the reachable
image contains an `Omega(B^(12/5))`-word distinguishable subfamily. Include
multiplicity, all exceptional branches, rank, factor logs, and identical
target descent.
