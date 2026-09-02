# P1553 5A/5C Multi-Edge Digitized Equality Projector Gate R99

Date: 2026-07-29

Status: `SCOPED_NEGATIVE_WITH_POSITIVE_REPRESENTATION_CONTROL`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can several small channels evade R98's width-`p` one-bond equality kernel and
also provide a cap-sized aggregate source index?

R99 freezes canonical binary digits of the integer representative in `[0,p)`.
For `k=ceil(log2 p)`, each field value has a digit vector

```text
d(x) in {0,1}^k.
```

Digitwise equality is

```text
EQ(x,y) = product_i (1-(d_i(x)-d_i(y))^2).
```

## Positive representation control

For `p=5,7,11,13,17`, the digit map is injective and `EQ(x,y)` is exactly
`1[x=y]` for every pair. Thus supplied radix digits give a genuine
`Theta(log p)`-edge equality representation.

This does not contradict R98. The product of edge alphabets is
`2^k in [p,2p)`, so the flattened cut still has capacity at least `p`.

## Constructor controls

For every nonempty binary digit fiber `S subset F_p`, the unique polynomial
indicator `1_S` has degree `p-1`. Its leading coefficient is `-|S|`, which is
nonzero because `0<|S|<p`. R99 verifies every interpolated fiber exactly on
the five primes.

The frozen standard constructors cost:

```text
full digit table                 Theta(p log p),
all fiber root lists             Theta(p log p),
all fiber coefficient tables     Theta(p log p),
sourcewise digit traffic          Theta(D log p).
```

Under `p=Theta(B^5)` and `D=Theta(B^(12/5))`, these are respectively `B^5`
and `B^(12/5)` up to logarithmic factors. Both miss the direct caps.

Degree `p-1` is not an arithmetic-circuit lower bound. R99 does not refute a
succinct digit/fiber circuit or an aggregate digit trie built directly from
the compact A/C divisor circuits.

## Occurrence controls

The supplied-digit occurrence replay counts two copies of value `2`, returns
one exact dyadic occurrence, and returns bottom for blind target `3`. These
controls credit the representation only; they materialize every occurrence
digit vector.

## Admission

Passed obligations: `15/31`

Lane admitted: `false`

Missing gates include a succinct aggregate digit index, a field-operation
digit extractor, actual 5A/5C integer and source replay, projective infinity,
proper-subsum, tangent and multiplicity branches, known-RHS rank, factor logs,
identical descent, a generic-prime family algorithm, and a complete Shoup
comparison.

## Exactly one next action

Construct or refute one succinct aggregate digit trie from the compact A/C
divisor circuits. Freeze the arithmetic digit or fiber-indicator circuit and
aggregation law before outcomes; require setup/state below `B^(9/4)`, fresh
count and complete source return below `B^(5/4)`, no sourcewise `D` traffic or
`p`-size advice, exact multiplicity and all exceptional branches, known-RHS
rank, factor logs, and identical target descent.
