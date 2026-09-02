# P1553 5A+5C black-box translated-resultant localizer gate R88

## Classification

- Owner: existing P1513/P1536/P1553/R80/R87 field-router lane.
- Evidence: exact conditional source localization, quotient Krylov rank,
  block-width cost tradeoff, and scalar multiplicity controls.
- Status:
  `ORACLE_SOURCE_LOCALIZER_LOGARITHMIC__SCALAR_BLOCK_KRYLOV_OR_HALF_GCD_OVER_CAP`.
- Cryptanalytic result: no public coefficient-free marked zero test, relation
  rank, factor logs, blind descent, Shoup-bound improvement, or ECDLP
  breakthrough.

R88 separates two obligations that were coupled in R87. Given a correct
translated-resultant zero test, a balanced subproduct tree localizes the
unique left endpoint in logarithmically many calls. Constructing that zero
test and returning the five right-source markers is the hard operation.

Materialized half-gcd is already owned by R80/P1513/R87. R88 adds a scoped
scalar/block-Krylov screen and does not rename those prior negatives.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R87 report | `f10ba663867815c9ee0b1234f4d9dee698d450a3a7171336d36f3e328ea2333a` |
| R87 gate | `16d635add67bc64d63d5870663f68ce37e35a21fa4feb436c428b7afbe6ed565` |
| R80 batched-gcd gate | `1ff3641688f4f0e13fd64f83aa540ea429a4164e0d0741b64b38acd804d7fb01` |
| P1513 direct-KU handoff | `27c8f1f15fd0c3b81ebe2008aa96db12417c3f6612c5c151212206dcba388dcc` |
| P1536 norm-jet audit | `81ec3515b584c36a809c155b5f26127bce91c09d7bfe6bccc425cdef07d51393` |

All five hashes are verified before the producer runs.

## Conditional source localizer

R88 reuses R87's frozen scalar split:

```text
32 five-A endpoint occurrences,
243 five-C endpoint occurrences,
one multiplicity-one target.
```

For a subset `S` of left endpoints, the conditional oracle reports whether

```text
product_(a in S) P_C(T-a) = 0.
```

One root test plus five balanced left-child tests localizes the unique left
endpoint:

```text
1 + ceil(log2 32) = 6 calls.
```

The complementary right endpoint and all ten deck choices replay exactly.
The positive control reads the materialized degree-243 `P_C` and a 243-word
right-source dictionary. It therefore proves only that localization around a
valid oracle is cheap; it supplies no candidate constructor.

## Scalar Krylov control

Work in the split quotient defined by the 32 distinct left roots. Multiplying
by `P_C(T-X)` is diagonal in the verifier root basis. On the accepted target:

```text
quotient dimension                 32
zero eigenvalues                    1
distinct eigenvalues               32
scalar Krylov Hankel rank          32
Berlekamp-Massey input terms       64.
```

Thus this exact instance has full scalar-Krylov linear complexity. At scale,
the quotient dimension is `m=B^2`, so even optimistically unit-cost scalar
samples require `B^2` fresh iterations, above `B^(5/4)`.

This is a control for the scalar Krylov grammar, not a lower bound for every
determinant algorithm.

## Block tradeoff

Give a standard block-Krylov method width `B^alpha`. Optimistically ignore
the cost of each quotient matvec. Then

```text
iterations       B^(2-alpha)
stored blocks    B^(2+alpha).
```

The online cap requires

```text
alpha >= 3/4,
```

while the setup-state cap requires

```text
alpha <= 1/4.
```

The feasible interval is empty. The five tested widths
`alpha=0,1/4,1/2,3/4,1` confirm that no point meets both caps.

This closes only explicit quotient blocks under that standard tradeoff.

## Standard-route reconciliation

| Route | Charged state or work | Result |
|---|---:|---|
| Materialized `P_C` | `B^3` | setup fail |
| Coefficient half-gcd | reads `B^3`, emits `B^2` remainder | setup/query fail |
| Scalar Krylov | `B^2` optimistic samples | query fail |
| Block Krylov | `B^(2-alpha)` by `B^(2+alpha)` | no cap-feasible width |
| Explicit nested C norms | `B^.6,B^1.2,B^1.8,B^2.4,B^3` | crosses setup at slot four |
| Conditional source descent | logarithmic oracle calls | constructor omitted |

R80 already proves that a standard materialized product-tree gcd can be exact
and source-returning while costing `B^3` setup/work and `B^2` workspace.
P1513 already closes standard KU, norm, primitive-element, and transposed
power-projection realizations. R88 does not broaden those scoped results.

## Multiplicity controls

An empty target is rejected by the scalar zero oracle. With two distinct
matches, the binary localizer returns one endpoint rather than rejecting the
fiber. With a repeated right root, a scalar zero decision likewise misses
nonreduced multiplicity. In both cases the norm and its first target
derivative vanish, so a marked first jet is still required.

No public marked-jet constructor, actual Semaev projective chart, signed
lift, infinity branch, or tangent/nonreduced source inverse is supplied.

## Scope and nonclaim

R88 closes:

- materialized coefficient half-gcd on the R87 split;
- scalar Krylov with full quotient linear complexity;
- explicit block-Krylov quotient state under the frozen width tradeoff;
- scalar-only binary localization as a multiplicity-complete primitive.

R88 does not close:

- a coefficient-free fixed-marker scalar resultant recurrence;
- a non-Krylov determinant/kernel identity;
- unrestricted arithmetic circuits over the compact elliptic divisors;
- rank, factor logs, identical descent, or Shoup improvement.

Six of 16 admission obligations pass. The lane is not admitted and the
breakthrough flag is false.

## Evidence

| Artifact | SHA-256 |
|---|---|
| Producer | `bd67b19a13e6e27bd4b2315aef5374d68d44d52f03a9b63bff3cefb6d5b12874` |
| Main report | `d73e017c731a54c6913aeaa94e6b5c6d54ca3757f56be14e8ca8ee5524031de1` |
| Frozen oracle interface | `a09a62f16c9b48bc5a75edefe57300dd1fa3beebee4b850912f0fbdebb24a0e7` |
| Operation ledger | `02b0a63db108e0f01294283a07f6a6a29f5388d3915502c2c9c10f11d40ef874` |
| Target/source replay | `998e341c074f7d97afe77ffceb62d62bfc65fa6153408531041a89d8fcee2ce7` |
| Multiplicity/exceptional controls | `e4ddb24e0a0febfeec97952bf5b0dc96b0049bf0bad4fa0d777c22694154def8` |
| Factor-log/descent receipt | `b53060e3ef0f023565b7f5063f7e34fe9282284a6507797bbe4f3a417cc3c99c` |
| Unit test | `5e7a9ffefda911a68525b3059981a22a6e859590990c7f48d606766537e7a74b` |

## Exactly one next action

Construct or refute one coefficient-free fixed-marker scalar resultant
recurrence for the five `C` decks modulo `P_A`. It must return zero,
multiplicity, and all five `C` source markers without `P_C` coefficients, a
`B^2` quotient vector or Krylov block, a `C` endpoint dictionary, or a
unit-cost determinant oracle; fit `B^(9/4)` setup and `B^(5/4)` fresh work
and replay every projective exceptional chart.
