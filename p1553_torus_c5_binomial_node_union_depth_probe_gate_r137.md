# P1553 torus C5 binomial-node union-depth gate R137

## Claim boundary

R137 combines the R136 all-nonzero path fork with a deterministic
fivefold product-set floor. It rules out every polylogarithmic-query
decision tree whose nodes are represented binomial zero tests on the
structured C5 color supports, regardless of the expanded support of the
path product.

Under the frozen uniform-random support model only, the same nodewise
root-union argument gives polynomial depth lower bounds for nodes with up
to four represented modes. That extension does not transfer to the
structured factor base.

R137 does not close structured nodes with three or more modes,
random-support nodes with five or more modes, nonzero-value tests,
coordinate comparisons, or arbitrary circuits. It supplies no source
index, known-RHS rank, factor logs, identical descent, Shoup improvement,
or ECDLP breakthrough.

Classification:

```text
CAUCHY_DAVENPORT_FORCES_EVERY_STRUCTURED_COLOR_FIVEFOLD_PRODUCT_SET_TO_AT_LEAST_FIVE_M_MINUS_FOUR__PRIME_ORDER_BINOMIAL_HAS_AT_MOST_ONE_ROOT__EXACT_BINOMIAL_ZERO_TEST_TREE_NEEDS_B_THREE_QUARTERS_ALL_NONZERO_PATH_DEPTH_REGARDLESS_OF_PRODUCT_EXPANSION__UNIFORM_RANDOM_SUPPORT_EXTENDS_POLYNOMIAL_DEPTH_TO_FOUR_MODE_NODES_ONLY__THIRTY_EXACT_PRODUCT_SET_AND_POSITIVE_INVERSE_EMPTY_CONTROLS__STRUCTURED_THREE_PLUS_FIVE_PLUS_LOW_SLP_NONZERO_VALUE_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Product-set floor

The pinned source is Matt DeVos, *The Structure of Critical Product
Sets*, arXiv:1301.0096. Its introduction states Cauchy-Davenport in
multiplicative notation for a prime-order group:

```text
|XY| >= min(q, |X|+|Y|-1).
```

Iteration gives, for every nonempty atom set `A` in the prime-order
pairing image,

```text
|A^5| >= min(q, 5|A|-4).
```

No discrete logarithm is computed. An abstract isomorphism between the
prime cyclic group and `Z/qZ` transfers the theorem.

A balanced structured color contains

```text
m = B^(3/4+o(1)) = q^(3/20+o(1))
```

distinct atoms. Its accepted C5 support contains `A^5` and therefore at
least `5m-4` targets. The complete C5 tuple count is at most
`B^(15/4)=q^(3/4)`, so the complement still has `q-o(q)` points.

## Binomial depth

A nonzero represented binomial

```text
a*z^u + b*z^v
```

with distinct exponents modulo prime `q` has at most one root in the
pairing image. After division, its root equation is `z^(u-v)=c`; the
nonzero power map is a permutation of the prime-order group.

R136 shows that the union of root sets along the all-nonzero path must
cover either the complete positive support or its complete complement.
For a depth-`d` binomial tree that union has at most `d` points. Exactness
therefore requires

```text
d >= min(|S|, q-|S|) >= 5m-4 = B^(3/4+o(1)).
```

This exceeds the polylogarithmic arbitrary-target query cap and is
independent of path-product expansion.

## Random-support extension

Under the frozen uniform-random support model, a color support has
`q^(3/4+o(1))` targets. A `t`-mode node has at most

```text
2*q^(1-1/(t-1))
```

roots. The all-nonzero path depth is therefore at least

```text
q^(3/4-(1-1/(t-1))+o(1))/2.
```

The exact depth exponents are:

| Maximum node modes | Exponent in `q` | Exponent in `B` |
| --- | --- | --- |
| 2 | `3/4` | `15/4` |
| 3 | `1/4` | `5/4` |
| 4 | `1/12` | `5/12` |

All are polynomial and miss the query cap. At five modes the exponent
gap closes, so the theorem preserves that route.

## Exact controls

All eight pairing decks and 30 active colors are replayed directly in
`Fp2`.

- Every singleton color has one fivefold product, meeting `5m-4=1`.
- Every two-atom color has six fivefold products, meeting `5m-4=6`.
- Every product-set bound is exact.
- A selected product is present in global C5.
- Its inverse is absent from global C5.
- Linear factors at all other color products leave the selected positive
  and inverse empty on the same all-nonzero path.

These small controls verify semantics only and receive no probability or
asymptotic credit.

## Admission

Fifteen of twenty-three obligations pass. The product-set floor,
deterministic `B^(3/4)` binomial depth lower bound, model-bound
one-through-four-mode depth lower bounds, and 30 exact controls are
admitted. Structured three-plus-mode trees, five-plus-mode low-SLP trees,
nonzero-value Frobenius DAGs, source indexing, rank, logs, identical
descent, Pollard-rho improvement, Shoup improvement, and breakthrough
remain false.

Disposition:

```text
ADMIT_CAUCHY_DAVENPORT_STRUCTURED_PRODUCT_SET_FLOOR__REJECT_ALL_STRUCTURED_BINOMIAL_ZERO_TEST_TREES_AT_POLYLOG_QUERY__ADMIT_RANDOM_SUPPORT_POLYNOMIAL_DEPTH_LOWER_BOUND_THROUGH_FOUR_MODES_ONLY__ADMIT_THIRTY_EXACT_PRODUCT_SET_AND_POSITIVE_INVERSE_EMPTY_CONTROLS_WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_STRUCTURED_THREE_PLUS_FIVE_PLUS_LOW_SLP_AND_NONZERO_VALUE_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Probe structured three-plus-mode zero tests, five-plus-mode low-SLP
zero-test circuits under the random support model, and nonzero-value
Frobenius-coordinate branches. Freeze every circuit node and branch,
replay exact positive and inverse-empty targets and C2+C3 sources, fit
`B^(9/4+o(1))` state and polylogarithmic arbitrary-target work, avoid
field DLP, and charge rank, logs, identical descent, memory, field
operations, and bits.
