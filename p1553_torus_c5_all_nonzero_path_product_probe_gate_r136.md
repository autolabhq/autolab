# P1553 torus C5 all-nonzero path product gate R136

## Claim boundary

R136 reduces the all-nonzero path of a decision tree of represented
polynomial zero tests to one product polynomial. Exact membership forces
a dichotomy:

- the expanded path product exceeds the available C5 Kruskal-rank
  threshold; or
- the union of the node root sets is large enough to cover essentially
  the whole prime-order subgroup complement.

On the structured factor base, R135 supplies a deterministic threshold of
six product modes. Under the frozen uniform-random-deck model only, it
supplies a threshold of `(5-o(1))*log2(q)` modes with overwhelming
probability.

R136 does not bound straight-line-program size from expanded support. It
does not cover tests of nonzero field values, coordinate comparisons,
table probes, or arbitrary circuits. It supplies no source index,
known-RHS rank, factor logs, identical descent, Shoup improvement, or
ECDLP breakthrough.

Classification:

```text
SPARSE_ZERO_TEST_TREE_ALL_NONZERO_PATH_COLLAPSES_TO_PRODUCT_POLYNOMIAL__REJECTING_LEAF_FORCES_PRODUCT_TO_COVER_POSITIVE_SUPPORT__ACCEPTING_LEAF_FORCES_NODE_ROOT_UNION_TO_COVER_SUBGROUP_COMPLEMENT__STRUCTURED_EXACT_TREE_REQUIRES_PRODUCT_ABOVE_SIX_MODES_OR_LINEAR_SCALE_ROOT_BUDGET__RANDOM_DECK_THRESHOLD_NEAR_FIVE_LOG2_Q__TWELVE_EXACT_FIVE_FACTOR_POSITIVE_INVERSE_EMPTY_REPLAYS__GROWING_SUPPORT_LOW_SLP_AND_NONZERO_VALUE_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Path-product theorem

Let a deterministic decision tree test nonzero represented polynomials

```text
f_1(z), ..., f_d(z)
```

along its all-nonzero root-to-leaf path, and define

```text
P(z) = product_i f_i(z).
```

Because the coefficient domain is a field, a point follows that path
exactly when `P(z)` is nonzero.

If the leaf rejects, every accepted point must have left the path at a
zero outcome. Therefore `P` vanishes on the full positive support.

If the leaf accepts, every point outside the positive support must leave
the path. Therefore

```text
H minus S subset union_i Z(f_i) = Z(P).
```

For an asymptotic C5 color support `S`, R135 excludes the first case when
the represented expansion of `P` has at most six distinct modes. For a
node with `t_i>=2` modes, the Kelley-adapted bound gives

```text
|Z(f_i)| <= 2*q^(1-1/(t_i-1)).
```

A one-mode nonzero monomial has no roots in `H`. Thus every exact
structured zero-test tree obeys

```text
expanded_modes(P) > 6
or
q-|S| <= sum_i 2*q^(1-1/(t_i-1)).
```

Since `|S|=q^(3/4+o(1))`, a polylogarithmic-depth path of fixed-support
nodes has root budget `o(q)`. Such a tree must escape through expanded
path-product support above six.

## Random-model extension

Under the frozen uniform-random-deck model, R135 replaces six by

```text
L(q) = (5-o(1))*log2(q)
```

with overwhelming probability. A fixed-node-support, polylogarithmic
depth tree must then have expanded all-nonzero path support above `L(q)`.
Because expanded product support is at most `product_i t_i`, nodes with at
most `t` modes require depth greater than `log_t L(q)` to escape this
fork.

That is only an `Omega(log log q)` depth condition. It remains compatible
with polylogarithmic query work and compact circuits whose represented
expansion grows rapidly. The random-model result receives no structured
factor-base credit.

## Exact controls

All eight pairing decks are replayed directly in `Fp2`. Across 30 active
finite colors, 12 contain two atoms and supply a six-point C5 progression.
For each available control:

- five linear two-mode factors are rooted at the first five progression
  targets;
- their degree-five product has exactly six represented modes;
- the product vanishes on exactly the first five progression targets;
- the sixth positive target is nonzero at every factor;
- its inverse is verified absent from the full C5 set and is also
  nonzero at every factor;
- the product-zero and any-factor-zero outcomes agree on all actual C5
  values; and
- all six sources replay and satisfy the color acceptance rule.

Thus the selected positive and exact inverse empty target follow the same
all-nonzero path. The 18 singleton finite colors have no control. These
finite witnesses receive no asymptotic credit.

## Admission

Thirteen of twenty-one obligations pass. The all-nonzero path theorem,
structured six-mode negative, model-bound near-`5*log2(q)` extension, and
12 exact path controls are admitted. Growing-support low-SLP paths,
nonzero-value and coordinate DAGs, structured seven-plus-mode products,
source indexing, rank, logs, identical descent, Pollard-rho improvement,
Shoup improvement, and breakthrough remain false.

Disposition:

```text
ADMIT_ALL_NONZERO_PATH_PRODUCT_ROOT_COVER_DICHOTOMY__REJECT_STRUCTURED_ZERO_TEST_DAGS_WITH_AT_MOST_SIX_EXPANDED_PATH_MODES__ADMIT_NEAR_FIVE_LOG2_Q_RANDOM_MODEL_EXTENSION_ONLY__ADMIT_TWELVE_EXACT_FIVE_FACTOR_POSITIVE_INVERSE_EMPTY_REPLAYS_WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_GROWING_SUPPORT_LOW_SLP_NONZERO_VALUE_AND_STRUCTURED_SEVEN_MODE_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Probe the two explicit escapes from the path-product theorem: construct
or refute a growing-support low-SLP selector whose expanded all-nonzero
product exceeds the rank threshold, and test nonzero-value
Frobenius-coordinate branches that do not reduce to zero sets. Freeze
every circuit node and branch, replay exact positive and inverse-empty
targets and C2+C3 sources, fit `B^(9/4+o(1))` state and polylogarithmic
arbitrary-target work, avoid field DLP, and charge rank, logs, identical
descent, memory, field operations, and bits.
