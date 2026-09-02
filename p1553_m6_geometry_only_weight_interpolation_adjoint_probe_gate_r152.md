# P1553 M6 geometry-only weight interpolation-adjoint gate R152

## Claim boundary

R152 tests whether arbitrary A/C atom tangents and adjoints can reuse the
scalar-blind R82 geometry without rebuilding setup.

It constructs exact weight-independent leaf derivative state in
`B^(3/4+o(1))` state and application cost. It does not construct the
internal weighted summation-polynomial/FFE elimination DAG or the complete
marker operator.

Classification:

```text
R82_A_C_PUBLIC_X_DECKS_ADMIT_GEOMETRY_ONLY_SUBPRODUCT_TREE__BARYCENTRIC_WEIGHT_INTERPOLANTS_ROUNDTRIP__DUAL_TANGENT_PAYLOAD_LINEAR__TRANSPOSE_ADJOINT_EXACT__ALL_DIVISIONS_PUBLIC_NONZERO_GEOMETRY_CONSTANTS__LEAF_STATE_AND_APPLY_B3O4_INSIDE_CAP__SIGNED_WEIGHT_SEPARABLE_INTERNAL_FFE_ELIMINATION_DAG_OPEN__NO_MARKER_OPERATOR_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Leaf compiler

For one public atom deck with distinct x-coordinates

```text
x_1,...,x_n,
```

freeze

```text
P(X) = product_i (X-x_i),
```

its subproduct tree, and the constants `P'(x_i)^(-1)`.

For arbitrary atom payloads `v_i`, the unique degree-below-`n`
interpolant is

```text
W_v(X)
  = sum_i v_i P(X)/((X-x_i)P'(x_i)),

W_v(x_i)=v_i.
```

This map is linear. For a tangent vector `d`,

```text
W_(v+epsilon*d) = W_v + epsilon*W_d.
```

Changing the tangent changes only the leaf payload. The subproduct tree,
Lagrange basis, and inverses remain fixed.

For a coefficient dual `g`, the transpose interpolation map returns atom
adjoints

```text
(I^T g)_i = <g,L_i>,
```

where `L_i` is the ith Lagrange basis polynomial. It satisfies

```text
<g,Iv> = <I^Tg,v>.
```

## Division safety

Every inversion is `P'(x_i)^(-1)`. It is:

- nonzero because the public x-coordinates are distinct;
- fixed by geometry before any tangent or adjoint vector is known;
- independent of atom weights and target markers.

No DLP, scalar label, root oracle, or weight-dependent pivot is used.

## Signed-point boundary

The public y-coordinate side table remains bound to every x-coordinate.
An x-only Semaev relation includes sign branches. R152 does not claim that
the interpolation map enforces the selected signed points or removes
extraneous branches. The internal circuit must carry a signed chart or an
exact branch filter.

## Actual controls

All four R82 families and both offsets are checked. Their A/C widths are:

```text
(2,3), (2,3), (3,5), (3,5),
(3,6), (3,6), (4,7), (4,7).
```

Every deck has distinct x-coordinates. All sixteen weight interpolants
roundtrip, dual tangent maps are linear, transpose pairings are exact, and
all barycentric denominators are nonzero.

The controls use only public point coordinates. They consume no verifier
scalar labels. Finite success receives no generic-family asymptotic credit.

## Cost gate

At the selected M6 scaling:

```text
|C| = B^(3/4+o(1)).
```

With fast polynomial arithmetic:

```text
geometry subproduct-tree state: B^(3/4+o(1))
one tangent interpolation:      B^(3/4+o(1))
one transpose adjoint:          B^(3/4+o(1)).
```

All are below:

```text
marker fresh cap: B^(5/4+o(1))
setup cap:        B^(9/4+o(1))
rho:              B^(5/2+o(1)).
```

The pinned multipoint reference is:

```text
Vishwas Bhargava, Sumanta Ghosh, Zeyu Guo, Mrinal Kumar,
and Chris Umans,
Fast Multivariate Multipoint Evaluation Over All Finite Fields,
arXiv:2205.00342v1.
```

R152 uses the standard univariate subproduct-tree specialization and its
transpose. It does not infer the cost of later elimination nodes.

## Admission

Fifteen of twenty-five obligations pass. Public geometry binding,
interpolation roundtrips, dual tangent linearity, transpose adjoints,
division safety, signed-y side tables, and the `B^(3/4)` leaf cost are
admitted.

No weight-separable internal elimination DAG, signed branch filter,
bidirectional marker operator, generic rank/density theorem, factor logs,
identical descent, or ECDLP algorithm is admitted.

Disposition:

```text
ADMIT_WEIGHT_INDEPENDENT_LEAF_TANGENT_AND_ADJOINT_STATE__CLOSE_LEAF_REBUILD_OBSTRUCTION_AT_B3O4__REQUIRE_SIGNED_WEIGHT_SEPARABLE_INTERNAL_FFE_DAG__NO_MARKER_OPERATOR__NO_LOGS__NO_DESCENT__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Use the frozen A/C subproduct trees and interpolation transposes as the
only weight-dependent leaves. Construct one division-safe
summation-polynomial/FFE elimination DAG whose internal topology and
pivots depend only on public geometry, whose dual payload propagates
without degree or state above `B^(9/4+o(1))`, and whose forward and reverse
marker batches cost `B^(5/4+o(1))`. It must enforce signed-point branches
and replay an exact-residual solve and shifted descent without DLP, root,
count, marginal, rank, or source oracles.
