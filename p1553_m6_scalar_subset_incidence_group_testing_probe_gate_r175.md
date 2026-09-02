# P1553 M6 scalar subset-incidence group-testing gate R175

Date: 2026-08-01

## Scope

R175 converts the exact tangent-aware R174 signed locator into a scalar subset
query and proves an output-sensitive balanced-tree recovery reduction. The six
finite controls recover all 140 R174 roots exactly.

The reduction exposes a sharply specified below-rho oracle interface, but does
not construct that oracle. Direct expansion remains above rho. This is not an
ECDLP algorithm, a generic locator, or a Pollard-rho or Shoup improvement.

## Scalar Incidence

Let `A(P)` be the R174 signed aggregate at selected leaf `P`. For every
nonempty selected subset `S`, define

```text
Sigma(S) = product_{P in S} A(P) in F_p.
```

Because `F_p` is an integral domain,

```text
Sigma(S) = 0  if and only if  S contains an R174 candidate.
```

This includes the geometric-tangent `Q=P` chart already established in R174;
it does not replace that chart with the derivative of the interpolated `V`.

## Balanced Recovery

Build a balanced binary tree over the selected leaves. Query the root, stop at
every nonzero node, and query both children of every zero internal node. The
zero leaves are exactly the `K` candidates.

For height `h=ceil(log2(n))`, each candidate has at most one zero ancestor per
level, so the number of queries is at most

```text
1 + 2 K h.
```

At any depth, the queried child subsets are disjoint. Their total size is at
most `n`, hence total queried subset volume is at most

```text
n (1 + h) = softly O(n).
```

The six controls make 358 queries against a bound of 1,622 and query total
subset volume 1,238 against a bound of 1,374. All 358 zero biconditionals are
exact; 316 queries are zero and 42 are nonzero. These finite counts receive no
asymptotic credit.

## Compact Nodes

Each queried node is replayed through a compact divisor descriptor

```text
U_S = product_{P=(x,y) in S} (X-x),
V_S = V mod U_S.
```

All descriptor remainders interpolate their selected endpoints. The transcript
binds 2,834 queried coefficient slots and exact hashes for every `U_S`, `V_S`,
endpoint subset, scalar result, and tree path.

## Missing Oracle

The conditional oracle may perform one softly `O(n+N)` preprocessing of the
compact selected divisor `U,V` and the `N` target points. It must then accept
any balanced-node `U_S,V_S` and return the exact R174 signed subset product in
softly

```text
O(|S| + N)
```

work per query. It must be reusable across subsets, preserve both the secant
divided-difference and geometric-tangent charts, and avoid target-dependent
preprocessing and candidate-dependent inversions.

Given that oracle and the R163 charged output contract `K=B^(3/4)`, group
testing would cost

```text
softly O(n + K N) = B^(9/4+o(1)),
```

with R163 target-label and source-backpointer recovery costing `B^2`. This is
strictly below the `B^(5/2)` rho proxy, but it is conditional on the unsupplied
oracle and therefore receives no attack credit.

## Cost Boundary

```text
selected divisor degree n:                     B^(9/4)
target count N:                                B^(5/4)
R163 charged candidate output K:               B^(3/4)
conditional preprocessing:                     B^(9/4)
conditional subset-volume work:                B^(9/4)
conditional per-query target overhead K N:     B^2
conditional oracle total:                      B^(9/4)
R163 label/backpointer postprocessing:          B^2
direct expanded tree factors n^2 N:             B^(23/4)
represented target dual-Chow body N^2:          B^(5/2)
represented selected-pair queries n^2:          B^(9/2)
rho proxy:                                      B^(5/2)
```

The direct route multiplies the `N` target factors across the `n` R174 pair
factors for every selected leaf and remains `n^2 N`. Representing the target
dual-Chow form already reaches rho. No standard route implements the required
reusable scalar query below rho.

## Generic Boundary

R160 and Shoup exclude below-rho credit for an encoding-invariant generic
locator at these caps. Any admissible implementation must exploit explicit
prime-field coordinates and fully charge all conversion and preprocessing.
R175 neither evades nor weakens that boundary.

## Admission

Admit the scalar zero-product biconditional, exact balanced-tree recovery,
query-count bound, softly linear subset-volume bound, compact node descriptors,
and conditional `B^(9/4)` consequence of the stated oracle contract.

Do not admit the reusable scalar oracle, an unconditional below-rho locator,
deterministic hash-to-curve transfer, a generic-prime coordinate-family
algorithm, a complete ECDLP attack, or a Pollard-rho or Shoup improvement.

Disposition:

```text
ADMIT_SCALAR_SUBSET_ZERO_PRODUCT_BICONDITIONAL__BALANCED_GROUP_TESTING_RECOVERS_ALL_R174_ROOTS__QUERY_BOUND_1_PLUS_2KLOGN__SOFT_SUBSET_VOLUME_N__CONDITIONAL_REUSABLE_ORACLE_ENVELOPE_B9O4_BELOW_RHO__ORACLE_UNSUPPLIED__DIRECT_N2N_B23O4__GENERIC_INVARIANT_LOCATOR_EXCLUDED__NO_UNCONDITIONAL_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute the reusable scalar subset-incidence oracle. Reject leaf
enumeration, `N^2` or `n^2` represented Chow bodies, `nN` per-node factor
expansion, target-dependent transforms, candidate inversions, and unit-cost
multipoint, norm, resultant, root, or generic locator oracles.
