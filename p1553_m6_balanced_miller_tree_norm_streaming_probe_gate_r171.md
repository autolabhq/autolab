# P1553 M6 balanced Miller-tree norm-streaming gate R171

Date: 2026-08-01

## Scope

R171 instantiates the generalized Miller straight-line program asserted by
R167 and tests the remaining R170 streaming hypothesis. The exact balanced
tree is real and compact. Node-local norm streaming does not lower total work:
after every internal sum and correction factor is cancelled, it returns the
same `N` target translates already isolated by R170.

This is a scoped negative for per-line and per-leaf streaming. It is not an
arithmetic-circuit, RAM, cell-probe, elliptic-resultant, or generic-group lower
bound.

## Exact tree

For a zero list `Z=(Z_1,...,Z_m)`, a leaf has function `1` and residual point
`Z_i`. Merge two states with residual points `A,B` by

```text
f_(A+B) = f_A f_B l_(A,B) / v_(A+B).
```

The resulting divisor is

```text
sum_i [Z_i] - [sum_i Z_i] - (m-1)[O].
```

A balanced tree has exactly `m-1` line merges and logarithmic depth. When the
zero list sums to `O`, it is the required principal function. Six controls
construct numerator and denominator trees and compare each with the dense
R167 Riemann-Roch witness at 16 safe points. All 192 ratios are constant and
nonzero.

## Line norm

Let `f0(Q)=U(x(Q))`. For a generic shift `P`, Weil reciprocity gives

```text
 product_(Q in S union -S) g_(A,B)(Q+P)
 ------------------------------------------------
                 g_(A,B)(P)^(2n)

       f0(A-P) f0(B-P)
   = ----------------------- .
     f0(A+B-P) f0(-P)
```

The controls verify 1,222 admissible line/shift identities exactly; 58 rows
with a pole or nonunit denominator are recorded rather than inverted.

At a selected endpoint, `f0(-P)=U(x(P))=0`. Individual line ratios therefore
cannot be specialized independently. The numerator and denominator origin
factors must first cancel symbolically. The full tree quotient then replays
all selected rows, including all 140 candidate-zero rows, exactly.

## Telescoping

Multiplying the line identities cancels every internal partial sum. A
principal size-`m` tree leaves

```text
product_i f0(Z_i-P) / f0(-P)^m.
```

The R167 numerator and denominator have equal size and share their completion
zero. Their origin and completion factors cancel. Multiplying the public
anchor/auxiliary correction cancels every denominator leaf. Exactly the
original target leaves survive: 40 across the six finite controls.

Balancing lowers depth and permits `B^(9/4)` live endpoint state, but it does
not lower node-local or residual-leaf work:

```text
balanced tree state:                    B^(5/4)
live endpoint value vector:             B^(9/4)
node-local line-norm streaming:          B^(7/2)
telescoped target-leaf evaluation:       B^(7/2)
raw signed-grid tree expansion:          B^(23/4)
represented aggregate output:            B^(9/4)
rho proxy:                               B^(5/2)
```

## Literature boundary

Miller gives the tangent-and-cord recurrence. Enge describes factored-line
storage and direct divisor evaluation while noting the dense degree growth;
neither supplies a moving all-pairs batch.

Moroz and Schost compute a represented bivariate resultant truncation in
softly `O(kd)` operations. Even the optimistic substitution `k=n,d=N` is
`nN`, and their input contract does not compile this elliptic divisor list.
Bhargava et al. evaluate a represented coefficient vector at explicit points
in nearly linear time; they do not construct or fuse the target-dependent
elliptic product from its leaves.

## Admission

Admit the explicit balanced Miller trees, equality with the dense witnesses
up to scalar, exact selected-row replay, node-local reciprocity on generic
shifts, and the signed telescoping identity.

Do not admit node-local streaming below rho, a nonlocal batched constructor,
deterministic hash-to-curve transfer, generic-prime algorithm, Pollard-rho or
Shoup improvement, or an ECDLP breakthrough.

Disposition:

```text
ADMIT_EXPLICIT_BALANCED_GENERALIZED_MILLER_TREES__DENSE_WITNESS_EQUALITY_UP_TO_SCALAR__EXACT_R167_NORM_REPLAY__NODE_LOCAL_WEIL_HOMOMORPHISM__INTERNAL_SUMS_AND_AUXILIARY_LEAVES_TELESCOPE_TO_EXACTLY_N_TARGET_FACTORS__BALANCING_REDUCES_DEPTH_AND_MEMORY_NOT_NN_B7O2_WORK__NONLOCAL_BATCHED_LEAF_TRANSLATE_OPERATOR_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next action

Construct or refute a nonlocal batched elliptic leaf-translate product
operator. From `U` and compact selected/target divisors, it must emit
`product_j U(x(T_j-P)) mod U` in softly `O(n+N)` work, preferably
`B^(9/4+o(1))`, without visiting the `n`-by-`N` pair grid, materializing `N`
dense quotient-ring elements, or invoking an uncharged norm, resultant, or
multipoint oracle.
