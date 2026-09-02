# P1553 M6 principal-target Pontryagin-resultant gate R176

Date: 2026-08-01

## Scope

R176 replaces the R175 signed subset product by evaluation of the compact R167
principal target function on a selected Pontryagin product cycle. It verifies
ordinary and candidate-specialized Weil reciprocity for every queried tree
node.

The identity is exact and removes the explicit target-factor loop after a
pair-sum cycle has been represented. That cycle has degree `mn`, so the
standard route remains above rho. This is not a circuit lower bound, an ECDLP
algorithm, or a Pollard-rho or Shoup improvement.

## Principal Signed Incidence

Let the R167 target witness be

```text
h = F_num / F_den,
div(h) = sum_T [T] - sum_R [R],
```

where the auxiliary poles `R` avoid every selected pair sum. Then

```text
h(P+Q) = 0  if and only if  P+Q is a retained target.
```

This condition is sign-sensitive. The diagonal `P=Q` is evaluated by the
elliptic group law, so no derivative of the interpolated `V` side table and no
separate tangent formula is needed.

Across the six controls, all 8,922 selected pair sums avoid the auxiliary
poles. Exactly 241 values vanish: 5 diagonal and 236 off-diagonal. Their 140
left roots are exactly the R175 and R167 roots.

## Pontryagin Norm

For a selected subset `A` of size `m` and the full selected divisor `D` of
degree `n`, define the effective cycle

```text
A * D = sum_{P in A, Q in D} [P+Q].
```

It has degree `mn`, and its principal target norm is

```text
H(A) = h(A * D) = product_{P in A, Q in D} h(P+Q).
```

`H(A)=0` exactly when `A` contains an R175 candidate. The six controls replay
the same 358 balanced tree nodes and identical zero projections as R175.

The group sum of the cycle is

```text
sum(A * D) = n sum(A) + m sum(D).
```

Every queried node verifies this identity exactly.

## Principal Completion

If the cycle sum is nonzero, append its negative `C` and construct

```text
div(f_(A*D)) = A*D + [C] - (mn+1)[O].
```

If the cycle sum is zero, use `A*D-mn[O]`. A generalized Miller tree evaluates
this function with one merge per represented pair-cycle point, up to constant
terms.

All completion values and auxiliary-pole evaluations are units in the finite
controls. Deterministic merge-order retries are used only to avoid removable
intermediate vertical-line collisions; the maximum retry index is one and is
bound in every transcript.

## Weil Reciprocity

For disjoint support,

```text
h(div f) = f(div h).
```

Writing `d=mn+1` when a finite completion is appended gives

```text
h(A*D)
  = f(targets) / f(auxiliary poles) * h(O)^d / h(C).
```

The `h(C)` factor is omitted when `C=O`. Candidate nodes are the zero
specialization of this identity and are never inverted.

All 358 completed-cycle reciprocity identities are exact:

```text
literal disjoint-support nonzero identities:  42
candidate-specialized zero identities:       316
```

## Represented-Degree Boundary

Reciprocity removes the explicit target loop only after the pair-sum cycle or
its principal function has been constructed. Both have represented size
`Theta(mn)`.

Across the queried finite nodes:

```text
pair-cycle degree volume:                    56,468
Miller merges:                               56,468
charged Miller probe evaluations:           868,648
```

These are finite verification counts and receive no asymptotic credit.

At the root `m=n`. Across a balanced query tree, total queried subset size is
softly `O(n)`, but every subset is paired with all `n` points of `D`. Therefore
both root and total represented pair-cycle volume are softly `Theta(n^2)`.

## Cost Boundary

```text
compact target principal witness h:             B^(5/4)
compact selected divisor state:                 B^(9/4)
root Pontryagin cycle degree n^2:                B^(9/2)
balanced queried pair-cycle volume:             B^(9/2)
represented completed Miller function:          B^(9/2)
direct h-SLP evaluation on pair cycle n^2 N:     B^(23/4)
fast represented evaluation after pair listing: B^(9/2)
conditional factored trilinear resultant:        B^(9/4)
R163 label/backpointer postprocessing:           B^2
rho proxy:                                       B^(5/2)
```

Ordinary Weil reciprocity improves on direct `n^2 N` expansion but remains far
above rho because it preserves the `mn` represented input. This closes only
standard pair-cycle, represented-function, and generalized Miller routes. It
does not prove that a factored arithmetic circuit must expand to `mn` state.

## Open Primitive

The remaining primitive is a factored trilinear elliptic resultant that takes

```text
(U_A,V_A), (U_D,V_D), h
```

and returns `h(A*D)` without materializing the pair-sum cycle. The required
contract is one softly `O(n+N)` setup for fixed `D,h`, followed by softly
`O(m+N)` work per balanced node. Combined with R175 and the R163 output
contract, this would retain the conditional `B^(9/4)` total.

R169 already found full rank for the tested ordinary diagonal displacement
generators. Recreating an `n` by `n` kernel under another name does not satisfy
the open contract.

## Admission

Admit the principal signed incidence function, pole-free pair controls,
diagonal group-law handling, exact R175 tree replay, Pontryagin degree and
group-sum identities, completed generalized Miller evaluations, and all 358
Weil-reciprocity identities.

Do not admit the factored trilinear resultant, a circuit lower bound,
deterministic hash-to-curve transfer, a generic-prime coordinate-family
algorithm, a complete ECDLP attack, or a Pollard-rho or Shoup improvement.

Disposition:

```text
ADMIT_PRINCIPAL_TARGET_SIGNED_INCIDENCE__H_OF_P_PLUS_Q_ZERO_IFF_POSITIVE_TARGET_SUM__DIAGONAL_BY_GROUP_LAW__358_R175_TREE_QUERIES_REPLAYED__358_COMPLETED_PONTRYAGIN_WEIL_IDENTITIES__PAIR_CYCLE_DEGREE_MN__STANDARD_ROOT_AND_TREE_STATE_N2_B9O2__RECIPROCITY_PRESERVES_MN_INPUT__FACTORED_TRILINEAR_RESULTANT_OPEN__NO_CIRCUIT_LOWER_BOUND__NO_RHO_SHOUP_BREAKTHROUGH
```

## Next Action

Construct or refute the factored trilinear elliptic resultant. Reject explicit
`mn` pair sums, a degree-`mn` principal function or Miller program, `n^2`
tensor or displacement state, target-dependent per-node preprocessing,
candidate inversions, and unit-cost resultant, norm, root, count, marginal,
rank, source, DLP, or generic locator oracles.
