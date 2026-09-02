# P1553 5A/5C Actual-Deck Non-Mergeable Target Pullback Gate R105

Date: 2026-07-29

Status: `CONDITIONAL_NORM_JET_SOURCE_ADJOINT_EXACT__SCALAR_CONSTRUCTOR_OPEN`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

If a target-injected whole-deck scalar norm/count circuit exists, does exact
integer multiplicity and coupled source return force it to emit provenance
leaves or an over-cap source dictionary?

R105 freezes one scalar source norm, eleven unit/power-sum deformation
channels, the lowest homogeneous jet, and Newton source inversion before
outcomes.

## Marker-jet identity

For canonical unordered sources `s`, let

```text
z_s(T) = kappa(endpoint(s)) - kappa(T) in F_(p^2)
N_T = product_s z_s(T).
```

Use the injective key `kappa((x,y))=x+w*y`, with `kappa(O)=0`. Deform every
factor by

```text
z_s + t0 + sum_(j=1)^10 marker_j(s) t_j.
```

If the target fiber is `Z` and `m=|Z|`, the first nonzero homogeneous jet is

```text
product_(r not in Z) z_r
  * product_(s in Z) (t0 + marker(s) dot t).
```

Therefore:

- the lowest nonzero jet order is the exact integer multiplicity `m`;
- normalization by the `t0^m` coefficient removes the nonzero product;
- factoring the homogeneous jet returns one coupled marker vector per
  source.

The ten markers are the first five power sums of the `A` indices and the
first five power sums of the `C` indices. Newton identities reconstruct each
monic degree-five index polynomial. Testing the public atom indices,
including repeated roots, recovers the unordered source in
`O(|A|+|C|)=B^(3/5)` operations.

## Exact replay

All eight actual families and 32 unique, maximum-multiplicity, blind, and
projective-identity controls replay exactly.

Two actual families contain double fibers. Their order-two marker jets factor
into both coupled source vectors, and both reconstructed sources group-replay.
All other positive controls are simple. Blind controls have nonzero scalar
norm and return bottom.

The finite verifier explicitly multiplies all `B^5` source factors. It gets
no candidate work credit.

## Conditional cost

For simple and double fibers, the eleven marker channels and order-two jet
have constant exponent. Source reconstruction costs `B^(3/5)`, below the
`B^(5/4)` online cap.

Baur-Strassen reverse differentiation gives constant-factor first-derivative
overhead for a scalar arithmetic circuit. Truncated order-`m` jets cost a
polynomial in `binomial(10+m,m)`. R105 proves only the actual `m<=2`
boundary; it does not prove a generic multiplicity tail or integer no-wrap
gate.

## Scope

R105 conditionally removes provenance-leaf and source-dictionary costs from
the simple/double-fiber scalar-circuit route. It does not construct the
scalar target norm/count circuit from compact `D_A,D_C`, and it does not
make the verifier-only `B^5` source product an algorithm.

## Admission

Passed obligations: `18/29`

Conditional source adjoint admitted: `true`

Scalar target norm constructor complete: `false`

Full lane admitted: `false`

Rank, factor logs, identical descent, generic multiplicity/no-wrap, charged
workspace, Shoup improvement, and breakthrough remain absent.

## Exactly one next action

Construct or refute the scalar target norm/count circuit isolated by R105.
It must consume compact `D_A,D_C` and `T` as one non-mergeable program,
support the eleven fixed deformations, avoid residual, translated
coefficient, quotient, source-factor, and determinant-oracle bodies, fit
both caps, and prove generic multiplicity and integer lift before any rank,
factor-log, descent, or Shoup claim.
