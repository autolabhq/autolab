# P1553 torus C5 linear-sketch circulant gate R124

## Claim boundary

R124 proves a full-dimension result for one precise descendant of the R123
nonrepresented circuit interface: universal target-independent linear
measurements of a C3 occurrence vector, followed by linear exact-count
decoding against every translate of a fixed C2 kernel.

It does not cover nonlinear preprocessing specialized to the coupled deck
powers, nonlinear membership-only decoding, adaptive data structures, or
general arithmetic circuits. It supplies no source locator, rank, factor
logs, identical descent, Shoup improvement, or ECDLP breakthrough.

Classification:

```text
C5_TRANSLATED_C3_C2_INNER_PRODUCT_EXACT__UNIVERSAL_LINEAR_SKETCH_DIMENSION_EQUALS_CIRCULANT_RANK__PRIME_ORDER_PROPER_BINARY_DECK_C2_KERNEL_HAS_FULL_RATIONAL_FOURIER_SUPPORT_Q_B5__SETUP_OVER_CAP__COUPLED_NONLINEAR_TARGET_DATA_STRUCTURE_OPEN__NO_SOURCE_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Translated-query identity

Write the deck incidence vector as `u` on the cyclic group of prime order
`q`, and let `*` denote cyclic convolution. Then

```text
u^(*5)(y) = sum_x u^(*3)(x) u^(*2)(y-x).
```

Thus every ordered C5 count is a translated inner product between the C3
occurrence vector and the C2 kernel. Five finite controls at orders 5, 7,
11, and 13 verify the identity against direct five-tuple enumeration.

## Universal linear-sketch theorem

Fix the C2 kernel `w=u^(*2)`. Suppose preprocessing stores `Sv`, where
`S` is an `s`-row linear map and `v` may be any C3 occurrence vector in
the ambient `q`-dimensional space. If every exact count

```text
<v, tau_y w>
```

is recovered by a linear decoder from `Sv`, every row of the circulant
`C_w` must lie in `rowspace(S)`. Therefore

```text
s >= rank(C_w).
```

Over a characteristic-zero splitting field,

```text
rank(C_w)
  = #{j : DFT(w)_j != 0}
  = #{j : DFT(u)_j != 0},
```

because `DFT(w)_j=DFT(u)_j^2`.

For prime `q` and a proper nonempty binary deck polynomial `U`, every
Fourier coefficient is nonzero over characteristic zero. At mode zero this
is the deck size. At a nonzero mode, `U(zeta_q^j)=0` would force

```text
Phi_q(X) = 1 + X + ... + X^(q-1)
```

to divide `U` over the rationals. A binary polynomial of degree at most
`q-1` can satisfy that condition only when it is zero or `Phi_q`, excluded
by the proper nonempty hypothesis. Hence `rank(C_w)=q`.

The five controls also have full rational circulant rank. Selected auxiliary
finite fields have all Fourier modes nonzero, but those finite checks receive
no asymptotic credit and are not needed by the characteristic-zero proof.

## Cost boundary

The inherited generic-prime scaling has

```text
q = B^(5+o(1)).
```

The universal linear exact-count sketch therefore needs `B^(5+o(1))`
measurements, above the `B^(9/4+o(1))` setup cap. Applying a zero test after
the exact linear count decoder does not change this dimension.

The standard represented alternative stores C3 in `B^(9/4+o(1))` space
and scans C2 in `B^(3/2+o(1))` work per target. It fits setup but misses the
polylogarithmic query requirement.

## Scope limits

The theorem is universal over all ambient C3 vectors for a fixed C2 kernel.
The actual ECDLP candidate has the coupled pair

```text
(w,v) = (u^(*2),u^(*3)).
```

A nonlinear preprocessing map may exploit that coupling, and a nonlinear
decoder may return only membership rather than the exact count. R124 proves
no lower bound for those routes, adaptive cell-probe or RAM structures,
bounded-error sketches, source locators, or arbitrary arithmetic circuits.

## Admission

Twelve of twenty obligations pass. The translated identity and scoped
universal linear-sketch negative are admitted. The coupled nonlinear
membership/source interface, known-RHS rank, logs, identical descent,
Pollard-rho improvement, Shoup improvement, and breakthrough obligations
remain false.

Disposition:

```text
ADMIT_EXACT_TRANSLATED_C3C2_INNER_PRODUCT_AND_FULL_CIRCULANT_RANK_THEOREM_FOR_UNIVERSAL_LINEAR_COUNT_SKETCHES_ONLY__REJECT_THAT_GRAMMAR_AT_B5_STATE__PRESERVE_COUPLED_NONLINEAR_TARGET_DATA_STRUCTURE__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Freeze one target-specialized nonlinear zero-test for the coupled pair
`(u^(*2),u^(*3))`. Test a multilinear, rational-Krylov, adaptive-probe, or
nonlinear-fingerprint construction against exact empty answers and five
projective sources under `B^(9/4+o(1))` setup and polylogarithmic arbitrary
target query caps, with complete noncancellation and source-to-descent costs.
