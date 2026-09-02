# P1553 R140 order-two four-minor claw gate

Date: 2026-07-29

## Decision

Admit the exact four-minor nonrigidity and Mobius-claw factorization as
a scoped negative result. Withhold selector, algorithm, rho, Shoup, and
breakthrough promotion.

## Admitted

- For normalized progression rows `{0,1,3,4}` and modes
  `{0,1,a,b}`, the determinant reduces to an explicit six-factor
  identity.
- Away from repeated or trivial modes, singularity is equivalent to
  the fractional-linear claw
  `z^b=-(z^(a+1)+z^a+z)/(z^a+z+1)`.
- The Mobius map is an involution and order-two Frobenius keeps its
  image in the full norm-one group.
- All 12 available actual color progressions have bounded known-mode
  claws. Every resulting minor has rank three, every kernel coefficient
  is nonzero, and every kernel polynomial vanishes on exactly
  progression rows `{0,1,3,4}` among the six checked sources.
- These witnesses refute a universal four-column full-spark extension
  of R139.

## Withheld

- Computing the Mobius image does not reveal its integer Fourier mode.
  Using it as `X^b` for arbitrary `X` requires mode recovery or a
  different representation.
- The standard known-mode collision or generic DLP baseline costs
  `q^(1/2)=B^(5/2)`, above the `B^(9/4)` setup cap. This baseline is not
  claimed as a lower bound against every structured algorithm.
- The finite scans, bounded modes, and constant four-of-six root sets
  receive no asymptotic selector credit.
- No asymptotically dense atom root set, cap-compliant source index,
  known-RHS rank, factor logs, identical descent, generic-prime-family
  transfer, rho improvement, or Shoup improvement is supplied.

## Next gate

Admit a successor only if it does one of the following without a field
discrete-log oracle:

1. Finds a known-exponent Mobius claw in
   `q^(9/20+o(1))=B^(9/4+o(1))` total setup and state.
2. Uses the directly computed Mobius image without recovering its
   hidden integer mode.
3. Constructs a genuinely nonzero-value Frobenius-coordinate selector
   with `B^(9/4+o(1))` state and polylogarithmic arbitrary-target work.

Every successor must replay positive sources and inverse empty targets,
freeze all coefficients and branches, and charge relation rank, factor
logs, identical descent, memory, field operations, extension degree,
and bit complexity.
