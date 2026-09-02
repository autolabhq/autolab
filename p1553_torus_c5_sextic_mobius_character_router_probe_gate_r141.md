# P1553 R141 sextic Mobius character router gate

Date: 2026-07-29

## Decision

Admit the sextic Mobius character as a polylogarithmic nonzero-value
inverse separator. Admit the `Omega(q)` translated-linear rank boundary.
Withhold source-router, algorithm, rho, Shoup, and breakthrough
promotion.

## Admitted

- In Cayley coordinates `X(t)=(1+u*t)/(1-u*t)`, `u^2=d`, the R140
  Mobius claw has the base-field form
  `d*(t_x*t_z+t_x*t_w+t_z*t_w)=3`.
- `chi_z(x)=T_z(x)^q` takes one of six norm-one coset values and is
  evaluable with `O(log q)` field operations without a discrete
  logarithm.
- For fixed `x!=x^-1`, the equality
  `chi_z(x)=chi_z(x^-1)` is a nontrivial fixed-degree character-sum
  condition in `z`. Weil square-root cancellation gives an equality
  fraction `1/6+O(q^-1/2)`. A union bound supplies
  `O(log |S|)` fixed parameters separating every value in any
  inversion-disjoint support `S` from its inverse.
- The translated Fourier coefficients are fixed-conductor torus
  character sums of size `O(sqrt(q))`. Parseval forces
  `Omega(q)=B^(5+o(1))` nonzero coefficients and translated-linear
  circulant rank.
- All eight actual controls separate every positive from its inverse
  with at most four deck parameters. Their `C2 x C3` character matrices
  have full row rank and their multiplicative defects attain all six
  cosets.
- All eight synthetic controls have Fourier support at least `q-1`.

The square-root character-sum input is attributed to Andre Weil,
"On Some Exponential Sums," PNAS 34 (1948), 204-207,
doi:10.1073/pnas.34.5.204.

## Withheld

- Inverse separation is not a source selector.
- Full finite row rank and synthetic Fourier support receive no
  asymptotic credit independently of the character-sum argument.
- The `Omega(q)` result covers translation-invariant linear
  convolution or sketches of the six labels only. It is not a lower
  bound for nonlinear circuits, adaptive data structures, RAM, or cell
  probes.
- No compact nontranslation composition law, complete five-source
  index, known-RHS rank, factor logs, identical descent,
  generic-prime-family transfer, rho improvement, or Shoup improvement
  is supplied.

## Next gate

Construct a nonlinear C2 source router from the sextic characters
without materializing their translated linear orbit. Freeze every
parameter, label, branch, candidate C2 pointer, C3 certificate,
inverse-empty path, and reverse five-source pointer. Require
`B^(9/4+o(1))` total state and polylogarithmic arbitrary-target work,
avoid field DLP, and charge rank, factor logs, identical descent,
memory, field operations, extension degree, and bit complexity.
