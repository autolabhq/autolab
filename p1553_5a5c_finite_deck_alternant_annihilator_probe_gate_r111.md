# P1553 R111 finite-deck alternant annihilator gate

## Scope

R111 tests the strongest direct finite-deck use of the R110
position-separated `L(11O)` Abel alternant: a univariate zero-mask
annihilator in its raw determinant value, together with one annihilator per
five-A or five-C source. It does not claim a lower bound for arbitrary
nonlinear circuits.

## Exact replay

- All 135744 canonical `5A+5C` sources on 8 actual and 8 matched instances
  were enumerated.
- The signed `Lambda^5 x Lambda^6` exterior pairing agrees with the direct
  `11 x 11` determinant on every frozen Laplace control.
- Determinant zero agrees with the direct elliptic target equation on every
  enumerated source.
- The two R105 multiplicity-two targets, R105 marker vectors, and R108
  canonical cycle weight 14400 replay exactly.
- The already-constructed zero masks have rank at most 2. This receives no
  constructor credit because the mask is the object that remains expensive
  to produce.

## Raw-value annihilator

For the finite set `D` of determinant values, the exact restricted zero mask
is

`q_D(z) = product_(d in D_nonzero) (1 - z/d)`.

Any polynomial equal to one at zero and zero at every distinct nonzero value
in `D` has degree at least `|D_nonzero|`; the displayed polynomial attains
that degree. On every R111 instance, distinct nonzero values occupy more than
98 percent of the full toy source body. The largest exact degree is 25868.
These observed degrees are not an asymptotic theorem.

Conditioning on one side does not repair the standard coefficient layout.
There are `B^(2+o(1))` five-A sources with degree up to `B^(3+o(1))`, or
`B^(3+o(1))` five-C sources with degree up to `B^(2+o(1))`; either explicit
coefficient family has `B^(5+o(1))` capacity.

## Disposition

Admit an exact scoped negative for raw univariate and standard
side-conditioned determinant annihilators. Preserve the representation-
sensitive exception: quotient the nonzero alternant gauge factors and solve
the resulting endpoint `Query2P1` membership and subset-stable source
recovery problem inside setup `B^(9/4+o(1))` and fresh work/workspace
`B^(5/4+o(1))`.

No exact source locator, factor-log solve, identical target descent,
generic-prime Shoup improvement, or ECDLP breakthrough is supplied.

## Exactly one next action

Construct or refute one gauge-normalized endpoint `Query2P1` index for the
position-separated alternant, including exact target updates, source
unranking, multiplicity, markers, factor logs, and identical descent.
