# P1553 5A/5C Scalar Target Norm/Count Circuit Gate R106

Date: 2026-07-29

Status: `CHARACTER_DIAGONAL_BODY_B5__NONCHARACTER_NORM_CIRCUIT_OPEN`

Breakthrough: `false`

Shoup-bound improvement: `false`

## Frozen question

Can the scalar target count isolated by R105 be evaluated through an exact
character-diagonal representation of the canonical `5A+5C` endpoint
histogram inside the direct caps?

R106 freezes the complex-character mode grammar, target phase, eleven R105
marker channels, actual/matched decks, and a composite-order positive control
before outcomes.

## Prime-cyclic uncertainty theorem

Tao's sharp uncertainty principle states that for nonzero `f` on the cyclic
group of prime order `q`,

```text
|supp(f)| + |supp(Fourier(f))| >= q+1.
```

Primary source: Terence Tao, *An uncertainty principle for cyclic groups of
prime order*, arXiv `math/0308286`, Mathematical Research Letters 12 (2005).

Let `H_A` and `H_C` be the canonical five-multiset endpoint histograms. The
target count is

```text
H = H_A * H_C,
Fourier(H) = Fourier(H_A) Fourier(H_C).
```

The two side transforms can have at most `|supp(H_A)|-1` and
`|supp(H_C)|-1` zero modes. Therefore

```text
|supp(Fourier(H))|
  >= q - |supp(H_A)| - |supp(H_C)| + 2.
```

Since the side supports are at most `B^2` and `B^3`, while `q=B^5`, the
canonical full-count spectrum has `B^(5-o(1))` live modes.

## Actual and matched controls

All eight actual and eight matched random decks satisfy the exact bound.
The minimum finite live-mode fraction is greater than `0.9984`.

For the largest family:

```text
q                       16,780,523
five-A support                  56
five-C support                 462
live-mode lower bound    16,780,007.
```

The composite positive control uses the even subgroup indicator on
`Z/16Z`. It has primal support 8 and Fourier support `{0,8}` of size 2, so
`8+2<17`. This confirms that the sharp additive bound depends on prime cyclic
order and does not reject genuine subgroup-sparse spectra.

## Charged character route

An explicit mode table, dense per-target character sum, or all-target FFT
has exponent `B^5`, above setup and online caps. Endpoint character
coordinates additionally require unavailable discrete-log labels.

The eleven unit/power-sum marker channels change only a constant factor, not
the mode exponent.

## Scope

This is an actual-family obstruction for explicit complex-character
diagonalizations and equivalent mode tables. It is not an arithmetic-circuit
lower bound. It does not cover finite-field Fourier representations where
the relevant minors vanish, or a target-injected nonlinear
resultant/norm/rational circuit that never exposes character modes.

## Admission

Passed obligations: `12/26`

Character-diagonal constructor class closed: `true`

Non-character scalar constructor open: `true`

Full lane admitted: `false`

No known-RHS rank, factor logs, identical descent, generic multiplicity
lift, Shoup improvement, or breakthrough is present.

## Exactly one next action

Construct or refute one non-character target-injected algebraic
norm/resultant circuit over the compact atom divisors. Freeze every nested
resultant, quotient-free transpose, marker deformation, integer lift, and
intermediate dimension; forbid character/DLP coordinates, residual and
translated-coefficient bodies, `B^5` Macaulay/source bodies, and unit-cost
determinants; require both caps and complete exceptional replay.
