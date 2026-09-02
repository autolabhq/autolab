# P1553 torus C5 order-two three-minor rigidity gate R139

## Claim boundary

R139 proves that every three-by-three minor of the prime-order Fourier
matrix is nonzero when the field characteristic acts as inversion on the
prime-order roots. All actual norm-one families satisfy this condition.

Combining the theorem with R138 closes every structured represented
trinomial zero-test tree at polylogarithmic arbitrary-target query cost.
It does not close four-plus-mode nodes, nonzero-value tests, coordinate
comparisons, or arbitrary circuits. It supplies no source index,
known-RHS rank, factor logs, identical descent, Shoup improvement, or
ECDLP breakthrough.

Classification:

```text
ORDER_TWO_FROBENIUS_RIGIDITY_FORCES_EVERY_THREE_BY_THREE_PRIME_FOURIER_MINOR_NONZERO__EVERY_STRUCTURED_ATOM_COLOR_HAS_THREE_COLUMN_FULL_SPARK__R138_FIBER_COVER_NOW_UNCONDITIONAL_FOR_TRINOMIALS__REJECTING_PATH_DEPTH_B_THREE_QUARTERS_AND_ACCEPTING_PATH_DEPTH_B_FIVE_HALVES__2048_ACTUAL_AND_EXHAUSTIVE_NINE_SYNTHETIC_NORMALIZED_SWEEPS__STRUCTURED_FOUR_PLUS_LOW_SLP_AND_NONZERO_VALUE_ROUTES_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Three-minor theorem

Let `q` be prime, let `zeta` have order `q`, and suppose the field
characteristic `p` satisfies `p=-1 mod q`.

Affine changes in row and mode exponents, row and column permutations,
and nonzero diagonal scalings preserve singularity. Any minor on three
distinct rows and modes therefore normalizes to rows

```text
{0,1,r}
```

and modes

```text
{0,1,s},
```

where `r,s` avoid zero and one modulo `q`.

The normalized determinant vanishes exactly if

```text
(zeta-1)(zeta^(r*s)-1) = (zeta^s-1)(zeta^r-1).
```

Both sides are nonzero. Since Frobenius sends `zeta` to `zeta^(-1)`,
conjugating the equality multiplies the left side by
`zeta^(-(1+r*s))` and the right side by `zeta^(-(r+s))`. Cancelling the
original equal nonzero sides forces

```text
1+r*s = r+s mod q,
```

or

```text
(r-1)(s-1) = 0 mod q.
```

This contradicts normalization. Every three-by-three minor is therefore
nonsingular. No discrete logarithm or random-support assumption appears.

## Tree consequence

Every atom subset inherits three-column full spark, so a nonzero
represented trinomial has at most two zeros on a structured atom color.

For a color of size `m=B^(3/4+o(1))`, R138's ordered-tuple fiber lemma
forces a rejecting all-nonzero path to have depth

```text
ceil(m/2) = B^(3/4+o(1)).
```

If the all-nonzero leaf accepts, the inherited global trinomial root
bound forces

```text
q^(1/2+o(1)) = B^(5/2+o(1))
```

depth to cover the `q-o(q)` complement. Both exceed polylogarithmic
arbitrary-target work.

## Exact controls

- Eight actual order-two controls check 2,048 normalized minors.
- Nine synthetic prime pairs `p=6q-1`, through `q=47`, exhaust every
  normalized `(r,s)` pair.
- Every determinant is nonzero and every matrix rank is three.
- Every Frobenius multiplier identity and contradiction exponent replays.

The finite sweeps check the self-contained proof implementation and
receive no asymptotic credit.

## Admission

Fifteen of twenty-three obligations pass. Order-two three-minor
rigidity and the resulting structured trinomial-tree negative are
admitted. Four-plus-mode and nonzero-value selectors, source indexing,
rank, logs, identical descent, Pollard-rho improvement, Shoup
improvement, and breakthrough remain false.

Disposition:

```text
ADMIT_SELF_CONTAINED_ORDER_TWO_THREE_MINOR_RIGIDITY__REJECT_ALL_STRUCTURED_REPRESENTED_TRINOMIAL_ZERO_TEST_TREES_AT_POLYLOG_QUERY__ADMIT_2048_ACTUAL_AND_NINE_SYNTHETIC_SWEEPS_WITHOUT_ASYMPTOTIC_CREDIT__PRESERVE_STRUCTURED_FOUR_PLUS_LOW_SLP_AND_NONZERO_VALUE_ROUTES__NO_LOCATOR__NO_RANK__NO_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Probe order-two four-column minors and structured four-mode root fibers,
or construct a nonzero-value Frobenius-coordinate selector. Freeze every
mode, coefficient, circuit node, branch, and reverse C2+C3 source
pointer; replay positives and inverse empties; fit `B^(9/4+o(1))` state
and polylogarithmic arbitrary-target work; avoid field DLP; and charge
rank, logs, identical descent, memory, field operations, extension
degree, and bits.
