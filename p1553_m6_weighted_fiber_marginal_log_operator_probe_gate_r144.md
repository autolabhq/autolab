# P1553 M6 weighted fiber-marginal log-operator gate R144

## Claim boundary

R144 proves an exact source-free reduction from six-factor relation fibers to
known-RHS factor-log equations and shifted target descent. It gives a
conditional `N^(9/20+o(1))` precomputation envelope at the R115 Cartesian
vertex. It does not construct the required weighted summation-polynomial/FFE
count index, preserve an offline/online split under differentiation, prove
structured asymptotic rank or density, complete projective integer lifting,
or supply an ECDLP algorithm.

It claims no Pollard-rho improvement, Shoup improvement, or breakthrough.

Classification:

```text
ORDERED_M6_RELATIONS_COMPRESS_TO_WEIGHTED_FIBER_COUNT_AND_ATOM_MARGINALS__KNOWN_RHS_LOG_IDENTITY_AND_SOURCE_FREE_SHIFTED_DESCENT_EXACT__EIGHT_ACTUAL_AGGREGATE_MATRICES_FULL_MEANINGFUL_RANK__CONDITIONAL_PRECOMPUTATION_N9O20__WEIGHTED_S13_COUNT_CIRCUIT_RANK_DENSITY_AND_GENERIC_TRANSFER_OPEN__NO_SHOUP_BREAKTHROUGH
```

## Weighted fiber

At the R115 vertex, let the Cartesian factor base be

```text
F = {A_a + C_c}.
```

For target `T`, attach formal weights to the atoms and define

```text
Z_T(u,v)
  = sum prod_j u_(a_j) v_(c_j),
```

where the sum is over all ordered six-factor sources satisfying

```text
sum_j (A_(a_j) + C_(c_j)) = T.
```

The exact ordered fiber count is

```text
c_T = Z_T(1,1).
```

The aggregate occurrence marginals are logarithmic weight derivatives:

```text
d^A_T(a) = (u_a partial Z_T / partial u_a)(1,1),
d^C_T(c) = (v_c partial Z_T / partial v_c)(1,1).
```

No individual relation row or source is present in this interface.

## Known-RHS identity

Use the meaningful Cartesian log coordinates

```text
x0         = log(A_0 + C_0),
delta_a(a) = log(A_a) - log(A_0),  a>0,
delta_c(c) = log(C_c) - log(C_0),  c>0.
```

Then

```text
log(A_a + C_c) = x0 + delta_a(a) + delta_c(c),
```

and the log dimension is `|A|+|C|-1`.

If `T=[k]G`, summing the six-factor log equation over every ordered source in
the target fiber gives

```text
6*c_T*x0
  + sum_(a>0) d^A_T(a)*delta_a(a)
  + sum_(c>0) d^C_T(c)*delta_c(c)
  = k*c_T mod q.
```

Thus one known scalar target supplies one aggregate relation row. If enough
such rows have full meaningful rank, they solve every Cartesian factor log
without selecting or opening an individual source.

## Source-free descent

Once factor logs are known, any target fiber with `c_T != 0 mod q` gives

```text
log_G(T)
  = c_T^(-1) times the aggregate marginal dot product.
```

For an arbitrary challenge `Q`, choose a known shift `r`, query
`T=Q+[r]G`, and subtract `r` from the recovered target log. This is an
identical descent identity; source recovery is unnecessary.

## Exact controls

The verifier exhausts canonical six-factor sources on all four R82 curve
families and both offsets, with multinomial weights restoring ordered
fibers. The meaningful dimensions and aggregate ranks are

```text
dimensions: 4, 4, 7, 7, 8, 8, 10, 10
ranks:      4, 4, 7, 7, 8, 8, 10, 10.
```

On every control:

- all `A` and `C` marginal sums equal `6*c_T`;
- every known-RHS aggregate identity is exact;
- selected independent fibers recover all meaningful and Cartesian factor
  logs;
- every positive target log and every checked shifted challenge replays;
- every selected empty target has count zero and is rejected;
- sampled six-factor sources replay projectively.

The build uses BSGS labels only in the independent verifier. The candidate
operator consumes no scalar labels. Enumeration, finite rank, and verifier
labels receive no asymptotic credit.

## Conditional cost

With

```text
N = B^(5+o(1)),
|A| = B^(1/12+o(1)),
|C| = B^(3/4+o(1)),
|F| = B^(5/6+o(1)),
```

the ordered six-factor body has exponent `B^5`, while the meaningful log
dimension is `B^(3/4)`. Conditional on a constant-density structured target
fiber and a reusable weighted count/marginal index:

```text
selected marginal matrix output: B^(3/2+o(1))
dense meaningful-log elimination: B^(9/4+o(1))
total conditional precomputation:  B^(9/4+o(1)) = N^(9/20+o(1))
successful shifted descent:        B^(3/4+o(1)) = N^(3/20+o(1)).
```

These exponents are below the `B^(5/2)=N^(1/2)` Pollard-rho scale, but they
are only a conditional reduction.

## Differentiation boundary

Baur and Strassen show that all first derivatives of one complete arithmetic
circuit can be obtained with constant-factor nonscalar overhead. This
supports the algebraic count-to-marginal reduction.

It does not automatically preserve an offline setup/online query split. If
the weighted count index uses `B^(9/4)` preprocessing, reversing the entire
setup for every target destroys the envelope. The missing object must compile
reusable derivative-capable state so that a fresh positive target emits its
`B^(3/4)` marginal vector without replaying preprocessing.

## Admission

Nineteen of twenty-nine obligations pass. The weighted-fiber identity,
known-RHS aggregation, source-free descent reduction, all finite semantic
controls, and conditional exponent envelope are admitted. The weighted
count index, offline/online transposed derivative index, structured
rank/density theorem, generic transfer, and campaign lane are not admitted.

Disposition:

```text
ADMIT_EXACT_SOURCE_FREE_FIBER_MARGINAL_LOG_REDUCTION__ADMIT_CONDITIONAL_N9O20_ENVELOPE__REJECT_FACTOR_LOG_OR_DESCENT_CREDIT_WITHOUT_WEIGHTED_COUNT_AND_TRANSPOSED_INDEX__NO_GENERIC_TRANSFER__NO_RHO__NO_SHOUP__NO_BREAKTHROUGH
```

## Exactly one next action

Construct the scalar-blind weighted six-factor fiber-count circuit
`Z_T(u,v)` directly from the compact `A/C` divisors and the factor-level
`S7` or expanded `S13` relation. It must build reusable state in
`B^(9/4+o(1))`, answer a fresh target count in polylogarithmic work, and
emit the `B^(3/4+o(1))` marginal vector without replaying setup. Freeze all
projective charts, multiplicity and integer lifting, known-target
rank/density, factor-log solve, shifted identical descent, memory, field
operations, and bit cost. It may consume no DLP, root, count, marginal, rank,
or source oracle.

## Primary sources

- Semaev, *Summation polynomials and the discrete logarithm problem on
  elliptic curves*, <https://eprint.iacr.org/2004/031>.
- Baur and Strassen, *The Complexity of Partial Derivatives*,
  <https://doi.org/10.1016/0304-3975(83)90110-X>.
