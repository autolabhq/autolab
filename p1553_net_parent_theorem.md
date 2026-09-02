# P1553 elliptic-net aggregate constructor gate

Task: `TASK-20260719-P1553-NET-P1`
Record: coordinator-side reconstruction for independent comparison
Evidence: theorem and literature only; no run

## Terminal statement

The standard rank-two elliptic-net and division-polynomial interfaces give an
exact zero test for one nondegenerate pair-pair target on a scalar fifth-deck
orbit. Their zero support is invariant under quadratic net rescaling. They do
not give a closed recurrence for the product over all pair-pair targets.

The first nontrivial mixed source/target seed cannot be obtained from the
normalization values `W(e_i)=W(e_i+e_j)=1`. Expanding the individual seed
block exposes `Theta(B^4)` pair-pair points. Compressing one mixed seed product
by the standard exact pair-divisor resultant costs `B^(2+o(1))` online work
and workspace. That single fixed-label seed already exceeds the frozen
`B^(5/4+o(1))` online cap.

This is a scoped negative for standard componentwise elliptic-net,
division-polynomial, supplied-term, and Hadamard-product recurrence routes. It
is not a lower bound against an unknown list-specific arithmetic circuit or
data structure, and it is not a Shoup-bound improvement or an ECDLP result.

## Frozen scalar-orbit interface

Retain the reviewed P1553 R4 setup. Let the selected fifth deck be

```text
a_j=A_0+[j]T,                 0<=j<B,
```

and let each source-labelled pair deck have `n=Theta(B^2)` occurrences:

```text
D_12={(labels,u)},            D_34={(labels,v)}.
```

For a fresh target `R`, define

```text
q_(u,v)=A_0+u+v-R.
```

Then

```text
u+v+a_j=R  iff  [j]T+q_(u,v)=O.
```

To avoid the normalized exceptional indices `e_2=(0,1)` and
`e_1+e_2=(1,1)`, shift the second point:

```text
Q_(u,v)=q_(u,v)-2T,
W_(u,v)(j+2,1)=Psi_(j+2,1)(T,Q_(u,v)).
```

On Stange's nondegenerate rank-two chart,

```text
W_(u,v)(j+2,1)=0  iff  [j]T+q_(u,v)=O.
```

Thus the formal aggregate

```text
F_R(j)=product_(u in D_12,v in D_34) W_(u,v)(j+2,1)
```

has the desired zero support on that chart. There are
`n^2=Theta(B^4)` factors.

## Gauge boundary

Equivalent elliptic nets have

```text
W'(z)=k*f(z)*W(z),
```

where `k` and every value of the quadratic function `f` are nonzero.
Therefore each relation zero and the zero set of `F_R(j)` are unchanged.
Interpolating the `F_R(j)` values at public occurrence labels and taking a
gcd with the squarefree fifth-label polynomial also preserves the same factor.

Raw nonzero values and raw dyadic interval products are not gauge invariant:
they acquire the product of all component gauge units. Only their zero ideal,
zero/nonzero decision, or a separately rigidified ratio is invariant. A route
that uses exact nonzero values must construct and charge a normalization. That
normalization depends on the same mixed net seeds described below.

## Complete-chart boundary

The standard rank-two net attached to `(T,Q)` assumes nonzero points with no
equal or inverse pair. P1553 cannot assume this for every `Q_(u,v)`.
Identity, `Q=+/-T`, tangent, vertical, infinity, repeated-endpoint, and
nonreduced components require saturated projective branches.

Clearing the factors `x(T)-x(Q)` without saturation can create false zeros.
Branching directly on those exceptional values restores exact semantics, but
the branch masks are themselves fixed-target pair-pair incidence queries.
The standard exact evaluation of one such branch has the same
`B^(2+o(1))` fixed-label online cost. The generic nondegenerate control below
already fails the cap, so exceptional handling cannot repair the exponent.

## Mixed-seed obstruction

A normalized rank-two net is generated from a bounded initial region, but that
region contains mixed terms depending on `Q_(u,v)`; normalization supplies
only selected values such as `W(e_i)` and `W(e_i+e_j)`.

There are two standard ways to construct the required mixed state.

### Componentwise seeds

Compute a constant-size seed block for every `Q_(u,v)` from its coordinates.
Each seed costs `B^o(1)`, but there are `Theta(B^4)` target-dependent
pair-pair points:

```text
online work       B^(4+o(1)),
represented state B^(4+o(1)).
```

This also materializes the target-shifted composed pair deck forbidden by R4.

### Aggregate seed products

Keep only products of one seed function over all `(u,v)`. The raw product
need not equal R4's key-difference resultant; its net normalization can carry
additional rational units and may be more expensive to represent. Grant the
cheapest audited gauge-invariant surrogate with the same zero support:
`r_R(t_j)`, the fixed-label translated pair-divisor resultant. The standard
exact degree-`n` pair-polynomial route already costs

```text
one mixed aggregate seed:
online work       B^(2+o(1)),
workspace         B^(2+o(1)).
```

Its zero/nonzero decision is the same fixed-fifth-label Query2P1 problem.
The target is fresh, so this seed cannot be stored in preprocessing. Even if a
hypothetical recurrence generated all remaining labels in `B^(1+o(1))`, the
mixed initialization dominates and violates the `B^(5/4+o(1))` online cap.

Calling the aggregate seed an oracle, a supplied recurrence term, a
precomputed target orbit, or target advice does not construct it.

## Failure of standard product closure

For each pair-pair point, a net recurrence has three summands

```text
A_q+B_q+C_q=0.
```

The desired aggregate diagonal products are

```text
A=product_q A_q,   B=product_q B_q,   C=product_q C_q.
```

The component identities do not imply `A+B+C=0`. Multiplying the component
identities produces all mixed choices

```text
sum_(sigma in {A,B,C}^Q) product_q sigma(q)=0,
```

not a three-term recurrence for the diagonal products. Retaining the individual
states costs `B^4`; retaining all mixed contractions is worse. A resultant
can compress one supplied aggregate value, but its standard construction is
the `B^2` mixed-seed route above.

This closes only the claim that standard elliptic-net closure survives
componentwise multiplication. It does not rule out a new proved aggregate
recurrence with an independently constructed compact state.

## Dyadic restriction and no-relation control

For a fifth-label interval `I`, an exact decision may use

```text
U_R(I)=product_(j in I) F_R(j).
```

Its zero status is gauge invariant and supports bisection. Its raw unit value is
not. In a no-relation query every factor is a unit, so no early zero or dynamic
split exists. Every negative dyadic branch must therefore be constructed or
replayed. A fast product formula for one already supplied EDS or net does not
aggregate the `B^4` mixed nets and does not construct the first mixed seed.

Repeated pair endpoints only repeat factors. They change intersection length
and row multiplicity but not the squarefree fifth-label support. Labelled
backpointers and final singleton verification remain necessary.

## Cost ledger

All exponents are in `B`, suppressing polylogarithmic and word-size factors.

| Route | Setup | One fresh target | Workspace | Result |
|---|---:|---:|---:|---|
| Two source-labelled pair trees | 2 | 0 | 2 | admitted preprocessing |
| Individual mixed net seeds | 2 | 4 | 4 | exact but expanded |
| One standard aggregate seed resultant | 2 | 2 | 2 | fixed-label decision only |
| B component resultants | 2 | 3 | 2 | exact fifth-label scan |
| Supplied aggregate recurrence | 2 | 1 after advice | 1 | advice includes missing mixed seed and closure |
| Standard dyadic replay | 2 | at least 2 per fresh mixed seed | 2 | no-relation branch dominates |

The required row is

```text
one fresh target and all replay <= B^(5/4+o(1)).
```

No reviewed standard net or division-polynomial route meets it.

## Full ECDLP path

The earlier conditional path remains only conditional:

```text
pair-index setup               B^2,
B relation targets             B^(9/4),
sparse factor-log algebra      B^2,
one masked descent             B^(5/4),
lambda                         0.45,
mu                             0.40.
```

This still assumes a subset-stable constructor, constant verified relation and
target success, collision and deck-rebuild accounting, `Theta(B)` independent
rows, factor-log completion, identical scalar-blind descent, and bit costs.
The standard net routes supply none of those missing claims.

## Controls

1. Favorable generic control: all `Q_(u,v)` are nonzero, unequal to
   `+/-T`, and one pair-pair point has one relation label. The net zero
   biconditional holds, but mixed initialization remains above the cap.
2. No-relation control: all aggregate components are units. No early split or
   zero-guided shortcut is available.
3. Gauge control: rescale every component net by an independent nonzero
   quadratic gauge. The gcd support is fixed while every raw unit product
   changes.
4. Exceptional control: include identity, `+/-T`, tangent, vertical,
   infinity, repeated, and nonreduced components. Unsaturated affine net
   formulas are rejected; direct branches restore semantics but not cost.
5. Fresh-target control: replace `R` after preprocessing. Every mixed seed,
   normalization, target orbit, or aggregate resultant must be rebuilt.

## Operation-level deduplication

- P1540 already owns direct elliptic-net recurrence, translated-coordinate
  linear complexity, QRT/Lax state, Fourier, and EDS index-location controls.
- P1513/IDEA-121 owns translated pair products and nonlinear common-factor
  location.
- P1551 owns represented endpoint-coefficient access.
- P1516 owns the two pair indexes with the missing target-local router.
- P1553 R4 owns the exact `z_R` component semantics and represented
  resultant/remainder route ledger.

The aggregate net restatement creates no new mechanism owner or P1554.

## Disposition

```text
REVISE_SCOPED_NEGATIVE_STANDARD_ELLIPTIC_NET_AND_DIVISION_POLYNOMIAL_AGGREGATION__SINGLE_COMPONENT_RELATION_ZERO_EXACT_ON_NONDEGENERATE_CHART__GCD_SUPPORT_GAUGE_INVARIANT__RAW_UNIT_PRODUCTS_GAUGE_DEPENDENT__EXCEPTIONAL_NET_CHARTS_REQUIRE_SATURATION__NORMALIZED_VALUES_DO_NOT_SUPPLY_MIXED_SEEDS__INDIVIDUAL_MIXED_STATE_B4__ONE_STANDARD_AGGREGATE_SEED_B2_AND_ALREADY_FIXED_LABEL_QUERY2P1__COMPONENTWISE_NONLINEAR_RECURRENCES_NOT_CLOSED_UNDER_DIAGONAL_PRODUCT__NO_RELATION_REPLAY_UNPAID__P1540_P1513_P1551_P1516_MERGE__UNKNOWN_AGGREGATE_RECURRENCE_CIRCUIT_NOT_RULED_OUT__NO_P1554__NO_RUN__NO_LOWER_BOUND__NO_BREAKTHROUGH
```

## Exactly one next action

Rerank outside standard elliptic-net, division-polynomial, EDS, QRT/Lax, and
supplied-recurrence families. Under existing P1553/P1513/P1551/P1516 ownership,
admit only one theorem-only, mechanism-distinct compiler that constructs both
the target-fresh mixed aggregate state and its restriction-stable zero decoder
inside total `B^(5/4+o(1))), with complete charts and replay; otherwise
preserve the specialized Query2P1 exception unchanged.

## Primary controls

- Stange, *Elliptic Nets and Elliptic Curves*,
  <https://arxiv.org/abs/0710.1316>.
- Lauter and Stange, *The elliptic curve discrete logarithm problem and
  equivalent hard problems for elliptic divisibility sequences*,
  <https://arxiv.org/abs/0803.0728>.
- Bostan, Gaudry, and Schost, *Linear Recurrences with Polynomial
  Coefficients*, <https://doi.org/10.1137/S0097539704443793>.

These sources define individual net recurrences, scale equivalence, zero
apparition, EDS index location, and supplied recurrence evaluation. They do not
supply the aggregate mixed-seed compiler required here.
