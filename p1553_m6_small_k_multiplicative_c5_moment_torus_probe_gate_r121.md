# P1553 M6 small-k multiplicative C5 moment/torus gate R121

## Claim boundary

R121 supplies exact norm-one torus, complete-homogeneous moment,
Berlekamp-Massey, annihilator, and C2|C3 source controls for the R120
embedding-degree-two fixtures. It closes only the named represented
moment, split, and bound subfunction-inversion routes.

It does not supply an inside-cap multiplicative C5 index, known-RHS rank,
factor logarithms, identical target descent, a Pollard-rho improvement, a
Shoup-bound improvement, or a generic-prime ECDLP breakthrough.

Classification:

```text
PAIRING_IMAGES_ON_K2_CONTROLS_HAVE_COMPLETE_NORM_ONE_CAYLEY_CHART__FIVE_PRODUCT_IS_EXACT_DEGREE5_SYMMETRIC_TORUS_FORM__ENDPOINT_MOMENTS_ARE_COMPLETE_HOMOGENEOUS_DECK_MOMENTS__BM_ORDER_EQUALS_INJECTIVE_C5_SUPPORT__FULL_MOMENT_AND_ANNIHILATOR_STATE_B15O4__C2C3_SPLIT_SETUP_B9O4_QUERY_B3O2__DINUR_GOLOVNEV_DIRECT_K6_AND_THEOREM4P1_POLYLOG_ROUTES_OVER_CAP_OR_INAPPLICABLE__TARGET_SPECIALIZED_NONLINEAR_TORUS_C5_CIRCUIT_OPEN__NO_RANK_LOGS_DESCENT_SHOUP_BREAKTHROUGH
```

## Exact torus representation

On every R82 pairing control,

```text
Fp2 = Fp[u]/(u^2-d),  d=-3,
q | p+1,
mu_q is contained in {z : z^(p+1)=1}.
```

For an odd q, `-1` is not in `mu_q`. The Cayley chart

```text
z(t) = (1+u*t)/(1-u*t)
t    = b/(a+1), for z=a+b*u
```

therefore covers the entire q-order pairing image. Multiplication is the
rational group law

```text
t1 (+) t2 = (t1+t2)/(1+d*t1*t2).
```

For five parameters, let `e_i` be their elementary symmetric functions and
put

```text
A = e0 + d*e2 + d^2*e4
B = e1 + d*e3 + d^2*e5.
```

Then

```text
product_i z(t_i) = z(t_target)
```

is exactly the degree-five multilinear equation

```text
B - t_target*A = 0.
```

All eight controls verify the chart roundtrip, binary law, nonzero
denominator, and five-product equation for every canonical C5 source. No
pairing-image discrete logarithm or candidate scalar label is consumed.

This is a compact predicate description, not a compact index over all deck
tuples.

## Complete-homogeneous moments

For pairing images `z_1,...,z_n`, the m-th power sum of the canonical
five-product multiset is

```text
S_m = sum_(i1<=...<=i5) (z_i1*...*z_i5)^m
    = h_5(z_1^m,...,z_n^m).
```

Because every control has characteristic greater than five, Newton's
recurrence

```text
r*h_r = sum_(j=1..r) p_j*h_(r-j),
p_j   = sum_i z_i^(j*m)
```

constructs each `S_m` from five deck power sums.

The producer computes `2M` direct and compact moments for

```text
M = binom(n+4,5)
```

on each control. Their hashes match exactly. Since all `M` products are
distinct, the Hankel matrix is a Vandermonde factorization with nonzero
weights. Berlekamp-Massey therefore has exact order `M`; the computed
connection polynomial equals the full product annihilator

```text
product_(y in C5) (X-y).
```

Every positive product is a root and the inherited empty target is not.

## Cost boundary

The R119 iid support theorem gives

```text
M = B^(15/4+o(1))
```

with high probability in its stated random-deck model. Thus a full moment
prefix, exact recurrence, or endpoint annihilator has state or output
exponent `15/4`, above the `B^(9/4)` setup cap. Computing each represented
moment directly from the `B^(3/4)` deck gives naive total work exponent
`9/2`.

The torus equation has constant degree, but evaluating or tabulating it on
the complete deck grid again exposes `B^(15/4)` tuples. No arbitrary
arithmetic-circuit lower bound follows.

The exact multiplicative C2|C3 split has:

```text
C3 setup = B^(9/4),
C2 query = B^(3/2).
```

It meets setup exactly, returns five projective occurrence backpointers, and
rejects the empty control, but misses the required polylogarithmic arbitrary
target query.

## Subfunction-inversion control

Dinur and Golovnev's direct balanced `kSUM` theorem at `k=6`, zero
polynomial query slack, gives

```text
S = B^(33/8+o(1)).
```

Their stated `kSUM` construction uses integer addition and residue maps;
their separate transfer is to XOR. Applying those maps to the torus product
would require a new torus-native decomposition or pairing-image discrete
logarithms.

The general Theorem 4.1 also assumes an output universe softly linear in the
function domain. Here the candidate domain has exponent `15/4`, while the
ambient arbitrary target universe `mu_q` has exponent `5`, so the direct
application is outside the theorem.

Even granting an admissible image-range recoding, the theorem's Items 4-5
can cover at most `D*L` injective image points for one sampled
decomposition. Expected `5/6` coverage forces

```text
D*L >= 5M/6.
```

At `delta=0`, its space term is softly `D*L^(3/2)`, hence at least
`Omega(M)=B^(15/4)`. This is a scoped obstruction for that theorem template,
not a lower bound for all torus data structures.

## Finite receipts

All eight R82 controls verify:

- every pairing image is norm one and lies in the finite Cayley chart;
- every binary torus law and canonical degree-five target form;
- direct and complete-homogeneous moments through order `2M-1`;
- Berlekamp-Massey order exactly equal to the distinct C5 support;
- equality of the recurrence and full endpoint annihilator;
- positive-root and empty-target annihilator evaluations;
- exact C2|C3 source recovery and projective replay;
- exact empty-query rejection.

The deck sizes are 3, 5, 6, and 7 at two offsets, with canonical C5 support
sizes 21, 126, 252, and 462. Finite enumeration receives no asymptotic
credit.

## Admission

Thirteen of twenty obligations pass. The inside-cap polylogarithmic
membership/source circuit, known-RHS rank, factor logs, identical descent,
Shoup improvement, and breakthrough obligations remain false.

Disposition:

```text
ADMIT_EXACT_NORM_ONE_TORUS_COMPLETE_HOMOGENEOUS_MOMENT_BM_AND_SPLIT_SOURCE_CONTROLS_ONLY__REJECT_FULL_MOMENT_ANNIHILATOR_TORUS_GRID_C2C3_QUERY_DIRECT_KSUM_THEOREM4P1_AND_FIELD_DLP_ROUTES_AT_FROZEN_CAPS__PRESERVE_TARGET_SPECIALIZED_NONLINEAR_TORUS_C5_CIRCUIT_AND_PAIRING_UNFRIENDLY_GAP__NO_LOCATOR__NO_RANK__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP__NO_BREAKTHROUGH
```

## Primary reference

- Dinur and Golovnev, *Improved Time-Space Tradeoffs for
  3SUM-Indexing*, <https://arxiv.org/abs/2512.04258v2>.

## Exactly one next action

Construct or refute one target-specialized nonlinear torus C5
membership/source circuit outside the full moment/Newton/Hankel, explicit
C2|C3 split, integer-residue kSUM, and Theorem 4.1 subfunction grammars. It
may inject the target into `B-t*A` before expansion, but must fit
`B^(9/4+o(1))` state and polylogarithmic arbitrary-target work, avoid field
discrete logarithms, return exact empty certificates and five projective
occurrence backpointers, and charge the complete pairing-to-descent path.
