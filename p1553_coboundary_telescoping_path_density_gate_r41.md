# P1553 coboundary telescoping and path-density gate R41

## Classification

- Owner: existing P1553/IDEA-195 implicit target and orbit-transfer frontier;
  no new idea ID.
- Evidence: exact divisor identity, exact toy cancellation audit, and scoped
  path-level support/work theorem; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_POSITIVE_TELESCOPE_NEGATIVE_PATH_DENSITY_GATE`.
- Labels: `theorem-only`, `toy`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: the R40 coboundary gives an exact constant-size
  endpoint quotient for a consecutive product. This is a real positive
  identity. When its subgroup poles overlap selected-zero translates, however,
  rational reduction cancels genuine hit factors and loses their source marks.
  Even granting separated poles and unit-cost endpoint evaluation, processing
  `K` path quotients explicitly costs expected `Omega(B^3)` work per
  scalar-blind hit after their target support is charged. A single global,
  cancellation-free, marked across-path locator remains open. No Shoup-bound
  improvement or ECDLP breakthrough follows.

R40 asked whether a partial orbit product has an addition law beyond a generic
product tree. For a coboundary, it does: the product telescopes. R41 corrects
the tempting conclusion that this solves the marked target query. The same
zero/pole cancellation that makes the quotient short may erase the existence
bit, and treating each pole-separated path separately restores the R38
support/work cancellation one level higher.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R36 source/target-support gate | `103670908f33baee0915372434899a3a710861f501e19c91599f4221fe3f2aa5` |
| R37 generic path reduction gate | `b8a9423edef4041fcc78ec9f4808ab63ab2f97aa4fc1b157afaedabadb87c772` |
| R38 target-walk support/work gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R39 implicit norm/coboundary gate | `2901305c2ac83fe51d2f23957389cb2dfb6d58b3c4bc0d2e439ad4ef3f6085f6` |
| R40 orbit-transfer Miller/net gate | `7aebb68ba410c223250fd907cd8891a9539553c202d16bcff3d0a79e7e35a07b` |
| R40 bundle hash list | `a91e942c9bdd2ba75981c1bad49afc25d0ac368ca97309bae8a4ab10de87ceb8` |
| R40 staging receipt | `3657a5e86b27d2cd8edbe725f3a2ea0d94911ebbec88ff8fc50a63c6e3849500` |

## Frozen coboundary interface

Let `q` have prime order `N=B^(5+o(1))`. Retain the R40 principal transfer
function and selected interval

```text
F(R)=h(R+q)/h(R),
Z={R_0+[i]q:0<=i<L},
L=B^2.                                           (1)
```

On the subgroup, `F` has simple zero set `Z` and a pole set `P` of the same
total degree, disjoint from `Z`. For a path in the other endpoint deck, write

```text
V_c={c+[j]q:0<=j<M}.                             (2)
```

The target query needs to decide and mark

```text
exists v in V_c, z in Z: R=v+z.                 (3)
```

It must remain defined when another factor is a pole and must return the path
position `j`, the endpoint `z`, and the R36 source mark.

## Theorem 1: every consecutive coboundary product telescopes

With the sign convention in (2),

```text
product_(j=0)^(M-1) F(R-c-[j]q)
  =h(R-c+q)/h(R-c-(M-1)q).                      (4)
```

All interior `h` factors cancel. Thus a product represented only as a reduced
rational function has a constant-size endpoint description, independent of
`M`. This is the positive partial-orbit identity requested by R40. No sigma,
net, or determinant machinery is needed.

Equation (4) is an identity of rational functions. It does not state that the
reduced quotient is a total zero-existence predicate for the unreduced factors.

## Theorem 2: telescoping may cancel genuine selected hits

Let

```text
a_i=ord_(R_0+[i]q)(F)=d_(i+1)-d_i.              (5)
```

At target coordinate `r`, the unreduced path product contains a selected zero
iff

```text
exists 0<=j<M: a_(r-c-j)>0.                     (6)
```

It contains a pole factor iff the same condition holds with `<0`. Its reduced
order is only the net valuation

```text
sum_(j=0)^(M-1) a_(r-c-j)
  =d_(r-c+1)-d_(r-c-M+1).                       (7)
```

If both zero and pole factors occur, (7) cancels their multiplicities. A raw
hit can become a nonzero finite value or a pole, and the factor position is
lost. Therefore a reduced endpoint quotient is a correct existence predicate
only after a disjointness condition such as

```text
(V_c+Z) intersect (V_c+P)=empty                 (8)
```

is proved.

Keeping an unreduced projective numerator/denominator pair avoids a false
field division, but the two components then retain the common interior factors.
Their simultaneous zero is precisely the information removed by (4). A
separate flag saying that some numerator factor vanished is the original
interval locator, not a consequence of the endpoint quotient.

This is a semantic obstruction to the reduced quotient, not a lower bound on a
cancellation-free circuit.

## Theorem 3: exact R40 toy cancellation audit

The script `p1553_coboundary_telescoping_selfcheck_r41.py` replays the exact R40
minimum transfer divisor, forms every path product of lengths one through nine,
and verifies (7) at all 103 target coordinates.

For the length-nine other-pair path, the unreduced zero factors cover

```text
V+Z={0,1,...,16},      17 target coordinates.   (9)
```

The raw pole-factor support has 28 coordinates. Exact statistics are

```text
raw zero support                         17,
raw pole support                         28,
zero/pole overlap                        16,
unambiguous raw zeros                     1,
reduced positive-order support            9,
reduced negative-order support           18,
raw hits not positive after reduction     8.     (10)
```

The eight lost hit coordinates are

```text
(0,1,2,3,13,14,15,16).                         (11)
```

The reduced quotient keeps positive order only on

```text
(4,5,6,7,8,9,10,11,12).                        (12)
```

This is an exact positive telescope and exact negative source-semantics
control. It is toy-only: another pole divisor may satisfy (8).

## Theorem 4: separated-pole path quotients still cost B^3 per hit

Grant the strongest favorable assumptions:

1. the poles are placed so that every used path satisfies (8);
2. evaluating both endpoint values of `h` costs `O(1)`;
3. a positive path can be unranked by logarithmic subdivision;
4. all paths and target masks are exact and source-marked.

Write the complete other-pair support as a union of `K` `q`-paths,

```text
V=union_(k=1)^K V_(c_k),
sum_k M_k=|V|<=B^2.                              (13)
```

Adding the length-`L` selected interval to one length-`M_k` path gives at most
`L+M_k-1` targets. Hence

```text
|V+Z|<=sum_k (L+M_k-1)
     =K*L+|V|-K.                                 (14)
```

Processing every endpoint quotient (4) explicitly costs `Omega(K)`. For a
uniform scalar-blind target, success is at most `(K*L+|V|)/N`. Since
`L=B^2`, `|V|<=B^2`, and `K>=1`, expected work for one hit is at least

```text
K*N/(K*L+|V|)
  >=N/(2L)
  =Omega(B^3).                                   (15)
```

Overlaps between path sumsets only reduce support and make (15) stronger. Any
nonconstant endpoint evaluation cost multiplies the bound. Thus perfect
telescoping, perfect pole separation, and perfect within-path unranking still
miss rho before R10, relation rank, factor logs, or descent.

Equation (15) is the R38 support/work cancellation at path granularity. It is
scoped to algorithms that process the `K` path results explicitly. A single
global operation over all path starts lies outside the theorem.

## The remaining global operation

To evade (15), a candidate must combine

```text
{h(R-c_k+q), h(R-c_k-(M_k-1)q):1<=k<=K}         (16)
```

without reading `K` endpoint pairs, while preserving uncancelled zero
existence and returning one path and one position. This is another marked
translated-product/common-factor locator, now over path starts rather than
individual endpoints. A product value or reduced rational quotient is
insufficient.

R37 already shows that an arbitrary path-only marked index at the required
setup/query rectangle would beat the generic Shoup boundary. R41 does not yet
prove that the degree-`B` pencil forces arbitrary R37 instances into (16), so
the coordinate-constrained global operation remains logically open.

## Cost and ECDLP gate

No current survivor supplies all of:

```text
degree-B asymptotic low-boundary pencil,
global cancellation-free across-path locator,
setup <=B^(9/4+o(1)),
average density-adjusted query <B^(1/2-o(1)),
marked path and within-path source output,
R10 coefficients, independent rank, factor logs, blind descent. (17)
```

The exact telescope is therefore a useful algebraic identity but not a target
router or relation generator.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_POSITIVE_TELESCOPE_NEGATIVE_PATH_DENSITY_GATE
COBOUNDARY_CONSECUTIVE_PRODUCT_TELESCOPES_EXACTLY
R40_TOY_RAW_HITS17_REDUCED_ZEROS9_LOST8
ZERO_POLE_OVERLAP_ERASES_EXISTENCE_AND_SOURCE
CANCELLATION_FREE_PAIR_RETAINS_INTERIOR_COMMON_FACTOR
EVEN_PERFECT_POLE_SEPARATION_AND_UNIT_ENDPOINT_COST_GIVE_B3_PER_HIT
EXPLICIT_K_PATH_RESULTS_CLOSED
GLOBAL_CANCELLATION_FREE_MARKED_ACROSS_PATH_LOCATOR_OPEN
NO_DEGREE_B_FAMILY_TARGET_R10_RANK_LOGS_OR_DESCENT
NO_SHOUP_CLAIM
NO_BREAKTHROUGH
```

Exactly one next action: reduce or separate the global across-path operation.
Instantiate (16) on R37's mixed-radix generic-DLP decks and prove that any
subset-stable, cancellation-free marked locator with setup `B^(9/4)` and
average query below `B^(1/2)` recovers the hidden scalar; alternatively exhibit
a pencil identity that fails on the R37 controls but combines all path starts
in sublinear work. Any survivor must return a path, a position, and the R36
source mark before binding to R10.
