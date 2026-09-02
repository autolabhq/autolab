# P1553 target-walk support/work gate R38

## Classification

- Owner: existing P1553/IDEA-195 pencil, target-density, and source-router
  frontier; no new idea ID.
- Evidence: exact support/work counting theorem and exact degree-three elliptic
  pencil toy search; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_POSITIVE_TOY_NEGATIVE_ROUTE_GATE`.
- Labels: `theorem-only`, `toy`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: a genuine elliptic pencil can have a one-path selected
  union with boundary one, so pencil geometry alone does not force the R37
  generic boundary. However, every explicit target walk that updates
  translated intersections by scanning entering/leaving points pays expected
  `Omega(B^3)` work per scalar-blind hit after its correlated batch-success
  probability is charged. No Shoup-bound improvement or ECDLP breakthrough
  follows.

R37 left the common degree-`B` pencil as the only admitted escape from the
generic mixed-radix reduction. The first candidate operation is natural:
choose one selected shift `q_0` with known scalar, issue a long arithmetic
walk of known-log targets, initialize `V intersect (R-Z)` once, and update the
intersection only at the path boundaries. If the boundary has size
`d_0<B^(1/2)`, one batch appears to cost less than rho.

R38 gives both sides of that proposal. A deterministic search finds a real
degree-three line pencil whose three complete fibers are nine consecutive
points under a known-log subgroup shift. Its selected union has boundary one.
The hoped-for algebraic lower bound on `d_0` is therefore false at toy scale.

But small boundary makes consecutive target hits cluster. A walk that touches
only `W` translated endpoint positions can be positive for at most
`|V|*W` base masks. Its exact work and its random-mask success probability
cancel, leaving `N/|V|=B^(3+o(1))` expected work. The cancellation applies to
arbitrary multidirectional target walks, not only one interval, as long as the
algorithm explicitly scans every changed translated endpoint.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R28 common-subdivisor gate | `c24183bfe4823a9318ae06769763c94cfed99cb4291807f06b85f397fd53f662` |
| R31 near-period run compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R35 shift-slice support/density gate | `e72c406ee3997eaac86b44d757a34793d3678a129331ef7be17aa09ae05ee0de` |
| R36 source/target-support gate | `103670908f33baee0915372434899a3a710861f501e19c91599f4221fe3f2aa5` |
| R37 generic path reduction gate | `b8a9423edef4041fcc78ec9f4808ab63ab2f97aa4fc1b157afaedabadb87c772` |
| R37 bundle hash list | `a4541b34c8a221cd0a1325f2e97cb1d809b01f3a46c8ff2ca4d7d59c9cb7ff58` |
| R37 staging receipt | `a8f3d76bd257b703b5baa121cfeac7271459495cd667f4a321e7326c44977f2a` |

## Frozen marked-endpoint interface

Let `G` have odd prime order

```text
N=B^(5+o(1)).                                     (1)
```

Let `V` be the signed endpoint support of the other pair deck and let `Z_*`
be a selected target-visible endpoint set with exact source marks:

```text
|V|<=B^2,
|Z_*|=Theta(B^2).                                 (2)
```

A mark on `z in Z_*` contains one complete first-pair/fifth occurrence whose
signed sum is `z`; a mark on `v in V` contains one complete second-pair
occurrence. Thus a hit

```text
v+z=R                                             (3)
```

immediately returns one verified five-list source. R9 permits this direct
empty-or-source interface; exact multiplicity is not assumed.

R35 supplies the marks conditionally from one dense shift slice. If

```text
H_q=Z intersect (Z-q) intersect (Z+q)
```

has `Theta(B^2)` points, then

```text
Z_q={P+q:P in H_q}                                (4)
```

has the same cardinality and every point keeps the source `(P,q)`. On an R31
path, (4) trims only endpoints, so a dense minimum-boundary shift preserves
the path representation up to constant boundary factors.

## Theorem 1: arbitrary explicit target walks have support/work cancellation

Choose a finite ordered set of public known-log offsets

```text
T=(t_0,t_1,...,t_(m-1)) subset G.                 (5)
```

For a uniformly randomized known-log base mask `R_0`, the batch queries are

```text
R_j=R_0+t_j.                                      (6)
```

Put

```text
A_j=t_j-Z_*,
D=union_j A_j=T-Z_*.                              (7)
```

The batch is positive only if

```text
R_0 in V+Z_*-T=V-D.                               (8)
```

Therefore its success probability obeys the exact support bound

```text
Pr[positive batch]
  <=min(1,|V|*|D|/N).                             (9)
```

Consider the explicit boundary-update schedule that scans all of `A_0`, then
for every transition scans the entering and leaving points in the symmetric
difference. Charge

```text
W=|Z_*|+sum_(j=1)^(m-1) |A_j symmetric_difference A_(j-1)|. (10)
```

Every point in the union (7) is either in `A_0` or first enters in one of the
symmetric differences, so

```text
|D|<=W.                                           (11)
```

Hashing `V` with occurrence backpointers makes the exact batch work
`Theta(W)` group additions and lookups. Combining (9)-(11), either
`W>=N/|V|` already, or independent random base masks need at least
`N/(|V|W)` batches in expectation. In both cases the expected work for one
scalar-blind hit is

```text
Omega(N/|V|)=B^(3-o(1)).                          (12)
```

The result includes all-negative batches and remains valid if one positive
batch returns many marked positions. One target descent needs at least one
positive batch, so (12) closes the complete ECDLP path before a relation-rank
claim can help.

This is a lower bound for the named explicit boundary-update schedule. It is
not a cell-probe, arithmetic-circuit, generic-coordinate, or elliptic-function
lower bound. A compressed algebraic test may represent `D` without scanning
its newly entered points and lies outside (10).

## Corollary 2: a one-shift interval has no favorable boundary exponent

Let

```text
t_j=[j]q_0,       0<=j<L,
d_0=|Z_*|-|Z_* intersect (Z_*-q_0)|.              (13)
```

Translation is bijective, so consecutive sets in (7) have symmetric
difference `2d_0`. More sharply, their union grows by at most `d_0` per step:

```text
|D|<=|Z_*|+(L-1)d_0,
W=|Z_*|+2(L-1)d_0.                                (14)
```

At the balanced scale `|Z_*|=B^2`, `|V|=B^2`, equation (9) gives

```text
Pr[positive batch]
 <=O((B^2+L*d_0)/B^3).                            (15)
```

The explicit batch work is `Theta(B^2+L*d_0)`. Repeating independently
masked batches therefore costs

```text
Omega(B^3)                                        (16)
```

for every `L` and `d_0`, up to constants and saturation at probability one.
In particular:

```text
d_0=1, L=B^2:       batch work B^2, success at most B^-1,
d_0=B, L=B^2:       batch work B^3, success at most constant. (17)
```

The first line is cheap but rarely aligned; the second is R37's generic
mixed-radix boundary. There is no intermediate exponent win for this update
schedule.

## Theorem 3: a genuine degree-three pencil has boundary one

The deterministic search
`p1553_low_boundary_pencil_search_r38.py` works over

```text
E/F_193: y^2=x^3+2x+3,
P=(1,44),
order(P)=103.                                     (18)
```

It searches every nonzero subgroup step and every length-nine path avoiding
the identity. For every partition into three triples, it requires equal Abel
sums, translates by the inverse of three into the `|3O|` plane embedding,
tests that each triple is a full line section, and tests that the three lines
are concurrent. Concurrency makes them fibers of one degree-three line
pencil, not merely three linearly equivalent divisors.

After 676,090 tested partitions, the first exact witness has

```text
known step scalar q_0=38,
path scalars=(1,39,77,12,50,88,23,61,99),
common fiber sum=47,
embedding translation=53.                        (19)
```

Its three full fiber blocks are

```text
(1,88,61),
(39,12,99),
(77,50,23),                                       (20)
```

and the concurrent line equations, in projective coefficient form, are

```text
(1,108,4),
(1,85,4),
(1,0,4),                                          (21)
```

with common base point `(1,0,48)`. Taking the first two lines as numerator
and denominator gives selected values

```text
0, infinity, 192.                                 (22)
```

The script evaluates the resulting rational map on all 103 subgroup points.
Each selected fiber is exactly its claimed three-point block, with no extra
subgroup preimage. The selected union is one `q_0` path, so

```text
d_0=1,
|Z intersect (Z-q_0) intersect (Z+q_0)|=7.        (23)
```

The report `p1553_low_boundary_pencil_search_report_r38.json` is reproduced
byte-for-byte. It proves that neither equal Abel sums, one common pencil, full
fibers, a known-log shift, nor complete plus/minus containment forces
`d_0=Omega(B^(1/2))` at degree three.

This is a toy existence control only. It does not give an asymptotic family,
a pair-deck construction with unknown logs, target rank, or an implicit
batch test. The subgroup order `103` is not at the asymptotic `B^5` scale for
`B=3`.

## Theorem 4: endpoint marks do not remove the cancellation

For the one-path toy, the dense shift slice marks the seven interior-plus
endpoints in (4), themselves one path with boundary one. More generally, if
R35 supplies `Theta(B^2)` marked endpoints by trimming `O(d_0)` positions per
R31 path, substitute that marked set for `Z_*` in Theorem 1. The exact source
backpointer makes every hit useful, but it does not enlarge `D` or the base-
mask support in (8).

Thus the positive R35/R36 source section and the positive small-boundary
pencil solve source replay, not fresh-target coverage. Multiplicity above one
marked endpoint remains irrelevant to target-support cardinality.

## Harness obligations and controls

1. The target base mask is randomized independently of setup; all offsets in
   a batch may be deterministic and known-log.
2. Every repeated batch uses a fresh independent base mask and charges its
   initialization and all negative updates.
3. `V` and `Z_*` retain occurrence backpointers, but support cardinality in
   (9) ignores multiplicity correctly.
4. A target walk in several known directions is covered by Theorem 1 through
   its actual ordered symmetric differences.
5. The theorem does not charge a hypothetical implicit norm, resultant,
   sketch, or coordinate circuit by `W` unless it enumerates the changed
   points.
6. The toy search verifies all selected subgroup fibers, concurrency, path
   boundary, and complete `+q_0/-q_0` domain exactly.
7. The toy finder uses scalar labels to search; it is not an oracle-free
   asymptotic constructor and does not hide that advice.
8. No fresh-target implicit index, R10 queried coefficients, independent
   rank, factor-base logs, scalar-blind descent, unrestricted lower bound,
   Shoup improvement, or breakthrough is supplied.

## Operation-level deduplication

- R28 owns residual-divisor certificates for near-full overlaps.
- R31 owns explicit path and boundary construction.
- R35-R36 own endpoint marks, source replay, and target-density accounting.
- R37 owns the path-only generic DLP boundary.
- R38 adds an actual low-boundary pencil witness and the arbitrary target-walk
  support/work cancellation for explicit boundary updates.

## Result boundary

```text
degree-three complete-fiber one-path pencil: exact toy pass
known-log selected shift: present in toy
toy boundary d_0: 1
toy complete plus/minus domain: 7 of 9 points
universal pencil boundary floor B^(1/2): false at toy scale
explicit target-walk batch work: W
batch base-mask support: at most |V|*W
expected scalar-blind hit work: at least N/|V|=B^3
implicit algebraic batch operation: absent and outside theorem
asymptotic low-boundary pencil family: absent
fresh target, R10, rank, logs, descent: absent
Shoup-bound improvement: false
breakthrough: false
```

## Exactly one next action

Test the only remaining escape from Theorem 1: an implicit pencil-level batch
operation whose work is `o(|T-Z_*|)` and which returns one marked
`V intersect (R_0+T-Z_*)` witness without scanning translated support. Build
or refute a saturated norm/resultant or low-degree recurrence from
`H_C(g)`, the R34 addition surface, and the R38 line-pencil control under
setup `B^(9/4+o(1))`, total masked-descent work below `B^(5/2-o(1))`, exact
source replay, and then R10/rank/log/descent accounting.
