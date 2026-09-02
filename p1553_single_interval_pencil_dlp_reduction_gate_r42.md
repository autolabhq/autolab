# P1553 single-interval pencil DLP reduction gate R42

## Classification

- Owner: existing P1553/IDEA-195 global marked-locator and pencil frontier;
  no new idea ID.
- Evidence: exact generic-group reduction and exhaustive exact elliptic-pencil
  toy control; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_STRONG_GENERIC_BOUNDARY_POSITIVE_PENCIL_TOY_GATE`.
- Labels: `theorem-only`, `toy`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: a single selected interval of length `B^2`, one
  `B^2`-point giant-step pair deck, and only `B` known-log queries encode an
  arbitrary prime-order DLP. Under setup `B^(9/4)`, every generic
  group-operation implementation needs average query exponent at least `3/2`,
  far above R36's required exponent below `1/2`. The exact R38 degree-three
  pencil realizes the selected interval, and the reduction recovers all 103
  toy secrets with no bad marked recovery. This does not exclude a
  coordinate-level algorithm outside Shoup's generic model, and no asymptotic
  interval-pencil family, heavy complete-key marks, R10 output, rank, logs, or
  descent is supplied. No ECDLP breakthrough follows.

R41 left one global cancellation-free locator over all path starts. R42 shows
that the path-only version is not merely near the generic boundary: with one
selected interval it gives a stronger DLP digit decomposition than R37 and
needs only `B` target queries. It also removes a possible toy objection. The
R38 selected pencil union is exactly such an interval after changing to its
known path generator, so a common degree-three pencil does not prevent the
reduction at toy scale.

The remaining escape is now precise. A qualifying operation must use
asymptotic pencil or complete-key coordinate structure unavailable to the
generic reduction. If it does so and still passes the marked interface, it is
an actual coordinate-level candidate for beating the generic bound, not a
contradiction to Shoup.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R9 projector-trace router gate | `400f4a49d75948a188df281633c1b974d2aa70142e852c6a55de7a810e3edf81` |
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R31 near-period compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R36 source/target-support gate | `103670908f33baee0915372434899a3a710861f501e19c91599f4221fe3f2aa5` |
| R37 generic path reduction gate | `b8a9423edef4041fcc78ec9f4808ab63ab2f97aa4fc1b157afaedabadb87c772` |
| R38 target-walk and pencil gate | `ecacb63d18cc2e4478fa0d3c6b71930a4ba4e0a50f3866bd46752ed7c37f7aa5` |
| R40 orbit-transfer Miller/net gate | `7aebb68ba410c223250fd907cd8891a9539553c202d16bcff3d0a79e7e35a07b` |
| R41 coboundary path-density gate | `c256de45204dc0c71983899bc710e0a5c3b147d38ee61d82049bb0eaabe17850` |
| R41 bundle hash list | `c817d7f5652a2d85761889e2026abf0ac576690c4088a3b2f274a6f951bddda6` |
| R41 staging receipt | `a612e89dc5ebeec6f87520dadf94f1a4b17548e55aef4985850e04d965011ac1` |

## Frozen one-interval marked interface

Let `G=<P>` have odd prime order `N`, and set

```text
B=ceil(N^(1/5)),
B^3<N<=B^5                                      (1)
```

in the asymptotic regime. Let `Z_0` be any public group point; its discrete
logarithm is not required. The selected pair deck is one labelled interval

```text
A_1={Z_0-[iB]P:0<=i<B},
A_2={-[j]P:0<=j<B},
Z={Z_0-[r]P:0<=r<B^2}.                          (2)
```

The other pair deck may depend on the DLP target `Q`, as ordinary
preprocessing data:

```text
A_3={Q-Z_0-[iB^4]P:0<=i<B},
A_4={-[jB^3]P:0<=j<B},
V={Q-Z_0-[kB^3]P:0<=k<B^2}.                     (3)
```

Issue the `B` known-log queries

```text
R_h=[hB^2]P,       0<=h<B.                      (4)
```

An exact marked locator returns `(i,j,a,b,h)` satisfying

```text
(Q-Z_0-[(iB+j)B^3]P)
 +(Z_0-[(aB+b)]P)=R_h.                           (5)
```

Occurrence labels are mandatory. Duplicate support points or modular wraps
are permitted because every returned equality is replayed in `G`.

## Theorem 1: B queries recover an arbitrary discrete logarithm

Given

```text
Q=[x]P,       0<=x<N,                            (6)
```

write the mixed-radix expansion

```text
x=kB^3+hB^2+r,
0<=k<B^2, 0<=h<B, 0<=r<B^2,                     (7)
k=iB+j, r=aB+b.                                  (8)
```

The ranges follow from `x<B^5`. The labelled endpoints from (2)-(3) obey

```text
v=Q-Z_0-[kB^3]P=[hB^2+r]P-Z_0,
z=Z_0-[r]P,
v+z=[hB^2]P=R_h.                                (9)
```

Thus at least one query (4) is positive. Conversely, every returned marked
equality (5) gives

```text
x=((iB+j)B^3+hB^2+(aB+b)) mod N.                (10)
```

Hence any marked answer, not only the canonical one, recovers the exact DLP.
The offset `Z_0` cancels and never needs a scalar label.

The construction uses one selected interval, not `B` arbitrary paths. The
other pair deck is a valid `B` by `B` occurrence deck and can always be viewed
as `B^2` singleton `P`-paths. Any claimed global across-path operation that
accepts arbitrary path lengths must therefore accept this instance.

## Corollary 2: the generic average-query floor is B^(3/2)

Suppose a generic group-operation implementation preprocesses (2)-(3) in
`B^(s+o(1))` work and answers each query (4) in amortized
`B^(kappa+o(1))` work, including negative answers and marked recovery. The
four lists, pair occurrence labels, and query points cost at most
`B^(2+o(1))` to construct. The reduction solves DLP in

```text
B^(max(s,1+kappa,2)+o(1)).                       (11)
```

Shoup's prime-order generic lower bound therefore forces

```text
max(s,1+kappa,2)>=5/2-o(1).                      (12)
```

Under the P1553 setup cap `s<=9/4`,

```text
kappa>=3/2-o(1).                                 (13)
```

R36 needs `kappa<1/2-o(1)` after target density. The global one-interval
marked locator is therefore a full exponent beyond the generic boundary at
the required setup, even before source replay and R10.

Equations (11)-(13) apply only when the implementation's informative
operations are generic group operations, equality tests, path labels, and
generic memory accesses. Weierstrass coordinates, rational pencil values,
Miller functions, finite-field elimination, and other coordinate operations
are outside Shoup's generic model and are not rejected by this corollary.

## Theorem 3: the R38 pencil contains the exact toy reduction

The script `p1553_single_interval_dlp_selfcheck_r42.py` works on the exact R38
curve and pencil

```text
E/F_193: y^2=x^3+2x+3,
P=(1,44), order(P)=103,
q=[38]P.                                         (14)
```

Since `38^(-1)=19 mod 103`, the R38 selected path has `q`-coordinates

```text
(19,20,21,22,23,24,25,26,27).                   (15)
```

Choose `B=3`, `Z_0=[27]q`, and orient the interval downward. The pair lists

```text
A_1={Z_0-[3i]q:0<=i<3},
A_2={-[j]q:0<=j<3}                               (16)
```

have exactly the nine endpoints in (15). Every endpoint evaluates to one of
the three selected R38 pencil values `0`, `infinity`, or `192`, so the
selected pair deck is not merely an abstract interval: it is precisely the
union of three complete fibers of the common degree-three pencil.

For the detailed secret `Q=[77]q`, the three known-log queries are

```text
[0]q, [9]q, [18]q.                               (17)
```

There are two labelled hits because of the small-group wrap. Both recover 77.
The canonical labels are

```text
giant labels (0,2),
query index 2,
selected labels (1,2).                           (18)
```

The self-check then repeats the construction for every secret
`x in {0,...,102}`. Exact results are

```text
103 secrets tested,
66 secrets with two marked hits,
37 secrets with three marked hits,
0 missing secrets,
0 bad marked recoveries.                         (19)
```

This is a strong positive toy control: a real common pencil, complete selected
fibers, a one-path union, arbitrary DLP targets, and marked scalar recovery
coexist. It is not an asymptotic attack. The group order is not at the
`B^5` scale for `B=3`, and the toy does not construct R35 heavy complete-key
support across fifth shifts.

## What remains outside the reduction

The common pencil alone does not exclude the DLP encoding at degree three.
The surviving asymptotic operation must therefore identify one of these
stronger inputs:

1. an asymptotic degree-`B` pencil whose `B` complete fibers form a
   `B^2`-point interval;
2. R35/R36 heavy complete-key marks and their source coupling across shifts;
3. a coordinate identity that combines all `B^2` giant endpoints while
   failing on the path-only interface;
4. a cancellation-free marked factor output under all exceptional charts.

Items 1-4 are not known to coexist. R38 proves item 1 only at degree three;
R36 proves conditional root source replay but not the global target locator.

A coordinate-level circuit that realizes all four with query below `B^(1/2)`
would not violate Shoup's theorem, because it would be non-generic. It would be
the requested cryptanalytic candidate and must then pass R10, rank, logs, and
blind descent.

## Disposition

```text
DRAFT_REVIEW_REQUIRED_STRONG_GENERIC_BOUNDARY_POSITIVE_PENCIL_TOY_GATE
ONE_B2_INTERVAL_PLUS_ONE_B2_GIANT_DECK_AND_B_QUERIES_ENCODE_DLP
EVERY_MARKED_HIT_RECOVERS_THE_SCALAR
GENERIC_SETUP_B9_OVER4_FORCES_QUERY_B3_OVER2
R36_REQUIRED_QUERY_BELOW_B1_OVER2_IS_FULL_EXPONENT_BEYOND_GENERIC
R38_DEGREE3_PENCIL_REALIZES_THE_SELECTED_INTERVAL
ALL_103_TOY_SECRETS_RECOVERED_ZERO_BAD_MARKS
COORDINATE_LEVEL_OPERATION_NOT_EXCLUDED_BY_SHOUP
ASYMPTOTIC_INTERVAL_PENCIL_HEAVY_KEYS_GLOBAL_LOCATOR_ABSENT
NO_R10_RANK_LOGS_OR_DESCENT
NO_BREAKTHROUGH
```

Exactly one next action: search for an asymptotic coordinate identity that is
defined only when the selected interval is a union of complete fibers of one
degree-`B` pencil and that combines the `B^2` giant endpoints without generic
scanning. Start from the R42 toy and require the identity to recover all 103
marked secrets while hiding scalar labels; then attempt a degree-nine lift.
Reject any circuit using only interval endpoints, group additions, generic
Miller support, or path equality, since those are exactly the reduction
interface. Bind any genuine pencil-only survivor to R36 source marks and R10.
