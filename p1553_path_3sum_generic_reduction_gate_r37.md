# P1553 path-3SUM generic-reduction gate R37

## Classification

- Owner: existing P1553/IDEA-195 fresh-target, path-index, and generic-group
  boundary; no new idea ID.
- Evidence: exact generic-group reduction and deterministic cyclic-group toy
  self-check; no cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_GENERIC_BOUNDARY_GATE`.
- Labels: `theorem-only`, `toy-control`, `non-run`, `model-bound`,
  `novelty-unverified`.
- Cryptanalytic result: R36's density-adjusted target-query threshold is
  exactly a generic-group breakthrough threshold. A path-only index below it
  would solve an arbitrary prime-order DLP in `o(sqrt(N))` group operations.
  This does not exclude an elliptic-coordinate algorithm using the degree-`B`
  pencil and complete-key marks. No such algorithm, Shoup-bound improvement,
  or ECDLP breakthrough is supplied.

R36 reduced the fresh-target projection to a marked translated intersection
between a `B^2`-point pair endpoint deck and a `B^2`-point set represented by
`B` translation paths. Its density loss requires average work below
`B^(1/2-o(1))` on `B^2` known-log targets for a strict sub-rho campaign.

R37 proves that the exponents are not an artifact of the path-block schedule.
The path-only problem already contains generic DLP. Given `Q=[x]P`, four
length-`B` lists encode a mixed-radix giant-step deck and `B` length-`B`
paths. One of `B^2` known-log queries returns labels from which `x` is read
exactly. Setup `B^s` and average query `B^kappa` therefore give a generic DLP
algorithm in

```text
B^(max(s,2+kappa)+o(1))                           (1)
```

group operations. Shoup's generic lower bound puts the boundary at
`sqrt(N)=B^(5/2+o(1))`.

The reduction deliberately omits the degree-`B` map `g`, its common pencil,
the selected values, and R35's heavy complete-key marks. Arbitrary mixed-radix
paths need not be fibers of one elliptic pencil. Those representation-specific
constraints are now the only admitted route to the requested speedup.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R10 factorized-pullback gate | `49eceb2dce63eaceda74afa052d0919d44166758516e177086f2296c545cd4f1` |
| R25 popular-difference gate | `28b24d76a8fbecf50b755c7f808283664bfd52ba50395cc097d43a191ef7f70a` |
| R29 partial orthogonal-grid gate | `2d675720d215e78d1fa83e3a33a91ceb6b41060c13031dedf1ba6943dbd5e9eb` |
| R31 near-period run compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R36 heavy-key source/target-support gate | `103670908f33baee0915372434899a3a710861f501e19c91599f4221fe3f2aa5` |
| R36 bundle hash list | `8af4ef463fac0f34253693860ab8acfcb9da683473e38f7b97d6eb6b5df75cc7` |
| R36 staging receipt | `af3b33bc7b1777e05be4b4b3a7eeaea514bd3162b414621cb7db6613ac569bd2` |

## Frozen path-only target interface

Let `G=<P>` have odd prime order `N`, let

```text
B=ceil(N^(1/5)),
M=B^2,
B^3<N                                           (2)
```

for the asymptotic regime, and retain occurrence labels even if a modular
wrap creates repeated group points.

The path-only target problem receives four public lists of length `B`:

```text
A_1, A_2, A_3, A_4 subset G,                     (3)
```

their marked pair endpoint decks

```text
Z={a_1+a_2:(a_1,a_2) in A_1 x A_2},
V={a_3+a_4:(a_3,a_4) in A_3 x A_4},              (4)
```

a public nonzero path step `q_0`, and a fresh known-log query `R=[r]P`.
It must return occurrence labels satisfying

```text
v+z=R,       (v,z) in V x Z,                     (5)
```

or exact empty. The setup may depend on all four lists but not on the query
label `r`. This is the unmarked necessary projection of R36's full target
router. A full P1553 answer additionally has to attach an extendible captured
key and invoke R36's exact source section.

## Theorem 1: mixed-radix path instances encode generic DLP

Given a generic DLP instance

```text
Q=[x]P,       0<=x<N,                             (6)
```

construct

```text
A_1={[-i*M]P:0<=i<B},
A_2={[j]P:0<=j<B},
q_0=P.                                            (7)
```

Then

```text
Z={[-i*M+j]P:0<=i,j<B}                            (8)
```

is exactly `B` disjoint `q_0`-paths of length `B`: the integer intervals are
separated by `M=B^2`, and their total coefficient range is below `N`.

Construct the second pair deck

```text
A_3={Q-[i*B^4]P:0<=i<B},
A_4={-[j*B^3]P:0<=j<B}.                           (9)
```

Its labelled endpoints are

```text
v_(i,j)=Q-[(i*B+j)*B^3]P.                        (10)
```

Finally issue all `M=B^2` fresh known-log queries

```text
R_r=[r]P,       0<=r<M.                           (11)
```

Write the ordinary integer quotient and remainder of the secret as

```text
x=k*B^3+d,
0<=d<B^3,
0<=k<B^2.                                        (12)
```

The bound on `k` follows from `x<N<=B^5`. Write

```text
k=i*B+j,       0<=i,j<B,
d=h*M+r,       0<=h<B, 0<=r<M.                   (13)
```

Equations (8)-(13) give the labelled witness

```text
v_(i,j)=[d]P,
z_(-h,0)=[-h*M]P,
v_(i,j)+z_(-h,0)=[r]P=R_r.                       (14)
```

Here `z_(-h,0)` denotes the occurrence in (8) with first index `h` and second
index zero. Thus at least one of the `B^2` queries is positive.

For any returned labels `(i,j,h,l,r)`, not just the canonical witness, (5)
and (8)-(10) imply

```text
Q=[(i*B+j)*B^3+r-(-h*M+l)]P.                     (15)
```

Therefore the exact discrete logarithm is

```text
x=((i*B+j)*B^3+r+h*M-l) mod N.                   (16)
```

Modular wraps and duplicate endpoints cannot create a false recovery because
(16) is derived from the returned marked group equality. Complete occurrence
labels are essential; a support-only yes/no bit would require a separate
label-recovery charge.

## Corollary 2: the `B^(1/2)` query threshold is the Shoup boundary

Suppose a generic-group path-only index has preprocessing work
`B^(s+o(1))` and amortized work `B^(kappa+o(1))` for the batch (11), including
all negative queries and one marked positive answer. Constructing the four
lists and the query points costs `B^(2+o(1))` group operations. The reduction
uses total work

```text
B^(max(s,2+kappa,2)+o(1)).                        (17)
```

Shoup's generic-group theorem gives an `Omega(sqrt(N))` group-operation
lower bound for nonnegligible success on prime-order DLP. Since

```text
sqrt(N)=B^(5/2+o(1)),                             (18)
```

every generic implementation of this path-only interface must satisfy

```text
max(s,2+kappa)>=5/2-o(1).                         (19)
```

Under the campaign setup cap `s<=9/4`, equation (19) forces

```text
kappa>=1/2-o(1).                                  (20)
```

This exactly meets R36's strict sub-rho requirement `kappa<1/2-o(1)`. The
same exponent appears from two independent directions:

1. target support `B^4` requires `B^2` known-log attempts for `B` accepted
   rows; and
2. the mixed-radix path family turns those `B^2` queries into one generic DLP
   digit decomposition.

Equation (19) is a generic-group result for the path-only interface. It is
not an unconditional lower bound for algorithms reading Weierstrass
coordinates, evaluating `g`, factoring its fibers, or exploiting algebraic
relations that arbitrary path instances do not possess.

## Theorem 3: target-path batching recovers rho directly

The reduction also explains the usual baby-step/giant-step scale. Let

```text
T={R_r:0<=r<B^2}.                                 (21)
```

If `Z` is a union of `O(B)` `q_0`-paths with total length `B^2`, then each
path contributes one interval of length `O(B^2)` to `T-Z`. Hence

```text
T-Z is a union of O(B) q_0-intervals,
|support(T-Z)|<=O(B^3),
total represented pair multiplicity=B^4.          (22)
```

The other endpoint deck has `|V|<=B^2`. A random-position model therefore
has `O(1)` distinct intersections between the `B^3` and `B^2` supports but
`Theta(B)` lifted target/source occurrences, matching R36's density count.

For interval block length `h`, materialize

```text
{v-[b]q_0:v in V,0<=b<h}                         size O(B^2*h),
{d_j+[a*h]q_0:j<=O(B),a<O(B^2/h)}                size O(B^3/h). (23)
```

A collision returns an exact interval offset after range verification. The
balanced represented schedule is

```text
h=B^(1/2),
work/state=B^(5/2+o(1)).                          (24)
```

With state capped at `B^(9/4)`, storing one side and streaming the other gives
the explicit endpoints

```text
h=B^(1/4) or B^(3/4),
work=B^(11/4+o(1)), state=B^(9/4+o(1)).           (25)
```

A low-memory generic collision walk can return to the `B^(5/2)` rho scale;
neither (24) nor (25) beats it. These are controls, not a proof beyond the
generic reduction.

## Toy self-check

`p1553_path_3sum_shoup_selfcheck_r37.py` instantiates Theorem 1 in the cyclic
group `Z/1009Z` with

```text
B=4,
x=777,
B^3=64,
B^5=1024>=1009.                                  (26)
```

The exact report
`p1553_path_3sum_shoup_selfcheck_report_r37.json` records:

```text
16 target labels,
16 distinct Z endpoints,
16 distinct V endpoints,
4 labelled hits,
0 bad recoveries,
canonical labels target=9, V=(3,0), Z=(0,0),
recovered x=777.                                  (27)
```

The script output reproduces the report byte-for-byte. This checks index and
sign arithmetic only. It is not an elliptic-pencil instance, an asymptotic
experiment, or evidence against Shoup.

## Why the degree-`B` pencil is the remaining escape

The reduction's `Z` is a valid mixed-radix pair sumset and path union, but R37
does not construct a separable degree-`B` map whose `B` selected fibers are
exactly those `B^2` points. On an elliptic curve, all fibers belong to one
two-dimensional pencil inside a degree-`B` complete linear system. Prescribing
`B` arbitrary degree-`B` divisors is therefore much stronger than merely
partitioning points into equal-size or linearly equivalent blocks.

Likewise, the reduction does not attach R35 keys with high support across the
selected shifts. A full target index may use:

```text
the common degree-B pencil,
the addition-surface equation,
the captured heavy-key dictionary,
normalization or residual-divisor marks,
and actual prime-field coordinates.               (28)
```

Any speedup derived solely from group addition, pair labels, path starts,
path offsets, and equality lookups is covered by Theorem 1. To beat (20), a
successor must identify the exact non-generic operation in (28), give its
complete source-preserving circuit, and show why the mixed-radix reduction is
outside its domain without hiding target-dependent advice.

## Harness obligations and controls

1. Preprocessing may depend on the DLP input `Q` through `A_3`; this is legal
   for a generic DLP reduction and every such operation is charged.
2. Query points (11) have known logs and are target-fresh relative to the
   preprocessed lists.
3. All `B^2` positive and negative query costs are included in (17).
4. Every returned endpoint carries its two occurrence labels; duplicate
   support points do not invalidate (16).
5. The reduction is classical. It makes no quantum-query claim.
6. The toy cyclic group checks formulas only and supplies no elliptic
   coordinate or pencil evidence.
7. The generic lower bound cannot reject a coordinate algorithm outside the
   generic model.
8. No marked pencil index, fresh-target P1553 router, R10 coefficients,
   independent rank, factor-base logs, scalar-blind descent, unrestricted
   lower bound, Shoup improvement, or breakthrough is supplied.

## Operation-level deduplication

- R25 owns the scalar-progression positive control and warns that its public
  orientation carries the hidden DLP axis.
- R29 owns the abstract rank-two grid control.
- R31 owns the path compiler.
- R36 owns the density-adjusted marked target-index requirement.
- R37 adds the explicit mixed-radix DLP reduction, its generic exponent
  boundary, the batch interval BSGS control, and the toy arithmetic check.

## Result boundary

```text
path-only target index contains generic DLP: yes
pair-sum lists required by reduction: four lists of size B
known-log queries: B^2
exact marked recovery: yes
generic total work: B^max(s,2+kappa,2)
Shoup boundary: B^(5/2)
with s<=9/4, generic query floor: kappa>=1/2-o(1)
R36 strict sub-rho requirement: kappa<1/2-o(1)
coordinate/pencil escape: admitted and unsupplied
toy cyclic reduction: pass, 0 bad recoveries
fresh target, R10, rank, logs, descent: absent
Shoup-bound improvement: false
breakthrough: false
```

## Primary-source boundary

The generic lower-bound control is Victor Shoup, *Lower Bounds for Discrete
Logarithms and Related Problems*, EUROCRYPT 1997,
<https://www.shoup.net/papers/> and
<https://doi.org/10.1007/3-540-69053-0_18>.

The indexing comparisons remain those in R36. None supplies or excludes an
elliptic-coordinate marked-pencil target index.

## Exactly one next action

Exploit or close the degree-`B` pencil marks excluded from Theorem 1. Require
one explicit coordinate-level operation that takes a fresh known-log target
and the R31 path package, uses the common fibers/addition surface/heavy-key
marks to return a complete key and pair endpoint in average
`B^(1/2-o(1))` work with setup at most `B^(9/4+o(1))`, and fails on R37's
arbitrary mixed-radix paths for a proved algebraic reason. Then bind any
survivor to R10, independent rank, factor logs, and scalar-blind descent.
