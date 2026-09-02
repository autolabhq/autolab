# P1553 signed partition-packing gate R30

## Classification

- Owner: existing P1553/IDEA-057/IDEA-195 complete finite-exceptional-image
  lane; no new idea ID.
- Evidence: exact signed block-packing and weighted-difference reduction; no
  run.
- Status: `DRAFT_REVIEW_REQUIRED_SCOPED_REDUCTION`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: when the selected value alphabet owns all but
  `O(B^2)` sources of a claimed `O(B^2)` complete image, the `+T/-T`
  interface creates `2B` near-complete fiber partitions of the same `B^2`
  points. A pair-packing inequality forces `Omega(B^4)` repeated point-pairs
  across these signed partitions. Weighted by signed-shift differences, this
  implies `Omega(B)` distinct difference shifts whose translate curves each
  place `Omega(B^2)` normalization-collision defect on the selected value
  grid. Thus the complete near-total version of R27 branch (B) collapses into
  the R28 selected-singularity branch on a signed difference deck. Mixed
  mechanisms or constant-mass partial containment remain outside this scoped
  closure. No residual-divisor pencil, target interface, or ECDLP
  breakthrough is supplied.

R29 treated one translated branch at a time. That abstraction admits the
affine-plane control: roughly `B` orthogonal partitions of `B^2` points. A
complete Kummer key contains both translated branches. The signed deck has
roughly `2B` partitions, exceeding the affine-plane scale when their domains
are nearly complete. R30 makes this excess quantitative without assuming an
affine plane, generic coefficients, or pairwise orthogonality.

The reduction is deliberately mechanism-scoped. It applies when collisions
outside the selected both-branch stratum are not credited to another
certified mechanism. If a candidate mixes several strata, it must report
their exact source masses and collision energies separately; it cannot assign
the same compression to each.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R24 distinguished-value mass gate | `dd9b27fa5114e2121463612f5afaf09db12de30a0a0c4397f1dc37aafac1e99f` |
| R25 popular-difference complete-triple gate | `28b24d76a8fbecf50b755c7f808283664bfd52ba50395cc097d43a191ef7f70a` |
| R27 translate-curve Bezout-saturation gate | `dee8a0913854e0ce3e717ecdbc3e8aea73b987b2b2fb71790a470f501dd0c2d9` |
| R28 selected-singularity common-subdivisor gate | `c24183bfe4823a9318ae06769763c94cfed99cb4291807f06b85f397fd53f662` |
| R29 partial-orthogonal-grid gate | `2d675720d215e78d1fa83e3a33a91ceb6b41060c13031dedf1ba6943dbd5e9eb` |
| R29 parent report | `b534dff7ba2407bd32fd4b1492882c384497f34d746cd6f4220f78d38cdea05f` |
| R29 bundle hash list | `5619581cd011919d6b3f4523ecb090bd150fa0b4fdf3ef9d4db799a647f83ee1` |
| R29 staging receipt | `38cb018730122b9390bdf0d51acb748bb4f3c1c2edfe4de95fdc7427bc5089ec` |

## Frozen complete selected-value model

Retain the R27-R29 separable degree-`c` map

```text
g:E'->P^1
```

with trivial translation stabilizer on the prime-order group `G'` of order
`N`. Freeze the canonical balanced point

```text
|C|=r=c,
Z={P in G':g(P) in C}, |Z|=z=c^2,
Q={T_1,...,T_L}, L=c,
N>c^2.
```

The equalities simplify constants. The same proof works with
`z=(1+o(1))*c*r` and `2L/r>=1+eta` for fixed `eta>0`.

Choose one signed representative for each Kummer shift class, so

```text
Sigma=Q union (-Q), |Sigma|=M=2L=2c.              (1)
```

Odd prime order makes the two signed copies disjoint.

For `T in Q`, define the exact both-branch selected count

```text
h_T=#{P in Z:P+T in Z and P-T in Z},
H=sum_(T in Q) h_T,
n=z*L=c^3.                                         (2)
```

Let `K_complete` be the attained complete-key image. Assume this selected
stratum owns the claimed nontrivial compression:

```text
every source outside the H sources is a singleton key and shares no key
with a source inside the stratum.                    (ownership)
```

This is the same scoped mechanism premise used by R24. It does not deny that
a separately certified mixed mechanism could identify outside sources.

## Theorem 1: an `O(c^2)` owned image forces near-total signed domains

The ownership premise gives the exact image bound

```text
K_complete>=n-H.                                   (3)
```

If

```text
K_complete<=A*c^2
```

for fixed `A`, then

```text
sum_(T in Q) (z-h_T)<=A*c^2.                       (4)
```

For a signed shift `S in Sigma`, put

```text
I_S=|Z intersect (Z-S)|.
```

Both-branch containment implies

```text
h_T<=I_T and h_T<=I_(-T).
```

Therefore (4) yields

```text
sum_(S in Sigma) (z-I_S)<=2*A*c^2.                (5)
```

The signed partitions are near-complete in aggregate. Their average missing
mass is only `O(c)` out of `z=c^2`; all but `o(c)` signed shifts have
`I_S=(1-o(1))*z` after any fixed-error trimming.

## Theorem 2: exact signed block-packing inequality

For `S in Sigma` and `b in C`, define the output block

```text
B_(S,b)={P in Z:P+S in Z_b},
k_(S,b)=|B_(S,b)|.
```

For fixed `S`, the `r` blocks are disjoint and

```text
sum_(b in C) k_(S,b)=I_S.                          (6)
```

Count unordered point pairs lying together in one output block:

```text
A_pair=sum_(S,b) binomial(k_(S,b),2).
```

Convexity in each signed partition gives

```text
A_pair
 >=(1/2)*sum_(S in Sigma) (I_S^2/r-I_S).          (7)
```

For one unordered pair `{P,R} subset Z`, `P!=R`, let

```text
m_(P,R)=#{S in Sigma:
          g(P+S)=g(R+S) in C}.
```

Then exactly

```text
A_pair=sum_({P,R}) m_(P,R).                        (8)
```

The repeated signed-partition pair count is

```text
R_pair=sum_({P,R}) binomial(m_(P,R),2).            (9)
```

For every integer `m>=1`, `binomial(m,2)>=m-1`. At most
`binomial(z,2)` point pairs have positive `m`. Combining (7)-(9) proves the
exact inequality

```text
R_pair
 >=(1/2)*sum_S (I_S^2/r-I_S)-binomial(z,2).        (10)
```

At the frozen point, equation (5) gives

```text
sum_S I_S^2/r=2*c^4-O(c^3),
sum_S I_S=2*c^3-O(c^2),
binomial(z,2)=c^4/2-O(c^2).
```

Hence

```text
R_pair>=c^4/2-O(c^3)=Omega(c^4).                  (11)
```

This is the signed analogue of the affine-plane dimension obstruction. It
does not assume that any two partitions are already orthogonal.

More generally, with near-complete domains the leading term in (10) is

```text
(z^2/2)*(M/r-1).
```

Thus the packing excess appears whenever the number of signed partitions is
larger than the selected-value alphabet by a fixed factor. The complete
Kummer deck has `M/r=2` at the canonical point.

## Theorem 3: packing excess forces selected singularity saturation

Let

```text
Delta=(Sigma-Sigma) minus {O},
r_Delta(V)=#{(S,U) in Sigma^2:S!=U and U-S=V}.
```

For `V in Delta`, define the selected ordered normalization-collision defect

```text
D_V=#{(X,Y) in (G')^2:X!=Y,
      g(X)=g(Y) in C,
      g(X+V)=g(Y+V) in C}.                         (12)
```

This is exactly the selected singularity-pair count for the translate curve
`Gamma_V`, before harmless ordered/unordered constant factors. R27 gives

```text
D_V<=2*((c-1)^2-1)=O(c^2).                        (13)
```

Each item counted by `R_pair` chooses a point pair and two signed shifts
`S,U` under which that pair has equal selected values. Translating the point
pair by `S` gives an item of `D_(U-S)`. Conversely, a fixed item of `D_V` has
at most `r_Delta(V)` signed-shift-pair lifts. Therefore, up to the fixed
orientation factor,

```text
R_pair<=sum_(V in Delta) r_Delta(V)*D_V.           (14)
```

Also

```text
sum_V r_Delta(V)=M*(M-1)=Theta(c^2),
max_V r_Delta(V)<=M=2c.                            (15)
```

Equations (11)-(15) imply two conclusions.

First, a positive fraction of signed shift-pair representations have a
difference `V` with

```text
D_V=Omega(c^2).                                    (16)
```

Otherwise the weighted sum in (14) would be `o(c^4)`.

Second, there are at least

```text
Omega(c) distinct differences V                    (17)
```

satisfying (16). Indeed, one such `V` contributes at most
`O(c)*O(c^2)=O(c^3)` to (14), while (11) requires `Omega(c^4)` total.

Thus an owned `O(c^2)` complete image cannot remain purely in R27 branch
(B). The signed interface forces `Omega(c)` distinct difference shifts whose
translate curves each consume a constant fraction of their entire
`Theta(c^2)` delta budget on `C x C`. These are exactly R28 branch-A objects,
now on a prospectively derived signed difference deck.

## Consequence for the surviving constructor

The independent approximate partial-grid route is closed under the ownership
premise. A candidate with near-total selected containment must supply the R28
common sub-divisors and residual identities

```text
sum(A_(V,a,b))-sum(B_(V,a,b))=c*V
```

for `Omega(c)` singularity-saturating differences `V`, in addition to the
original complete-key counts and sources.

This does not prove that the resulting residual-divisor pencil is
impossible. It removes a bookkeeping escape: a candidate cannot call the
one-branch transitions almost simple while ignoring the repeated pairs
created by the second signed branch.

If the ownership premise fails, the candidate must identify the other
mechanism and provide a disjoint collision-energy ledger. Merely stating
that outside sources may also collide is not a constructor, because those
collisions still owe geometry, counts, sources, and cost.

## Controls and boundaries

1. The signed deck uses distinct `+T` and `-T` representatives in odd prime
   order. Constant Kummer sign factors do not alter exponents.
2. The block-packing inequality retains unbalanced block sizes and erasures;
   only the asymptotic evaluation freezes `r=L=c` and `z=c^2`.
3. Equation (14) is representation-weighted, so repeated expressions of one
   difference are charged rather than mistaken for distinct curves.
4. The conclusion extracts `Omega(c)` distinct saturated difference curves,
   not `Theta(c^2)` distinct curves.
5. Constant-mass partial containment below the packing threshold and mixed
   independently certified collision mechanisms remain open.
6. R30 is not an arithmetic-circuit lower bound and does not refute the R28
   residual-divisor pencil.
7. No run, complete source compiler, fresh target, R10 index, density, rank,
   factor logs, blind descent, unrestricted lower bound, Shoup improvement,
   or breakthrough is supplied.

## Deduplication

- R24 owns the source-mass ownership principle for a selected stratum.
- R25 owns popular differences and additive energy.
- R27 owns delta/Bezout saturation and the initial two-branch dichotomy.
- R28 owns common sub-divisors and residual Abel sums.
- R29 owns one-branch partial orthogonal grids and affine-plane controls.
- R30 adds the complete signed block-packing inequality and the weighted
  reduction of near-total branch (B) to `Omega(c)` R28-saturated difference
  shifts.

## Scoped disposition

```text
owned O(B^2) complete image: selected both-branch mass is B^3-O(B^2)
signed near-complete partitions: 2B
repeated signed point-pair mass: Omega(B^4)
distinct difference shifts with selected delta Omega(B^2): Omega(B)
independent near-total branch-B mechanism: closed into R28
residual-divisor pencil: still open
mixed or constant-mass mechanisms: separately chargeable, not closed here
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Analyze the forced R28 residual-divisor family on the `Omega(B)` saturated
signed differences from (16)-(17). Require one prospective degree-`B` pencil,
compact common-subdivisor generation, residual `cV` certificates, only
`O(B^2)` complete keys, and exact coupled sources. Reject scalar intervals,
isogeny stabilizers, repeated difference advice, mixed-mechanism double
counting, or any route missing fresh target, R10, rank, logs, and descent.
