# P1553 near-period run compiler gate R31

## Classification

- Owner: existing P1553/IDEA-195 translation-correlation and finite
  exceptional-value lane; no new idea ID.
- Evidence: exact constructive set representation and cost theorem; no run.
- Status: `DRAFT_REVIEW_REQUIRED_CONDITIONAL_POSITIVE`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: the near-total selected containment forced by R30 is
  not merely a structural obstruction. It gives an exact DLP-free compiler
  for every one-branch and complete `+q/-q` source domain. Choose a public
  signed shift with minimum boundary; the `B^2` selected points decompose
  into `O(B)` paths under that shift, and every complete source domain is a
  union of at most `O(B)` aligned subpaths. All `B` domains, exact counts,
  and one source per run fit `O(B^2)` construction and state. This is a real
  positive incidence compiler. It does not compute the histogram of complete
  value triples along the runs, preserve arbitrary factor-base restrictions,
  handle fresh targets, answer R10, or solve ECDLP.

R30 turns an owned `O(B^2)` complete image into near-total invariance of the
selected support `Z` under the signed fifth deck. A generic sparse-sumset
formulation still appears to require `B^3` point-shift incidences. R31 avoids
that table by choosing one near period as a local public step. No scalar log
of that step is used: path construction needs only elliptic additions, point
serialization, and exact membership lookups in the already represented
`B^2`-point support.

Translation commutes with the path step. Therefore the intersection of any
fixed number of translates of `Z` has path starts only where one constituent
translate has a path start. Complete both-branch containment is an
intersection of three translates, so it has at most three times the base
number of runs. This supplies the source geometry that R15 left open.

The values of `g` along those runs need not be simple. Producing exact counts
of

```text
(g(P), unordered pair {g(P+q),g(P-q)})
```

without visiting all `B^3` represented positions is a separate aligned-run
colored-histogram problem. R31 exposes that residual precisely and does not
promote the incidence compiler into a relation algorithm.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R15 translation-correlation sparse-convolution gate | `ea007be0b237b91127e6a42501873f914ad31d6d4fd4393c3ae6b85005893b9b` |
| R25 popular-difference complete-triple gate | `28b24d76a8fbecf50b755c7f808283664bfd52ba50395cc097d43a191ef7f70a` |
| R28 common-subdivisor gate | `c24183bfe4823a9318ae06769763c94cfed99cb4291807f06b85f397fd53f662` |
| R30 signed partition-packing gate | `8876faa2b65f6a44410495209617b50e99cc781caaf73a2f4cb8b359b294006f` |
| R30 parent report | `db6c20a7877a85a754b7bac3becc08fd087c5151283652cebea5840a4c086fed` |
| R30 bundle hash list | `ffedf5472d54e778fc2d8e0f379f38e45c8bce41ec224030978ab42178dc0edf` |
| R30 staging receipt | `f53b15dc9a85860f1472146c5e2ebc0333acb8067d9aa57761c708bac591e7d8` |

## Frozen represented-support model

Let `G'` be cyclic of odd prime order `N`, represented by complete elliptic
points with exact equality and group addition. Freeze

```text
Z subset G', |Z|=z=c^2,
Q={q_1,...,q_L}, L=c,
Sigma=Q union (-Q), |Sigma|=M=2c.
```

The degree-`c` map `g` and value set `C` define `Z`, but the path theorem uses
only the represented point set after it has passed the R24-R30 algebraic
gates.

For `s in Sigma`, define the directed boundary deficit

```text
d_s=z-|Z intersect (Z-s)|
   =#{P in Z:P+s not in Z}.                         (1)
```

R30's owned-image theorem gives

```text
sum_(s in Sigma) d_s<=2*A*c^2                     (2)
```

when the attained complete image has at most `A*c^2` keys and outside
selected sources are singleton and unshared.

Store the `z` point encodings in an exact hash dictionary. A lookup returns
the point's selected value label, factor-base occurrence metadata, and later
its path coordinates.

## Theorem 1: one public signed shift has only `O(c)` paths

Choose

```text
q_0 in Sigma
```

minimizing `d_s`. From (2) and `M=2c`,

```text
d_0=d_(q_0)<=A*c.                                  (3)
```

Because `q_0` is nonzero in a prime-order group, its translation is one
cycle on `G'`. Since `z<N`, the induced directed graph on `Z`, with edge

```text
P -> P+q_0 when P+q_0 in Z,
```

is a disjoint union of paths, not cycles. Its path starts are

```text
Start(Z)={P in Z:P-q_0 not in Z}.                  (4)
```

The number of starts equals the number of missing outgoing edges:

```text
|Start(Z)|=d_0=O(c).                               (5)
```

Walking each path once labels every selected point by

```text
(path_id,offset,path_length)
```

in exactly `O(z)` group additions and dictionary operations, with `O(z)`
state. The coordinate is local and public. It neither computes nor needs
`log_P(q_0)` or the scalar gaps between different paths.

## Theorem 2: a fixed intersection of translates has few runs

For public shifts `a_1,...,a_k`, put

```text
X=intersection_(i=1)^k (Z-a_i),
Z-a={P:P+a in Z}.                                  (6)
```

Translation preserves the `q_0` path boundary, and

```text
Start(Z-a_i)=Start(Z)-a_i.
```

If `P` starts a `q_0`-run of `X`, then at least one constituent membership
`P+a_i in Z` fails at the predecessor `P-q_0`. Therefore

```text
Start(X) subset union_(i=1)^k (Start(Z)-a_i),
|Start(X)|<=k*d_0.                                 (7)
```

This is an exact run-count bound, including empty intersections and paths of
length one.

The runs can be constructed without walking their interiors. Enumerate the
at most `k*d_0` candidate starts in (7), keep those in `X` whose predecessor
is not in `X`, and use the precomputed path coordinates of every `P+a_i`.
The run length is exactly the minimum remaining length among those `k`
constituent paths. Thus construction costs `O(k*d_0)` additions and lookups
plus output records.

Each record stores:

```text
source path and offset,
one translated path and offset for every a_i,
run length,
one exact starting source point.
```

All points in the run are obtained by adding successive multiples of `q_0`
to the stored starts; no scalar chart outside the local offsets is present.

## Theorem 3: all complete fifth domains compile in `O(c^2)`

For one fifth shift `q`, the complete selected source domain is

```text
H_q=Z intersect (Z-q) intersect (Z+q)
   ={P in Z:P+q in Z and P-q in Z}.                (8)
```

Equation (7) with `k=3` gives

```text
H_q is a union of at most 3*d_0=O(c) q_0-runs.     (9)
```

The one-branch domain `Z intersect (Z-q)` uses at most `2*d_0` runs.

Across all `L=c` fifth shifts, the complete run package has

```text
at most 3*L*d_0=O(c^2) records.                   (10)
```

Including the initial path decomposition, exact construction work and
retained state are

```text
O(z+L*d_0)=O(c^2).                                (11)
```

Integer source counts are the sums of run lengths. One source for every
nonempty full domain or run is already stored. Signed shifts, identity
avoidance, and boundary points are exact because the construction uses the
complete group law and dictionary membership rather than affine formulas.

This is below the campaign's `B^(9/4+o(1))` setup/state cap at `c=B`. It is a
genuine positive compiler for translation containment and source replay at
the full selected-support level.

## Theorem 4: compact translation runs do not imply compact value triples

Attach to every selected point its value label

```text
lambda(P)=g(P) in C.
```

One complete run record for `q` aligns three finite strings along the common
step `q_0`:

```text
lambda(P+t*q_0),
lambda(P+q+t*q_0),
lambda(P-q+t*q_0),       0<=t<run_length.          (12)
```

The required image contribution is the exact histogram of

```text
(lambda_0(t), unordered pair {lambda_+(t),lambda_-(t)}).
```

There are only `O(c^2)` run records, but their total represented length is
the original `Theta(c^3)` source mass. Arbitrary label strings can realize a
new key at every position. The claim `K_complete=O(c^2)` promises repeated
keys globally, not an algorithm for finding and counting them.

The residual operation is therefore:

```text
an exact output-sensitive histogram of aligned triples from O(c^2)
three-string run records over an alphabet of size c, with total length c^3,
returning integer counts and one coupled source for every attained key.
```

A literal scan costs `Theta(c^3)`. Per-path dense bitsets, a represented
three-way incidence tensor, or one record per position restores the same
traffic. Rolling hashes certify equality of supplied substrings but do not
enumerate exact value triples or counts. R31 does not prove a lower bound
against compressed strings, sparse polynomial products, range-color data
structures, grammar compression, or a family-specific recurrence induced by
`g`.

## Restriction, target, and rank boundary

The run package is stable under restrictions that are unions of stored
`q_0` intervals: intersecting one run with such a restriction only fragments
at the restriction boundaries. The campaign's canonical dyadic factor-base
and pair-tree restrictions are not automatically path-compatible. An
accepted compiler must either build a path-compatible exact source tree or
prove that all adaptive intersections retain cap-sized fragmentation.

The package is target-independent containment data. It does not:

1. specialize the complete image under a fresh target;
2. answer R10's queried multiplicative coefficients;
3. establish relation density or independent signed row rank;
4. solve factor-base logarithms; or
5. perform scalar-blind target descent.

Path coordinates can also expose rank loss. If value fibers or relation rows
depend only on a bounded number of path origins and offsets, the factor-base
columns collapse to that bounded chart. A passing histogram must report
actual independent columns rather than count repeated path occurrences as
rank.

## Controls and boundaries

1. The theorem requires explicit represented `Z`; constructing the algebraic
   degree-`c` pencil and its `c^2` subgroup points remains charged upstream.
2. The minimum-boundary step belongs to the prospective signed deck and is
   selected without source or target adaptation.
3. No global scalar of `q_0`, path origin, or gap is computed.
4. Empty, singleton, and boundary runs are represented exactly.
5. The `O(c^2)` result covers source containment, counts, and run starts, not
   the complete value histogram.
6. Arbitrary dyadic restrictions may fragment runs and remain unproved.
7. No run, complete relation campaign, fresh target, R10, rank, factor logs,
   descent, unrestricted lower bound, Shoup improvement, or breakthrough is
   claimed.

## Deduplication

- R15 owns generic two-frequency correlation and the missing coupled source
  interface.
- R25 owns popular differences and scalar-progression controls.
- R28 owns selected common fiber sub-divisors.
- R30 owns signed near-total containment and packing.
- R31 adds the minimum-boundary path decomposition, intersection-run theorem,
  `O(c^2)` complete containment compiler, and aligned-run colored-histogram
  residual.

## Scoped disposition

```text
near-total selected support: supplied conditionally by R30 ownership
public minimum-boundary step: exists with O(B) boundary
selected support paths: O(B)
all complete plus/minus source domains: O(B^2) exact run records
construction and state: O(B^2)
scalar DLP chart: not used
complete value-triple histogram: absent, standard scan B^3
arbitrary dyadic restriction stability: absent
fresh target and R10: absent
complete ECDLP path: absent
```

## Exactly one next action

Build or refute the aligned-run colored-histogram residual in Theorem 4 for
the actual degree-`B` pencil. Require exact complete keys, integer counts, one
coupled source, path-compatible adaptive restrictions, and total setup below
`B^(9/4+o(1))` without scanning `B^3` positions. Then bind the same structure
to fresh-target action, R10, independent rank, factor logs, and scalar-blind
descent; reject dense bitsets, represented incidence tensors, scalar gap
logs, or run counts without value counts.
