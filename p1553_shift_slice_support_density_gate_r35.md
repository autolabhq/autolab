# P1553 shift-slice support/density gate R35

## Classification

- Owner: existing P1553/IDEA-195 erased-image, selected-shift, target-density,
  and Query2P1 lane; no new idea ID.
- Evidence: exact normalization-energy, sampling, and target-count bounds; no
  cryptanalytic run.
- Status: `DRAFT_REVIEW_REQUIRED_CONDITIONAL_POSITIVE_AND_DENSITY_GATE`.
- Labels: `theorem-only`, `model-bound`, `novelty-unverified`,
  `independent-review-required`.
- Cryptanalytic result: a fixed good fifth-shift slice can be enumerated in
  `O(B^2)` work and necessarily contains `Omega(B^2)` distinct oriented
  complete keys. `O(log B)` random slices recover every key occurring on a
  constant fraction of shifts, and those keys carry all but a controllable
  fraction of the complete source mass. This is a genuine cap-sized support
  compiler with exact coupled sources and no summation-polynomial
  elimination. It does not preserve target density if the returned fifth
  occurrence is restricted to the scanned slices: `m` retained shifts give
  random-target success at most `O(m/B)`. Under the direct online cap and
  setup cap, that route has campaign exponent at least `N^(0.6-o(1))`, above
  rho. A locator for unscanned sources above the captured keys remains open.
  No Shoup-bound improvement or ECDLP breakthrough is supplied.

R34 found a reduced degree-`O(B^2)` carrier for all selected shifts but left
its radical-first construction open. R35 observes that support construction
does not need the global curve union. One fixed shift has only `B^2` candidate
endpoints, already inside setup. The translate-curve normalization defect
bounds collisions within that slice, so a dense slice cannot collapse to a
small key set.

The apparent shortcut is therefore real at the support layer and false at the
source-density layer. Sampling a few slices discovers nearly all heavy global
keys, but one stored source per key still comes from the sampled fifth deck.
Using only those sources removes a factor `B` from the five-list candidate
mass. To exploit the unscanned multiplicity, the algorithm must answer a new
marked fiber query over all fifth labels under every adaptive restriction.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R13 erased-image geometric-resolution gate | `78ff09226852f140a8c33ce5ba98106ed62007b7126673d3f3af2c49d08a2cc5` |
| R27 translate-curve Bezout-saturation gate | `dee8a0913854e0ce3e717ecdbc3e8aea73b987b2b2fb71790a470f501dd0c2d9` |
| R31 near-period run compiler gate | `d624b76f30e94289f85180ef3a4de14d0887437a25b1fcefe495d1f7216fa4b9` |
| R34 summation-surface/grid-residue gate | `84bb7fa7288a76099378a9b475aededbbcb27e53564f7f55a730e127e457656a` |
| R34 bundle hash list | `3e90b0b3976629e2d6a76de4d48c29912c6f96b48213e81e94ab5ccb52e05612` |
| R34 staging receipt | `a4aa4dd51f0dd4c016040538f8f5d07332bc70aaa6a6090d2e5a625c4a3aaef0` |

## Frozen oriented slice model

Retain R34's maximal-stabilizer quotient and selected supports:

```text
g:E->P^1, degree(g)=c=B,
Z={P in G:g(P) in C}, |Z|=c^2,
Q subset G minus {O}, |Q|=L=c.
```

For `q in Q`, define the complete domain and oriented key

```text
H_q=Z intersect (Z-q) intersect (Z+q),
kappa_q(P)=(g(P),g(P+q),g(P-q)).                  (1)
```

The unordered projective branch key is a constant-degree quotient of (1).
R35 works on the oriented cover, keeps the sign of `q`, and merges the two
orientations only after exact source verification.

For an attained oriented key `y`, put

```text
m_q(y)=#{P in H_q:kappa_q(P)=y},
n_q=|H_q|,
k_q=#{y:m_q(y)>0}.                                (2)
```

The favorable near-total selected boundary supplies `n_q=Theta(c^2)` for a
positive fraction, and in R30 for all but a controlled boundary mass, of the
selected shifts.

## Theorem 1: one good slice has quadratic distinct support

The first two coordinates of `kappa_q` factor through the bidegree-`(c,c)`
translate curve

```text
Gamma_q=closure{(g(P),g(P+q)):P in E}.
```

R27 gives normalization genus one and total delta defect

```text
delta(Gamma_q)=(c-1)^2-1.                         (3)
```

If several normalization points map to one singular point, every unordered
pair of distinct branches contributes at least one to its delta invariant.
Therefore complete oriented-key collisions within one shift obey

```text
sum_y binom(m_q(y),2)<=delta(Gamma_q).             (4)
```

The slice second moment is consequently

```text
sum_y m_q(y)^2
  =n_q+2*sum_y binom(m_q(y),2)
  <=n_q+2*((c-1)^2-1)
  <3c^2.                                           (5)
```

Cauchy-Schwarz now gives

```text
k_q>=n_q^2/(sum_y m_q(y)^2).                      (6)
```

If `n_q>=alpha*c^2` for fixed `alpha>0`, then

```text
k_q>=alpha^2*c^2/(alpha+2)=Omega_alpha(c^2).       (7)
```

This is stronger than an output-size promise: every dense good slice itself
has quadratic distinct support. Enumerating it is direct. Scan the represented
`c^2` points of `Z`, test `P+q,P-q in Z` in the R31 dictionary, read the three
stored value labels, and insert the complete projective key into a hash table.
The work and state are

```text
O(c^2) field/group/hash operations and O(c^2) words. (8)
```

Each output key stores one exact `(P,q)` and the endpoint occurrence
backpointer already attached to `P`. No scalar logarithm, resultant, grid
residue, or `c^3` histogram is used.

## Theorem 2: logarithmically many slices capture heavy global support

Let `Y` be the global oriented image, `K=|Y|<=A*c^2`, and put

```text
m(y)=sum_q m_q(y),
t(y)=#{q:m_q(y)>0}.                                (9)
```

Summing (5) over all shifts gives

```text
sum_(q,y) m_q(y)^2<3c^3.                          (10)
```

For the keys with `t(y)<tau*c`, Cauchy-Schwarz first over their shift support
and then over the at most `A*c^2` keys gives

```text
m(y)^2<=tau*c*sum_q m_q(y)^2,
sum_(t(y)<tau*c) m(y)<=sqrt(3*A*tau)*c^3.         (11)
```

Thus keys occurring on fewer than a `tau` fraction of shifts carry at most a
`sqrt(3*A*tau)` fraction of the full `c^3` source scale.

Choose `s` shifts independently and uniformly from `Q`. A key with
`t(y)>=tau*c` is missed with probability at most

```text
(1-tau)^s<=exp(-tau*s).                            (12)
```

For

```text
s>=(lambda+3+log_c A)*tau^(-1)*log c,             (13)
```

a union bound over `K<=A*c^2` keys gives failure at most
`c^(-lambda-1)` for capturing every high-shift-support key. Scanning and
unioning those slices costs

```text
O(c^2*tau^(-1)*log c) work,
O(A*c^2) retained distinct-key state,              (14)
```

for fixed `A,tau,lambda`. Every emitted key and source is exact; randomness
can only omit keys. This is a one-sided Monte Carlo support compiler, not an
exact global histogram or completeness certificate.

## Theorem 3: sampled sources lose fifth-deck target density

The support compiler does not make unscanned occurrences available. Suppose
the relation algorithm returns only sources whose fifth x-class belongs to a
retained set `Q_0` of size `s`. The other four colored factor decks have at
most `B^4` x-class tuples. After target-sign normalization, fixing four source
signs determines at most one fifth signed point, so the total normalized
signed source candidates are at most

```text
16*B^4*s.                                         (15)
```

For a uniformly randomized known-log target `R` in the prime-order group,
the exact expected number of candidates hitting `R` is at most

```text
16*B^4*s/N=16*s/B,                                (16)
```

and the success probability is at most the same quantity. This is a direct
counting bound; it does not assume uniformity of image keys or independence
between candidates.

If `s=B^(gamma+o(1))`, the reciprocal target-density loss is at least

```text
d_loss>=1-gamma.                                  (17)
```

Literal slice construction costs `B^(2+gamma+o(1))`, so the setup cap
`B^(9/4+o(1))` requires `gamma<=1/4`. At the inherited direct fresh-target cap
`kappa=5/4`, the campaign exponent is bounded by

```text
lambda
 >=max(2+gamma,1+(1-gamma)+5/4,2)/5
 =max(2+gamma,13/4-gamma)/5.                      (18)
```

For `0<=gamma<=1/4`, the minimum of (18) occurs at `gamma=1/4` and is

```text
lambda>=3/5.                                      (19)
```

This is above Pollard rho's `N^(1/2+o(1))` time. In particular, the
`O(log B)` support sampler has density `B^(-1+o(1))` if it uses only its
stored sources. To retain constant five-list density by literal slices would
need `s=Theta(B)` and `B^3` setup work.

Equation (19) does not reject a faster target query, a batched target family,
or a source locator that reaches unscanned fifth labels. It closes the claim
that cap-sized slice enumeration plus the existing direct online allowance is
already a complete rho-beating campaign.

## Theorem 4: heavy support is not an adaptive source oracle

The dictionary from Theorems 1-2 proves

```text
captured key subseteq actual global image
```

and stores one or several scanned sources per captured key. It does not give
the exact global multiplicity `m(y)`, nor does it answer a child that excludes
all stored sources but contains an unscanned source with the same key.

To exploit the high source mass certified by (11) without paying the density
loss (17), the next operation must take a captured key `y` and a canonical
dyadic restriction and return

```text
#{(P,q) in restricted Z x Q:kappa_q(P)=y}
```

plus one coupled source when positive, using all `B` fifth labels but without
the `B^3` global histogram. This is a marked source-fiber query on R34's
selected-shift curve union. Standard routes are:

| Route | First charged object | Result |
|---|---:|---|
| Scan stored sampled sources | `B^(2+gamma)` setup | exact but density loss (17) |
| Scan all fifth labels for every endpoint | `B^3` incidences | over setup cap |
| Store all sources per captured key | total `B^3` payload | over state cap |
| Per-key scan of all shifts and fiber points | up to `B^2` online | over fresh-target cap |
| Compact marked inverse on addition surface | unsupplied | live exception |

R31's run package compresses the domains but not this key-conditioned count;
R34's addition surface compresses the carrier but not the selected-shift
source multiplicity. The locator is therefore the exact point where those two
positive structures must meet.

## Controls and boundaries

1. The slice support theorem uses oriented complete keys. Unordered merging
   changes constants and retains both sources and diagonal tags.
2. The delta bound counts distinct normalization branches. Ramification at
   one branch does not create an uncharged second source.
3. The sampling theorem is one-sided Monte Carlo: every output verifies, but
   a missed heavy key is not detectable from the sample alone.
4. The target-density bound applies when returned fifth sources are restricted
   to the scanned deck. A proved unscanned source locator escapes it.
5. A target-dependent choice of slices is fresh advice and must be charged in
   every known-log collection and blind descent query.
6. No exact global multiplicities, adaptive source oracle, fresh-target
   action, R10 coefficients, independent rank, factor logs, descent,
   unrestricted lower bound, Shoup improvement, or breakthrough is supplied.

## Deduplication

- R13 owns erased-image multiplicities, source sections, and the child-query
  obligation.
- R27 owns translate-curve normalization and delta defect.
- R31 owns exact selected-domain run construction.
- R34 owns the addition surface and selected-shift curve union.
- R35 adds only fixed-slice support enumeration, heavy-support sampling, and
  the exact sampled-fifth target-density charge.
- R9-R10 retain Query2P1 counts, rank-two control, and complete campaign
  accounting.

## Scoped disposition

```text
one dense fixed shift: Omega(B^2) distinct exact keys in O(B^2) work
O(log B) random shifts: cap-sized heavy-global-key support sampler
exact global key multiplicities: absent
returned sources restricted to s shifts: target success at most O(s/B)
direct-cap campaign with s<=B^(1/4): lambda at least 3/5
unscanned restricted source-fiber locator: absent
fresh target, R10, rank, logs, descent: absent
```

## Exactly one next action

Construct or refute a marked source-fiber locator above the R35 captured key
dictionary. Given one complete projective key and any canonical dyadic child,
it must count and return a coupled endpoint/fifth source across all `B` shifts
inside `B^(5/4+o(1))` fresh-target work and workspace, using the R31 runs and
R34 selected-shift curve union without a `B^3` incidence table. Then implement
the fresh-target action and R10 rank-two control and charge relation density,
independent rank, factor logs, and identical blind descent.
