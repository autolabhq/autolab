# P1553 5A+5C coordinate-filtration probe R83

## Classification

- Owner: existing P1539/P1553/R82 colored-sum and source-router frontier; no
  P1554 and no new idea ID.
- Evidence: 72 frozen x-, y-, and encoding-hash filter profiles on all eight
  R82 instances, target-coupled offset controls, an exact divisible-order
  quotient positive control, and 512 supplied-source Semaev-chain replays.
- Status: `COORDINATE_BUCKET_FILTRATION_ENTROPY_REPLAY_FAIL`.
- Labels: `exact-finite`, `prospective`, `scalar-blind-source`,
  `scoped-theorem`, `verifier-dlp-separated`, `novelty-unverified`.
- Cryptanalytic result: no subcap known-RHS source, factor-log solve,
  identical target descent, Shoup improvement, or ECDLP breakthrough.

R82 produced a passing local `3A+3C` endpoint compiler but returned to the
colored ten-sum

```text
5A+5C=R                                                        (1)
```

for a full relation or target descent. R83 tests the most direct proposed
escape: discard partial sums with field-coordinate buckets and replay only a
target-coupled offset.

No tested bucket is addition-compatible or complete. Replaying every bucket
or offset is exact, but restores the discarded source entropy. This closes
only frozen coordinate-bucket filtrations. A target-forced algebraic
invariant, compact source section, marked resultant, or arbitrary FFE circuit
remains open.

## Bound inputs

| Input | SHA-256 |
|---|---|
| R82 Cartesian-sum report | `ccc83fec0dc411ce35f27f21bcb1e543f6fe3d85a95aa24217701d8c9bbf5832` |
| R82 gate | `7c34e1d905c95a756689d4ec0ea92c6bd47808bcb3407d858ce08cccf75fd55e` |
| R31 registry | `0c76f5d8385bf97b9008314736d8d3a8593e6d96fc4f25af380c2ef47fc8907f` |
| Registry-bound P1539 audit | `634e5a7d2847e849a2e46178f31500f19109e9a9d88a2bf8c70d1f0afe4d467a` |

## Deduplication

The P1539 audit already identifies its locator as colored elliptic 5SUM and
screens direct splits, current kSUM indexing, and neutral-mask Wagner merges.
R83 does not relabel that route. Its narrower addition is specific to R82:

1. retain one canonical `5A+5C` source for every attained endpoint;
2. measure both static buckets and target-coupled offsets;
3. replay every offset exactly;
4. compare with a true proper quotient in a divisible-order group; and
5. bind supplied sources to an exact factorized Semaev chain.

Semaev's [summation polynomials](https://eprint.iacr.org/2004/031) define the
field-equation interface. Wagner's
[generalized birthday algorithm](https://people.eecs.berkeley.edu/~daw/papers/genbday.html)
is the staged-filter comparison. Shoup's
[generic lower bound](https://www.shoup.net/papers/dlbounds1.pdf) remains the
nonclaim boundary.

## Frozen filters

For every attained R82 endpoint, freeze the lexicographically first weighted
multiset source

```text
(a_1,...,a_5;c_1,...,c_5).                                    (2)
```

The partial split is

```text
U=A_a1+A_a2+C_c1+C_c2+C_c3,         size exponent B^(13/5),
V=R-U,                               size exponent B^(12/5).    (3)
```

For `m in {2,4,8}`, test

```text
h_x(P)=x(P) mod m,
h_y(P)=y(P) mod m,
h_H(P)=SHA256(encode(P)) mod m.                                (4)
```

Identity maps to bucket zero. Static filters retain `h(U)=b`. The
target-coupled version retains

```text
h(U)-h(R)=d mod m.                                             (5)
```

All source choices, splits, maps, moduli, and target coupling are frozen
before filter outcomes. Verifier BSGS labels only index attained target
points; the filters consume group points and public encodings.

## Prime-order quotient theorem

Let `G` have prime order `q`. The kernel of every homomorphism `G->H` is a
subgroup of `G`, hence it is `{O}` or `G`. A nonconstant homomorphism is
therefore injective and has image order `q`. It cannot be a proper
many-to-one filtering quotient.

This theorem does not say that x-, y-, or encoding buckets are homomorphisms.
The finite controls test and reject that premise directly. Across every
instance, map, and modulus, no tested bucket is additive on its sampled
partial endpoints.

The positive control uses

```text
Z/808Z -> Z/8Z,     z |-> z mod 8.                             (6)
```

Its kernel has size 101. All 4,096 checked additions commute with the
quotient. This distinguishes a real Wagner-compatible quotient from an
arbitrary encoding bucket.

## Source survival

Across 72 profiles:

```text
best static single-bucket survival fraction       0.1480..0.7063,
best target-coupled offset survival fraction      0.1270..0.6111,
single bucket or offset retaining every target    0/72,
all bucket replays recovering every target        72/72,
all coupled-offset replays recovering every target 72/72.       (7)
```

The range includes small finite decks and nonuniform coordinate residues.
The decisive result is exact incompleteness, not proximity to `1/m`.

## Entropy-replay theorem

At relation scale,

```text
number of atom sources          B^(5+o(1)),
distinct attained targets       B^(5+o(1)),
group order N                   B^(5+o(1)).                    (8)
```

Freeze one canonical source per attained target. Any accepted static subset
of `M=B^(5-delta)` canonical sources covers at most that many targets, so its
average target success density is at most

```text
B^(-delta+o(1)).                                             (9)
```

Unless the constraint is forced by the target equation or a compact selector
covers every endpoint, restoring constant success requires
`B^(delta-o(1))` masks, offsets, or retries.

For the R82 explicit join:

```text
baseline work                         B^2.6,
desired work per filter               B^1.25,
required filtering/replay exponent    delta=1.35,
restored total work                   B^(1.25+1.35)=B^2.6.      (10)
```

For the generic collision baseline:

```text
baseline work                         B^2.5=N^0.5,
required filtering/replay exponent    delta=1.25,
restored total work                   B^2.5.                    (11)
```

This is a counting theorem for explicit accepted-source buckets and
target-coupled offset replay. It is not an arithmetic-circuit or elimination
lower bound.

## Semaev/FFE replay

For 64 canonical sources per instance, order the five A atoms and five C
atoms and retain cumulative group sums. Every regular merge verifies

```text
S_3(x(T_i),x(P_(i+1)),x(T_(i+1)))=0.                           (12)
```

The exact receipt is

```text
supplied sources                       512,
group endpoint failures                  0,
regular S3 auxiliary-chain checks     4,608,
S3 failures                               0,
identity exceptional steps                0.                   (13)
```

This factorized chain is an exact certificate for a supplied source. It does
not construct a source, choose a complete bucket, or solve the ten-sum inside
the online cap.

## Campaign boundary

R83 passes 11 of 16 obligations: frozen geometry and filters, canonical
source coverage, static/coupled measurements, the prime-order theorem,
divisible-order positive control, nonadditivity controls, exact bucket
replay, exact group sources, and exact S3 chains.

It fails the subcap full query, factor logs without verifier DLP, identical
descent, generic-prime breakthrough, and Shoup-improvement obligations.

The result rejects only the tested coordinate-bucket grammar. The surviving
mechanism must cover every attained target without enumerating all masks. A
target-forced low-degree invariant or a P1510-style marked-resultant source
section is outside this result.

## Evidence

| Artifact | SHA-256 |
|---|---|
| `p1553_5a5c_coordinate_filtration_probe_r83.py` | `0707a53515279e5e5985623d16a731eb1d41dbfb5b6a6a980dcc0cc30a514833` |
| `p1553_5a5c_coordinate_filtration_probe_report_r83.json` | `1478cdf21493ffbeaed0859af849ea6f2835027f23db7e3e4c008f3f24db500c` |
| `frozen_5a5c_coordinate_filtration.json` | `3a5f7e7fb347098745ce96d2028e33c406263368a92792a8bb53e3e8e2c1e77f` |
| `partial_filter_composability_and_false_positive_controls.json` | `b0dd2c5f2675e8e0967a6c0722f26f35700ac9b68f41aafa5f461e17c1cb71af` |
| `summation_polynomial_ffe_source_replay.json` | `68bbc08fb17f6f63096c9f9e0de23da8ebbc6d7483b10a6348911086bc90c6fa` |
| `full_query_state_cost_ledger.json` | `4fcaf43e7678bd0542eb6e353f1a183978060d59486a581ed48492c1f2d6600a` |
| `factor_logs_and_identical_descent.json` | `46ad990d3e3d0f41ed4042490ffd6c8a68980869e966304d4936f807476928f0` |
| `tasks/ecdlp_index_calculus/tests/test_p1553_5a5c_coordinate_filtration_probe_r83.py` | `50757d8fb6e4a6d2720a5a6ea73b9c97295eeb8b0d1619cce2ada88743c205cf` |

Targeted replay:

```text
Ran 5 tests in 0.068s
OK
instances=8 filters=72 replay=True lane_admitted=False
```

## Disposition

```text
REJECT_FIXED_COORDINATE_FILTRATION_ONLY__R82_CANONICAL_SOURCE_PER_ATTAINED_TARGET__X_Y_HASH_BUCKETS_M2_M4_M8__NO_ACTUAL_FILTER_ADDITIVE__NO_SINGLE_BUCKET_OR_TARGET_OFFSET_COMPLETE__ALL_BUCKET_REPLAY_EXACT__CONSTANT_RELATION_ENTROPY_RESTORES_FILTER_EXPONENT__DIVISIBLE_ORDER_QUOTIENT_POSITIVE_CONTROL__EXACT_S3_AUXILIARY_CHAIN__NO_SUBCAP_CONSTRUCTOR__NO_FACTOR_LOGS__NO_DESCENT__NO_SHOUP_CLAIM__NO_BREAKTHROUGH
```

## Exactly one next action

Construct or refute one P1510-style marked-resultant source section for the
R82 `2A+3C` versus `3A+2C` split. It must cover every attained target without
bucket replay, return one jointly coupled atom source, and receive full
coefficient, state, fresh-query, rank, factor-log, and identical-descent
receipts.
