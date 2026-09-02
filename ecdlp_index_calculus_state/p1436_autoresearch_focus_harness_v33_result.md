# P1436 Autoresearch Focus Harness V33 Result

## Claim boundary

No generic-prime-field ECDLP speedup, Pollard-rho improvement, Shoup-bound
improvement, or algorithmic breakthrough is claimed.

## R83: coordinate-filter entropy replay

R83 freezes one canonical `5A+5C` source per attained R82 target and tests 72
x-coordinate, y-coordinate, and encoding-hash bucket profiles, including
target-coupled offsets.

```text
actual additive filters                         0/72,
single complete bucket or target offset         0/72,
all-bucket/offset replay exact                 72/72,
best static survival fraction          0.1480..0.7063,
best coupled survival fraction         0.1270..0.6111.
```

The divisible-order positive control
`Z/808Z -> Z/8Z` passes all 4,096 homomorphism checks. Prime-order EC
coordinate buckets do not. At `B^5=N`, reducing the explicit `B^2.6` join to
`B^1.25` per filter requires `B^1.35` exact replay and restores `B^2.6`
total work; the generic `B^2.5` route similarly restores rho scale.

The FFE interface is exact but remains verifier-side evidence:

```text
supplied sources                         512,
regular Semaev S3 chain checks         4,608,
group or S3 failures                       0.
```

Report SHA-256:

`1478cdf21493ffbeaed0859af849ea6f2835027f23db7e3e4c008f3f24db500c`

Gate SHA-256:

`7907df5232c7a6322c797ec33b7600042e223c2c21cb5c832b285995a77fafdf`

Parent receipt SHA-256:

`4ecfc30a08c258d70a3cd87137a8cd6f3104c170b27cfd7b425f8a893a5a677e`

## Harness routing

V33 report:

`ecdlp_index_calculus_state/p1436_autoresearch_focus_report_seed1432001_exact_v33.json`

SHA-256:

`56ba53f7efe44b222445ef2af09be1ba3cc8d0a013a74f2d8767d32ec51fdbce`

Schema: `ecdlp.p1436_autoresearch_focus_report.v25`.

Nineteen hash-bound lanes are closed. The top frontier is:

```text
s6_5a5c_marked_resultant_source_section
```

The next experiment must freeze the `2A+3C | 3A+2C` split algebras, marked
variables, elimination order, containment proof, and source selector. It
must cover every accepted target without bucket replay, return one jointly
coupled atom source, fit the direct caps, and complete rank, factor logs, and
identical target descent.

V33 remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`. The active objective is
unchanged.

## Artifact hashes

```text
harness              d63b4711d730fdd6881e3e65136e86124a2f78824099a99e62f3f076ab09aed5
harness tests        eb436370b2894984bcc6fc88a5348a002e4b0270bd293255f5a7d25129e188d8
V33 focus note       82f4426c741719be2913e9ce4c93fc97912c8b1cd4e5b28d8914d3a1bc395ab4
V33 FFE inventory    4120a34e1f7435e626808e5d0c05e4f86d82a343220b3ef559801854e13b6727
V33 FFE replay       da71c77fb546a5dc1159d3d1de19107bec1b03312feb0cd103430d64c77b350d
```

## Verification

- R83 targeted tests: 5 passed.
- Focus-harness tests: 49 passed.
- Full ECDLP task suite: 142 tests passed.
- Nineteen R69-R83 and harness Python modules compiled.
- Eight R76-R83 parent YAML receipts parsed; all 85 declared input/artifact
  hashes and all eight archived-gate bindings matched.
- All eight parent receipts have `breakthrough=false`.
- A clean R83 rerun reproduced all six generated JSON artifacts byte for
  byte.
- V33 has 19 closed lanes, `promotion_allowed=false`, and the expected top
  marked-resultant action.
- `git diff --check` and explicit trailing-whitespace checks passed after
  final freeze.
