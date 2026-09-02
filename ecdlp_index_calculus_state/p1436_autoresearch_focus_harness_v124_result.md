# P1436 autoresearch harness V124 result

Date: 2026-08-01

## Status

`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`

V124 binds R175 as the 111th closed frontier lane and routes the first
experiment to `s67_reusable_scalar_subset_incidence_oracle` at priority 312.

## R175 Result

Let `A(P)` be the exact tangent-aware signed aggregate constructed in R174 for
selected leaf `P`. For every nonempty selected subset `S`, define

```text
Sigma(S) = product_{P in S} A(P) in F_p.
```

Because `F_p` is an integral domain, `Sigma(S)=0` exactly when `S` contains an
R174 candidate. A balanced binary tree can therefore recover all candidates by
stopping below nonzero nodes and querying both children of every zero node.

For `K` candidates among `n` leaves, the query count is at most

```text
1 + 2 K ceil(log2(n)).
```

At each depth the queried subsets are disjoint, so total queried subset volume
is at most

```text
n (1 + ceil(log2(n))) = softly O(n).
```

The six controls make 358 exact scalar queries against a bound of 1,622. They
query subset volume 1,238 against a bound of 1,374 and recover all 140 R174
roots. All 358 zero-product biconditionals are exact: 316 queries are zero and
42 are nonzero.

Each queried node is bound to a compact descriptor

```text
U_S = product_{P=(x,y) in S} (X-x),
V_S = V mod U_S.
```

All `V_S` remainders interpolate their node endpoints. The six transcripts
bind 2,834 queried descriptor coefficient slots. Full leaf enumeration is used
only to verify the finite controls and receives no asymptotic credit.

## Conditional Oracle

R175 isolates the missing primitive. After one softly `O(n+N)` preprocessing
of the compact selected divisor and the `N` target points, a reusable oracle
must accept any balanced-node `U_S,V_S` and return its exact R174 signed subset
product in softly

```text
O(|S| + N)
```

work. It must preserve the secant divided-difference and geometric-tangent
charts, avoid target-dependent per-query setup, and avoid candidate-dependent
inversions.

Given that oracle and the R163 charged candidate-output contract
`K=B^(3/4)`, balanced recovery would cost

```text
softly O(n + K N) = B^(9/4+o(1)),
```

and R163 target-label and source-backpointer recovery costs `B^2`. This
conditional envelope is below rho, but the reusable oracle is not supplied and
receives no attack credit.

## Cost Boundary

```text
selected divisor degree n:                     B^(9/4)
target count N:                                B^(5/4)
R163 charged candidate output K:               B^(3/4)
conditional preprocessing:                     B^(9/4)
conditional subset-volume work:                B^(9/4)
conditional per-query target overhead K N:     B^2
conditional oracle total:                      B^(9/4)
R163 label/backpointer postprocessing:          B^2
direct expanded tree factors n^2 N:             B^(23/4)
represented target dual-Chow body N^2:          B^(5/2)
represented selected-pair queries n^2:          B^(9/2)
rho proxy:                                      B^(5/2)
```

The available direct implementation expands all target factors across the
selected-pair grid and costs `B^(23/4)`. Representing the target dual-Chow form
already reaches the rho exponent. R160 and Shoup also exclude credit for an
encoding-invariant generic locator; any admissible implementation must exploit
the explicit prime-field coordinate representation and charge its conversion
and preprocessing.

R175 passes 15 of 22 obligations. It admits the scalar reduction, exact
balanced recovery, compact node descriptors, and the conditional cost
consequence only. Oracle admission, lane admission, rho improvement, Shoup
improvement, and breakthrough flags remain false.

## V124 Routing

- Harness schema: `ecdlp.p1436_autoresearch_focus_report.v111`.
- Bound and closed frontier lanes: 111.
- First focus: `s67_reusable_scalar_subset_incidence_oracle`.
- Natural full-rank, verified-log, and below-rho cells: 0 of 1.
- Promotion allowed: false.

## Verification

- R175 focused tests: 16 passed in 29.416 seconds.
- Harness tests: 141 passed in 1.017 seconds.
- Full ECDLP suite: 1,164 passed in 611.833 seconds.
- R175 clean replay: all six JSON outputs byte-identical.
- V124 clean replay: note byte-identical; three JSON outputs equal after
  removing only top-level `generated_at`.
- Parent audit R76-R175: 100 receipts, 1,949 recursive path/hash bindings, zero
  mismatches, missing paths or rounds, and duplicate rounds.

## R175 Hashes

- Producer: `f525ecc47496485919b4f2397b505842eac43ea7c1b53b90c6beaff4fa07d536`
- Report: `5541b18b9ce670ca95902e75cf9fc14c2a35f9725c9ee756c0502963cbafc543`
- Frozen interface: `1710f98bdb7d29d4c016abb0272915d6fdcdf471d35a6e7f79941b22cc1be7ad`
- Cost ledger: `0c8790183f1d05593308421e1cce0eabec21c3ee52fef2ed3c4f1dc84f0e5818`
- Replay: `6eaf07b4a9507ca96681c85f79b8d33763033d7bd71faae5e8417987969f084f`
- Controls: `1ff598263a892c4a8d87c04799b66383b671ffca9ecdb05c005c45219447fb2d`
- Tree ledger: `cff099668231e6b239bc1a651492938affc09c8531e61489b7384dff48258c00`
- Tests: `9cab6bb5fe9baa27d1d16e2874a3ceb782d0b93f799ce04befbd2cfb8472704f`
- Gate: `ccad60c12a8951476915c864d249135a369d0832a0a29704ef1cbd445739d3a1`
- Parent: `e46cbd385ba78e5e14fd10dd70414b3a990a94493b8d32064258c362e56399c8`

## V124 Hashes

- Harness: `c7e9126dc7f5d7024a36c6a9f7d7504d0f478d874b23c6751050efbae45261dc`
- Harness tests: `2bb8f8268cfec5b909180c1bacacd63b979cb45f54925734b9bb4d397d7f3a8b`
- Focus report: `8f3f55251502f27784fd27abd36dcde2985428de659374288cffc5ab671c9d02`
- Note: `664d659e9ff477853675ea696810a3eb58b2848d6dc93f127873be6cfaba7c2e`
- Evidence inventory: `e6a03a7f62c7197962d3800e14cdcce007b24d525833668a34f68adba1ca74c5`
- Replay plan: `66c0698dacebe8d86f8cb0a706e07fbb683af7529e527eb1ff3bcfdceebefff6`

## Claim Boundary

No generic-prime ECDLP algorithm, Pollard-rho improvement, Shoup lower-bound
improvement, or breakthrough was produced. Exact identities, finite controls,
root recovery, a verifier pass, and a conditional oracle envelope receive no
unconditional asymptotic attack credit.

## Next Action

Construct or refute a reusable arithmetic DAG that preprocesses `U,V` and the
`N` target points once and evaluates every balanced-node signed subset product
in softly `O(|S|+N)` work. Reject leaf enumeration, represented `N^2` or `n^2`
Chow bodies, `nN` per-node factor expansion, target-dependent transforms,
candidate inversions, and unit-cost multipoint, norm, resultant, root, count,
marginal, rank, source, DLP, or generic locator oracles.
