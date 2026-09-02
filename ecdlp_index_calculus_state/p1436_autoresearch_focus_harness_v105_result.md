# P1436 autoresearch focus harness V105 result

Date: 2026-07-29

## Result

V105 binds R156 as the 92nd closed frontier lane and routes the highest
priority action to
`s48_projective_singleton_direct_rank_reverse_ffe`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R156 quotients the exact opposite-row symmetry from R155 by projectively
normalizing each nonzero signed singleton row. This identifies a row and
its negative without changing the row span or rank. The signed A6
coefficient support grows as `Theta(r^6)` for `r` independent A inversion
pairs, matching the selected `B^(1/12)` A-pair scaling.

The preregistered finite grid contains 108 controls:

```text
A inversion-pair counts: 2, 3, 4
C inversion-pair counts: 5, 6, 7, 8
logarithmic factors:      2, 4, 8
seeds:                    15601, 15602, 15603
```

Every control replays the relation and opposite-row identities exactly.
The modulus is fixed before labels and ranks from the maximum A6 support,
maximum C6 support, and the occupancy multiplier
`ceil(log_factor * ln(C_pair_count))`. Rank deficits are split exactly
into uncovered columns, insufficient projectively distinct rows, and
residual linear dependency.

Full rank occurs in 62 of 108 controls. Counts by A-pair count are
`[18, 22, 22]`, so extra A diversity gives only a modest finite gain and
then saturates on this grid. Counts by logarithmic factor are
`[6, 21, 35]` out of 36; only one factor-eight control fails. This is
finite evidence that logarithmic relation supply is the dominant
variable after the projective quotient. It is not an asymptotic rank
theorem and does not transfer the independent sparse-row comparator to
the correlated convolution family.

R156 supplies no hash-to-curve rank transfer, reverse signed marker FFE
operator, candidate factor logs, or identical target descent. It consumes
no candidate DLP, root, count, marginal, rank, or source oracle. No
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R156 tests: 13 passed.
- Full harness tests: 122 passed.
- Full ECDLP suite: 849 passed in 87.788 seconds.
- R156 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R156: 81 receipts, 1,515 bindings, 0 mismatches.
- V105 frontier preflights: 92 provided, 92 closed.
- R156 obligations: 15 of 26 passed; lane admission false.
- R156 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V105 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R156 producer: `b9caa2eb7bab40d3c3907111d1836130f3cc69394b0489445177996532f4a22a`
- R156 report: `8c04aa3e5640f661d9d2c30e24dcd769dfc4fa6b07c3947f57a83f75965414f3`
- R156 frozen interface: `feb947fe5fcef1b7c016f11c1cd978882ab87358bbb855ac846f2934a4a41088`
- R156 cost ledger: `db4ed01dace9c597b38686180406fc9392d7eedce1f09dd3b92e639e1d395f2a`
- R156 source replay: `eeb66740bcd2ad794082e780fcf0f80f1c23a4a40e15c0bc1a6dfe9ba99b4b4e`
- R156 controls: `82e745fb615edad055b00e24feb77f02ee45f33712e83ffe04e80e3f4c12d3ad`
- R156 logs/descent: `fd84ab20b0d1fcee454e75c80f49fc7cf9d9b9dcf9f87e8eb6ece939ec18bc45`
- R156 tests: `f67774f461f02dea8b454467c8398e786991095fc3b1a5c8b995ff62fd58a99e`
- R156 gate: `0d47ca3ee6534eb38287f8a611612d594d96f710c0b6cac675fdd2d7bc619bc1`
- R156 parent: `dbcc1e54ee782a39380b28dead9ae770c161be054f752ca687d79e295d4dde4d`
- Harness: `cdbbb8541860e4fcbd15d605548bf8e85ea5476ad519b7652378d55468df99f9`
- Harness tests: `c45694c3bbd8771fc5d194787f8004cf42d18a3a0a53fa4cb87ee3fab66fafbd`
- Report: `1cbc8dbe67484e4f67aee698e95eedc467136fcdf280a24f027f41af52cea17d`
- Note: `beba6c77394f10c95b0a3c1f96c559e9f9fd44941b68f46ac330f4d145dc8dd2`
- Inventory: `26cbd390f9bfde77d6c07578c988de4b8ec8b33ab2106b114f6fea056180562c`
- Replay plan: `a488fa9c16c6080e3eea57e97d71ad81a4dde29b511f21c7ddea428b732ee48b`
