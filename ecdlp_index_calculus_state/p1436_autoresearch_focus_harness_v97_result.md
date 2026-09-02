# P1436 autoresearch focus harness V97 result

Date: 2026-07-29

## Result

V97 binds R148 as the 84th closed frontier lane and routes the highest
priority action to `s40_structure_aware_occurrence_autocorrelation`. The
report remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R148 identifies the unweighted R147 occurrence-pair query with static
3SUM-indexing on two identical lists of length

```text
n = B^(9/4+o(1)).
```

The weighted campaign query is stronger: it requires exact integer
multiplicity and full `A/C` atom marginals.

Linear state plus the trivial scan costs `n` per target, hence

```text
B^(5/4) targets times B^(9/4) work
  = B^(7/2)
  = N^(7/10).
```

The full pair-sum table instead costs `n^2=B^(9/2)` setup and state.

The Fiat-Naor-derived published tradeoff

```text
T S^3 = soft-O(n^6)
```

instantiated at linear state gives `T=n^3` and is dominated by the trivial
scan. The Dinur-Golovnev improvement

```text
T S = soft-O(n^(5/2))
```

starts above `n^(3/2)=B^(27/8)` state and uses
`n^2=B^(9/2)` preprocessing, so neither resource fits the R115 setup cap.

These are upper-bound instantiations for generic decision or witness
indexing, not lower bounds. They leave open an operator that exploits the
occurrence divisor as a triple elliptic convolution of the compact `C`
divisor before applying generic static indexing.

No exact count/marginal index, factor logs, identical target descent,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R148 tests: 11 passed.
- Full harness tests: 114 passed.
- Full ECDLP suite: 746 passed.
- R148 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R148: 73 receipts, 1,344 bindings, 0 mismatches.
- V97 frontier preflights: 84 provided, 84 closed.
- R148 obligations: 15 of 25 passed; lane admission false.
- R148 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V97 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R148 producer: `415bbb0d00b426855886607d6cb9409dff21f5730dc605079264e0b36c87141f`
- R148 report: `b01496572b919ffd15406ee83bcd185675b96669d0cd40f51972ddf56f2caed7`
- R148 frozen interface: `071b8e2364b6ba0da5bf949b7f0051af25d558b4ea244c6a0aa31bf4f6d1b133`
- R148 cost ledger: `3161ea7199f1200b1a9a6643b2cad20f082e343f9a85ca36235a218df8418e7d`
- R148 source replay: `190281a3930976fe162232be997201694955015eea27387691aeb6aab3792511`
- R148 controls: `3bee12463c7c4966421832f1edb89673aac3f9139151489faf180298ed5db5b0`
- R148 logs/descent: `8197c71bfe22bec14b3db5ff5e594d1602229d3b02ad87520a357e5932ec6a85`
- R148 tests: `319b2b006e35a7aa9abf942980527d5172e0ca26ab6e04aab039b7b4612e28dc`
- R148 gate: `50e465e70d3e57457a64185cd9b86fc9b08bd9ebc56cfe6969f8c1f04508d9ca`
- R148 parent: `6cd74b061465a2037edb64872907866d4bad2c1a247b97a9c4e01f6332c26fda`
- GGHPV PDF: `b9161a299ee5227bdf11be0bbfec1c58a9348deb8d261875b935d573b4112785`
- Dinur-Golovnev PDF: `e56522544d9ae28ec542825fcd2e7238360a05306a79d0b757a910dda382420c`
- harness: `5591c0b6e1fdd88ec15f1a77a755157932e28caf9717023bedabb3869d7d2951`
- harness tests: `47662db1c2d3ddd4f5ce5a341feda5e0d8383446d2c4e5d93bb5ef50d775f3fd`
- report: `6ef9692f64180329a281a4adaf1c76f611ff147170a355cfe4c1358a1e2a3a8a`
- note: `e510444f02b097da4aee92b0ba257db21ded47d4e94bdd3fe95f1958bcf7b61d`
- inventory: `42eeec2b68beda6b8978ca1f211482a4479570047e83d4a93cd51230010a052d`
- replay plan: `f95dddd6e43e7ec34f1ba1721045a464adfe345b9d96e0e6e25403f820afc734`
