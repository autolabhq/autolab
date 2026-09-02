# P1436 autoresearch focus harness V112 result

Date: 2026-08-01

## Result

V112 binds R163 as the 99th closed frontier lane and routes the highest
priority action to `s55_unlabeled_aggregate_union_factor_constructor`. The
report remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R163 couples R161's x and signed-y residuals before aggregation. For target
`j`, the Fermat projector

```text
e_j = (1-a_j^(p-1))(1-b_j^(p-1))
```

is one exactly when both residuals vanish at the same C3 endpoint. Hence

```text
G = gcd(U, product_j (1-e_j))
```

is exactly the union of left C3 endpoints participating in any target
decomposition, with denominator-exception roots checked directly. Separate
products of x and y residuals are invalid because they can vanish from
different targets.

If `K=B^(3/4+o(1))` targets have unique positive-C6 sources, then
`deg(G)<=20K=B^(3/4+o(1))`. Once `G` is available, factor it and scan all
`N=B^(5/4)` targets for every root `P`; hashing `T_j-P` in the persistent C3
dictionary recovers the target label, both C3 backpointers, and the C6 source
in total `B^2` work. This is below rho's `B^(5/2)` by exponent `B^(1/2)`.

The six public controls recover every union factor, expected source, target
label, and public point identity, and reject the empty targets. A separate
`Q=-2P`, `T=-P` control verifies a genuine positive denominator-exception
match with `x(T)=x(P)`.

The finite producer constructs `G` by endpoint enumeration and receives no
attack credit. No below-rho aggregate-union constructor, unconditional
generic-prime algorithm, Pollard-rho improvement, Shoup improvement, or
ECDLP breakthrough is claimed.

## Verification

- Focused R163 tests: 16 passed.
- Full harness tests: 129 passed.
- Full ECDLP suite: 966 passed in 123.787 seconds.
- R163 clean replay: all six JSON outputs byte-identical.
- V112 clean replay: note byte-identical; three JSON outputs identical after
  removing `generated_at` timestamps.
- Parent audit R76-R163: 88 receipts, 1,667 bindings, 0 mismatches, no missing
  paths, and no missing or duplicate rounds.
- V112 frontier preflights: 99 provided, 99 closed.
- R163 obligations: 20 of 27 passed; lane admission false.
- R163 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V112 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R163 producer: `225108ac4f89b44c88db5db93aa3c2c1b4578c4b0993f7f736781b469b3d77f9`
- R163 report: `0a35b4fee30c7abf4ba69232e0f7af65d48ec785aef832c934bf03d5366645ab`
- R163 frozen interface: `2971fc80ba63eb35303f35592dab57c91d532f31886cfde21b632ef04626c8d7`
- R163 cost ledger: `28f090f312d5177d4f7d4132e5b772b56b42ff372c7866db23e227b3888e4722`
- R163 replay: `83fa1b944f0b336e8c81c3e166ff0ce94f18949201d93f0b440af87df58e655e`
- R163 controls: `dc485b0fd593aadd11db178fd2c2edc5510fc443dca928fcc58627a818c4d5ff`
- R163 labels/backpointers: `815fbd1a3e64a6f203a34541c5a29c407553144b934b56a5641e38ec4a15b9ee`
- R163 tests: `6f829d95211f8d4b2785e6392319435e6a651a41028de506b2cfadbce621058a`
- R163 gate: `3064a108bf063ab59a0991be6910b70bf3a2e498974add780f6a57e157a63212`
- R163 parent: `cbaf7edfdf8ca8082138be914b6af923d913be75b61b44cf3da774584207f90e`
- Harness: `30a4537f7f9404e513ea4686a6ddf0693c0009d80a431dd03de20ee76bbd9dac`
- Harness tests: `c6bf42cab6f537af39b036e9913855c1160a5e8e7c5feded94155b430fa5a3e9`
- Report: `5acda999099f14ff64b5e505261227cc528515823f0bff9d21ad750895321548`
- Note: `d7ce0b0c795aa2339417fd1780b313c07112ecf934ddc77f5af7b96a16a973f8`
- Inventory: `9bc775ec150d3608d74f27f423e398bf7cdf7f3fbd2bf6fa93d22addc2abc35b`
- Replay plan: `799cd028effd20b7abfc452975598f1a37e2e0b71e4bfae2ba22489e120f2761`
