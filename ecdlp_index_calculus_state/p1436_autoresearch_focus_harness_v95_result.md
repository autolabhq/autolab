# P1436 autoresearch focus harness V95 result

Date: 2026-07-29

## Result

V95 binds R146 as the 82nd closed frontier lane and routes the highest
priority action to
`s38_source_equivalent_batched_count_marginal_index`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R146 proves an information boundary for the R144 count-and-marginal
interface. On a target fiber containing one canonical unordered
`A`-multiset/`C`-multiset source pair, let its ordered expansion weight be
`c_T`. Then

```text
d_A(a) = c_T r_A(a)
d_C(c) = c_T r_C(c).
```

Exact count plus full marginals therefore recovers every canonical atom
multiplicity by integer division. The source-free interface is
source-equivalent on singleton canonical fibers even though it does not
recover an irrelevant pairwise alignment.

All eight finite controls independently replay R144 counts and marginals.
Their singleton positive-fiber fractions range from `54/55` to `1`, and
every singleton marginal vector inverts exactly.

Under an explicitly independent-uniform model, the dense six-factor
canonical occupancy is Poisson with

```text
lambda = 1/(6!)^2 = 1/518400,
```

giving conditional singleton probability `0.9999990355000988`. This is
random-model evidence only and is not transferred to the structured factor
base.

For every deterministic source-to-target map, mean positive occupancy
`L=M/H` and uniform hit probability `H/q=M/(Lq)` conserve total source
density. Superconstant aggregate occupancy therefore loses the reciprocal
hit density unless a targetable structured family is constructed. This is
an accounting identity, not a computational lower bound.

No implicit count/marginal index, factor logs, identical target descent,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R146 tests: 11 passed.
- Full harness tests: 112 passed.
- Full ECDLP suite: 721 passed.
- R146 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R146: 71 receipts, 1,303 bindings, 0 mismatches.
- V95 frontier preflights: 82 provided, 82 closed.
- R146 obligations: 15 of 25 passed; lane admission false.
- R146 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V95 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R146 producer: `088ece9b3359cd80b510ccb6bd2aba7fbc2ff037978e1e54a9d8159d902034a3`
- R146 report: `6b7c1eefcb4d062bf5eaee84e88ab2f5e1fd65ba060f16c1ce4adaf0dbfb64f9`
- R146 frozen interface: `0570199c52a34c19819852c003870f6962148b1f8ff80e13326a3553665b527b`
- R146 cost ledger: `8f40955365b4fc074ed83a98dd168f8f61b747a968d7e71682e3fdaeb9e90711`
- R146 source replay: `b33844c11cb0b60562292be41d9f7e6738bc290650c64d17cd6f16dc96cb2d24`
- R146 controls: `9a2a729ce1169ca979b52047452149e551cffc2e68ea875ef819def7c322412d`
- R146 logs/descent: `cc960f475d3029958cd71bdc22b1c7c4411fe30fca31a08580eb1d6c3f8d9037`
- R146 tests: `6b23ddbffe6a3ef455c90b15628b117a4ea74aa2c9f51c390846d7bfbbd11af2`
- R146 gate: `8b0dc8312273770d5f7f44a26b89a1ce06873b20994d7445fbf8b3d55ba02216`
- R146 parent: `8c6b9f0505a3b26b3443a40f1cb38cc4e0a36f58515d5db3c510b3004e6232bb`
- harness: `a1ffef23f2fda0e45c634ce55cb425dac81f363bafab79d6e84ecbb26dc9fa8e`
- harness tests: `e52db6de0de6706c538d4ff292619ad6d2f7d247ab063e7d0964bf22264a12c8`
- report: `3765a8618a2615021ecea430927eb3c7f9de34fe2db123b7dffe9709b1f01bb8`
- note: `0d4434cd43ac7ea7c9dc11b7d5712f767fc25b96cc1f7c3fcc136a4076c39959`
- inventory: `d1e7f640e9321fc03500d571ad9e31494788ddfbcb724965a2bb6325e0cf6027`
- replay plan: `404557b15f68fb41c98871207ba3c46cbbe239c83ccf05aa512c34220db0ede9`
