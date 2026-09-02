# P1436 autoresearch focus harness V72 result

Date: 2026-07-29

## Result

V72 binds R123 as the 59th closed frontier lane and routes the highest
priority action to
`s15_nonrepresented_fourier_resultant_torus_c5_source_circuit`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R123 verifies two exact realizations of the R121/R122 multiplicative torus
C5 membership predicate. For multiplicative characters
`chi_j(z)=z^j`, the ordered five-product count is

```text
c5(y) = q^-1 sum_j (sum_i z_i^j)^5 y^-j.
```

The characters require no field discrete logarithm. Full inversion,
however, requires `q=B^5` modes. On all eight R82 controls, the ordered
moments equal the occurrence-weighted distinct-support moments, and their
Berlekamp-Massey order equals the full C5 support. Under the iid model that
state is `B^(15/4)`.

The ordered product polynomial also satisfies

```text
P_(a+b)(Y) = Res_X(P_a(X), X^deg(P_b) P_b(Y/X)).
```

A synthetic order-11 subgroup in `F_353^*` verifies all 11 Fourier counts,
positive and empty membership, and equality between the direct ordered P5
evaluation and the P2|P3 product resultant. The represented P3 setup and a
target-scaled P2|P3 query each have `B^(9/4)` degree/state, while symbolic
P5 has `B^(15/4)` output.

These are representation audits, not a general arithmetic-circuit,
cell-probe, or data-structure lower bound. A target-specialized
nonrepresented scalar zero-test with a reverse five-source adjoint remains
open. No Shoup improvement or ECDLP breakthrough is claimed.

## Verification

- Focused R123 tests: 10 passed.
- Full harness tests: 89 passed.
- Full ECDLP suite: 463 passed.
- R123 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R123: 48 receipts, 837 bindings, 0 mismatches.
- V72 frontier preflights: 59 provided, 59 closed.
- Structured breakthrough and Shoup-improvement claims in R123: false.
- V72 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R123 producer: `fb18e752bac3b39bef9d7aa235decdb5913d7de8ff39c6c05c661a54951ba326`
- R123 report: `9d7c4bf836e106202526233038409bdbbae704632df16b4da84453cc05eb2870`
- R123 frozen interface: `f836bb2301cb472197aa26705e2bc8edd6f67d98447a718dffc2f4f935caf9f4`
- R123 cost ledger: `cd53e6948d00d5befa7598e6f5b4f1fdbc1295c6182f979e3a68ec2652dce72a`
- R123 source replay: `efb3425ffada1ceab062a3667688833e9f05c06c6dbba011b44afc9b43dff06b`
- R123 controls: `c3ab2d7ed113d231df174f689ac7efca353e0016c0e091bbc67e5d79e97912d4`
- R123 logs/descent: `1a967d9b7fb6b9e10563bf242d390cff83b6ee5f238b7bd453ab9a2f4b1623a7`
- R123 tests: `c4363e594a487828e03fa9b5c8c61aefad5ceb9ce099fe45b9a91f88bceb5a81`
- R123 gate: `1b0711d92aeee1a1a4bb2208cdbc772d8a07e1dff14b423803e75f1bd859733f`
- R123 parent: `23bb3ca35ea47189f1e5436d8caaec09cb5ace3c6ced214db9046103b1068bea`
- harness: `2c43ec205c9d099e33bd76360064cad0744d0b09895a4393f4eb55734387a786`
- harness tests: `2fe6f9e786203f2d95dc8c6ecf2fd4256a2b81e6b2c61c06e2fe6f0678617d6a`
- report: `f50b320182aeaaf8b6d512b4203c3f60e0d5872e0e2e5588dd5d65e022a390d1`
- note: `06e76a68ef7a33edca5924016177c7296eb6d44577c6a927d0b080959dd8d058`
- inventory: `d9183cd3427196324adea23c0b7dcd9341bc08d5f34d072997660e09c15240b4`
- replay plan: `8d2a8d91b69baa720e58cafaed0ff1ad10adc4605bf330a4a985b7ba3f642e96`
