# P1436 autoresearch focus harness V68 result

Date: 2026-07-29

## Result

V68 binds R119 as the 55th closed frontier lane and routes the highest
priority action to
`s11_m6_suboutput_implicit_c5_membership_source_circuit`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

For an iid uniform deck `C_1,...,C_n` in a prime cyclic group of order
`q>5`, two distinct canonical five-source multiplicity vectors collide with
probability exactly `1/q`. With `M=binom(n+4,5)`, the expected number of
colliding source pairs is `binom(M,2)/q`, and support deficiency is at most
that collision count. Hence

```text
Pr[|5C| < (1-epsilon)M] <= (M-1)/(2 epsilon q).
```

At `n=B^(3/4+o(1))` and `q=B^(5+o(1))`, this gives
`|5C|=B^(15/4+o(1))` with probability `1-o(1)`. The theorem is restricted to
the iid random-deck model; no asymptotic transfer to the filtered R82 hash
process is claimed.

An exhaustive `q=11,n=3` control matches the exact pair-collision theorem
over all 1,331 decks. All eight finite R82 projective decks have injective
canonical `C5` endpoint maps, exact fixed-sign source replay, and exact
empty-target rejection without consuming scalar labels. Finite enumeration
receives no asymptotic credit.

Explicit endpoint dictionaries, radical/source selectors, and output-linear
image compilers require `B^(15/4+o(1))` random-deck state or output and exceed
the `B^(9/4+o(1))` setup cap. This does not lower-bound a sub-output nonlinear
circuit or data structure.

The surviving interface is a target-specialized exact five-sum
membership/source circuit with at most `B^(9/4+o(1))` persistent state and
polylogarithmic query work. No Shoup improvement or ECDLP breakthrough is
claimed.

## Verification

- Focused R119 tests: 8 passed.
- Full harness tests: 85 passed.
- Full ECDLP suite: 419 passed.
- R119 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun: note byte-identical; three JSON outputs identical
  after removing their `generated_at` timestamps.
- Parent audit R76-R119: 44 receipts, 739 bindings, 0 mismatches.
- V68 frontier preflights: 55 provided, 55 closed.
- Structured true breakthrough/Shoup flags across R119 and V68: 0.
- Promotion: withheld; natural below-rho cells: 0.

## Hashes

- R119 producer: `99bd176b07eb119edad2ddbb6a2ff890b1109f1ef0a8cc26af061279e7bca578`
- R119 report: `1ff46d9c7a66ade7e617fc5523bf69d125ba1a102cc360b3cdee2ed0df1f75a9`
- R119 frozen interface: `9ba25b63d0dfb3dd2a4be31aeb3b2d0fba3624e5b3d3fcdfdd2c4be9b2abccda`
- R119 cost ledger: `933a3ad858ddb68b89a4328a2d404bca4df3a20b21fbac9f41a3514f75308c5c`
- R119 source replay: `408066047d2401dce4f57382b3231cda04667fd80599e65a5aa106b8c2af867e`
- R119 controls: `06df2f774602cf9f467cfd08db68d4df73edb227cb3e55aa12914d8d0132afae`
- R119 logs/descent: `e1b6a3e891b8ad5e1ce636326992ed5d0a9fa0a0c9ab6d4bda1c0971a1b0ed3a`
- R119 tests: `751c167df2c39c6122828584fc8aaf1c8ac98215cf576d1d506e60ca3d3159d1`
- R119 gate: `941e375918c4acd1be8293fcc40666879b4f5178b4454e28c21487cb9be9a9e9`
- R119 parent: `f1aceb3511e275dcfd6b3d43d76d24eab3de78b61ff42de7e5e11aee00645eea`
- harness: `3d06a1e66063d45995f7bff0263a3194be4df39cbc43132c048a950577854fdf`
- harness tests: `01501a624f5d11b24dae909de8da6e83772818e4254f9f98d9ce1f73ceb6b5d5`
- report: `9016511717ed433dbe5e38765862559566f90e0774bb48b8fada142152f4627b`
- note: `4e3bab4f895730c3747e7447171cb1aef9d60cd0762033d280ef890a8e1d6598`
- inventory: `f241f4b208737d2fcbd91c0c209d0eb163f1cfd247d2eb9d3caa8a4f29b57098`
- replay plan: `99554d95f14d9ec237a1af10b29e3ff5c2eb502aadbe50ae9c7144743f66d4a0`
