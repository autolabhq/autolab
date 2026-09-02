# P1436 autoresearch focus harness V71 result

Date: 2026-07-29

## Result

V71 binds R122 as the 58th closed frontier lane and routes the highest
priority action to
`s14_target_specialized_nonoccurrence_torus_c5_source_circuit`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R122 re-optimizes every fixed relation arity under the explicit stored
`C_s` and enumerated `C_r` grammar. With `s+r=m-1`, density retry `delta`,
deck exponents `alpha<=beta`, and `B^beta` meaningful rows, the charged
relation-collection exponent is

```text
L = delta+m*alpha+(r+2)*beta
  = (delta+m(alpha+beta))+(1-s)*beta.
```

Relation supply gives the first parenthesis at least 5, while the
`B^(9/4)` setup cap gives `s*beta<=9/4`. Hence

```text
L >= 11/4+beta > 5/2
```

for every fixed arity and polynomially nonempty meaningful C deck. At the
actual R115/R121 vertex, the exact C3|C2 split costs `B^(11/4)` fresh work
and `B^(7/2)` relation collection.

For iid random decks, every setup-eligible stored `C_s` support remains at
occurrence scale with high probability. This follows from the exact `1/q`
collision theorem and `s*beta<=9/4<5`. Exhaustive `q=11,n=3` controls verify
the collision theorem for source arities one through four.

The theorem covers explicit occurrence/output tables and their iid-distinct
support analogues only. It does not transfer to every filtered deck and is
not a general circuit, cell-probe, or RAM lower bound. A target-specialized
nonoccurrence torus C5 circuit remains open. No Shoup improvement or ECDLP
breakthrough is claimed.

## Verification

- Focused R122 tests: 9 passed.
- Full harness tests: 88 passed.
- Full ECDLP suite: 452 passed.
- R122 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R122: 47 receipts, 809 bindings, 0 mismatches.
- V71 frontier preflights: 58 provided, 58 closed.
- Structured true breakthrough/Shoup flags across R122 and V71: 0.
- Promotion: withheld; natural below-rho cells: 0.

## Hashes

- R122 producer: `053aa87da1e5443965b6141b34c2bc7a71b358d8c7bfd01658f1235726091185`
- R122 report: `e7725ca09099cd1945af08959d381679e3ea905a6b6fd7f5f7085116f0ab949a`
- R122 frozen interface: `1f74ae158c4ef1070b1d87fe59a6d815a6b09c558d857c6f229cd5e665937de4`
- R122 cost ledger: `18027809e82ddac7c6d848b638da79b231ff5fc1dbf425766c033bc829cc6720`
- R122 source replay: `7e7267fc053e064b654075fcdda408880562e8db03a30e70ee2bdbbbe646f7a3`
- R122 controls: `d055d78dcfc6deb9427195bf567ebe0d4c61d3a4e55f5b6468e498fbe3b99a4f`
- R122 logs/descent: `d42ba1a0b0eb9365451eb4fcd95612d93becaab3dedf2ce470f377bf99096ae1`
- R122 tests: `c7b036f8410d696e1c57bc0bd052a904cab8e105a863f8102b73ea27697c142a`
- R122 gate: `337f0f24c3e6b5ee03f85990c8f8d621157951da75b01f66511917428266008d`
- R122 parent: `14732beb737e155da02ff823d88df1806ae3893228741ddacf6015f07717b95c`
- harness: `99579cc8c5ae39bc55f29f7d4016e4853a8d388d22017ab42bb93a624dd0487a`
- harness tests: `e6907f9ff47da9253ee1db0069552d2802c68be4ee6db0ab97ba2d740cb784e3`
- report: `883932ed66ac50a1ebd257b6be45e2439a83ca74d5cef49527beeab16684904f`
- note: `8b880af153b9e49cb930c5b70584903814fa0a4491fe40014afc6230ebf8f43c`
- inventory: `3e0dcda423dbde416924635510e5598858b12f1eb8e0e3bec45bc5a13e8e8912`
- replay plan: `509a53924b39352d467b974041352602e948b91d64385d8299053e6b991fd10d`
