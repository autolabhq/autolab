# P1436 autoresearch focus harness V62 result

Date: 2026-07-29

## Result

V62 binds R113 as the 49th closed frontier lane and routes the highest
priority action to `s6_5a5c_transposed_nonuniform_c5_leaf_generator`. The
report remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R113 proves that the canonical `C3` endpoint sets are not fixed translation
orbits: each is a nonempty proper subset of a prime-order group, hence has
trivial translation stabilizer, and every canonical ordering has nonconstant
successor differences. Any nonzero fixed translation instead has full group
orbit length `B^(5+o(1))`.

The exact nonlinear prefix product has order one and one scalar of state, but
still evaluates all `B^(3+o(1))` `C5` membership leaves. Standard product
trees retain the same leaf/state exponent. Precompiling all target-independent
zeros restores `B^(5+o(1))` occurrence degree; exact toy support occupies at
least 99.4 percent of the source body.

This closes fixed-translation orbits, standard product trees, and bounded
prefix state as a low-work claim only. Arbitrary variable-coefficient and
transposed evaluators remain outside the negative.

No rank, factor-log, target-descent, Shoup, or breakthrough gate passes.

## Verification

- Targeted R113 plus harness tests: 86 passed.
- Full ECDLP suite: 366 passed.
- Producer and harness files compile.
- R113 clean rerun: all six JSON outputs byte-identical.
- Parent audit R76-R113: 38 receipts, 617 bindings, 0 mismatches.
- V62 frontier preflights: 49 provided, 49 closed.
- Structured true breakthrough/Shoup flags: 0.
- Promotion: withheld; natural below-rho cells: 0.

## Hashes

- R113 producer: `126a2405ae94aa01143ecdf88927f7f4fedb95922aeb7539b467516c97979e1c`
- R113 report: `5b961649fa71ced6b049a3500316114b3c9f3806b5b61639471d9a6f86b01102`
- R113 source replay: `ef6bcec03bde76982453f5d508f6cdf9b6f32ba9ea1bca36ce143715b9dbb3df`
- R113 parent: `9945ff193a3fd2cc7022e27e68ddf5cf6e40d0ab5301a9450433fe8c0128f129`
- harness: `fa584eb2433c85a3475f96c02f8b2faaa730cc5fec2cbeca5271d3ca9ecd821a`
- harness tests: `a0efea381a2d42bedbd90faf1a0d0f8afe6471e7b5baf1b9a51d36dfe5983476`
- report: `52ade62f9a268da57d60ac432e5dfc7ee763e75d19aa1b820d59fa418f4a327a`
- note: `970f4d0bdb661c1e1df9eb3480e05b5b2a2c7ce5dae2e0630ab66a1a7b0aee38`
- inventory: `3aa81bbf13ad052b476e87363b828aa90711a9c361d84c7b2027e5dffa0ef864`
- replay plan: `8d247f9d59533dc5ebe83229a3a185046652343c43911dab89bbe49269e6c1d1`
