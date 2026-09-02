# P1436 autoresearch focus harness V65 result

Date: 2026-07-29

## Result

V65 binds R116 as the 52nd closed frontier lane and routes the highest
priority action to
`s8_m6_target_batched_c3_pair_sum_elliptic_transpose`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R116 gives the exact missing interface behind R115. At

```text
m=6, alpha=1/12, beta=3/4,
```

the `3F | 3F` self-convolution coefficient equals an `A6`-weighted batch of
`C3+C3` pair-sum coefficients. Persistent `C3` state has exponent `9/4`,
the `A6` target batch has exponent `1/2`, and the fresh `5/4` cap leaves
average pair-query work `3/4`.

Three finite prime-order projective controls give identical direct
`A6*C6`, batched `A6*(C3*C3)`, and `(A3*C3)^2` counts. Positive rows replay
one `A6` plus two `C3` occurrence backpointers as six coupled `F=A+C`
factors; a labeled semantic subdeck supplies an exact empty-answer control.
No finite enumeration receives asymptotic credit.

The current Dinur--Golovnev indexing curve at the online-compatible
`delta=1/3` point requires setup exponent `39/8`; setup compliance would
require `delta>=3/2`, outside its range. The 2026 preprocessed-unknown-
universe route has `n^2=B^(9/2)` preprocessing and minimum state
`B^(15/4)`. Its integer residue and FFT maps do not transfer to public
prime-order elliptic encodings without unavailable DLP labels.

Standard explicit elliptic routes also miss: a target-by-target translated
`C3` divisor batch costs `B^(11/4)`, a materialized `C3+C3` resultant has
occurrence degree `B^(9/2)`, and a dense group transform has `B^5` modes.
These are scoped route failures, not arithmetic-circuit or data-structure
lower bounds. The surviving question is a jointly transposed elliptic
coefficient functional with an exact source adjoint.

No Shoup improvement or ECDLP breakthrough is claimed.

## Verification

- Focused R116 tests: 8 passed.
- Full harness tests: 82 passed.
- Full ECDLP suite: 392 passed.
- Producer, harness, and focused tests compile.
- R116 clean rerun: all five JSON outputs byte-identical.
- Parent audit R76-R116: 41 receipts, 676 bindings, 0 mismatches.
- V65 frontier preflights: 52 provided, 52 closed.
- Structured true breakthrough/Shoup flags across R116 and V65: 0.
- Promotion: withheld; natural below-rho cells: 0.

## Hashes

- R116 producer: `dcaf00bf51a2a8757a12e407cbe581f45f58c54895623f857d996898a5835f46`
- R116 report: `9c5ebb9e99eaada2296c3d4f0fcbdb472995f3ae071bb68138e364e266f93d25`
- R116 frozen interface: `d2e8038408ddc46223b2fb725403791ef448b2ae51b60c0076b0d39c84ae1302`
- R116 cost ledger: `f128638edd6d6bbe523e15f47a07bed6b792860bbd3ea3fa1bcf3b3544e0421d`
- R116 source replay: `eba70be7b0ca9829fb18c9646e5a10c303b8d22452e3d0098633d43fe7400ae1`
- R116 gate: `1a17b8e396affe9ced0a5d589059ec6d3d000ee23f56522e1f55e353bac72542`
- R116 parent: `50d5439738e7ba815ddb7bab07fb8d03000bb9a2af42b5e99bbca9009e08760b`
- 2026 source PDF: `6c676ae909461219b8d2f4480225aade0ab46633c989e6249c980dae79c203ca`
- harness: `4e751515bf00415d7702909e56c60df7656ab2c72662897ec1c81f159352f242`
- harness tests: `0ced19ee76ab1a7c39dc5a88854b1ac46a8ab233238cde30e60b0da0474e0427`
- report: `806609b51852d067afd2d3cf3124ccf5c7f487ddf676022b37bdeade254169a5`
- note: `54af253b882019f1c420fe2390224887536d2dd79bd8e523c28bdad7182c4359`
- inventory: `59ad9522277998269e83e52612d15b50573c97343eea3be9504383b5979c5e64`
- replay plan: `9623688a00d8e46658111178a33226a1486931a37cbb6fde5c03a4ee8fedfb36`
