# P1436 autoresearch focus harness V107 result

Date: 2026-07-29

## Result

V107 binds R158 as the 94th closed frontier lane and routes the highest
priority action to
`s50_conditioned_short_relation_full_rank_reverse_ffe_descent`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R158 refutes the global feasible-C6 coefficient-map injectivity target.
With `d=B^(3/4)`, there are `Theta(d^6)=B^(9/2)` feasible vectors in a
group of order `q=B^5`, so the expected number of collision pairs is
`Theta(d^12/q)=B^4`.

The surviving statement is near-injectivity at a fixed source. A fixed
l1-six vector is collision-bad with probability `O(B^(-1/2))`. Thus the
expected bad singleton fraction is `O(B^(-1/2))`, and Markov bounds the
fraction by `B^(-1/4)` except with probability `O(B^(-1/4))`. Each bad
source removes at most `2d` signed target rows, so this source loss removes
only an `o(1)` fraction of the complete row universe.

That complete singleton relation-row universe is exactly

```text
R_d = {r in Z^d : ||r||_1 is 5 or 7}.
```

It has size `Theta(d^7)=B^(21/4)`. For `q>84`, distinct canonical combined
A/C forms are nonproportional. Their zero events are pairwise independent,
so `O(log B)` independent A batches give
`Theta(B^(23/4)log B)` candidate forms and concentrated
`Theta(B^(3/4)log B)` relation events. The relative Chebyshev failure bound
is `O(B^(-3/4)/log B)`, while expected repeated hits on relation rows are
`O(B^(-15/4)log(B)^2)`.

For a fixed factor-base column, this argument supplies only
`Theta(log B)` expected incidents. Pairwise independence therefore proves
a vanishing uncovered fraction, not zero uncovered columns or full rank.
Exact transfer from independent cyclic labels to the conditioned
hash-to-curve sampler also remains open.

All 36 finite controls prune collision-ambiguous sources before admitting
rows. Across the repeated grid, 10,578 theoretical rows are pruned. Every
control covers all columns and has full rank. Relation-event versus
distinct-row totals for one, two, and four A batches are respectively
`185/183`, `323/320`, and `692/673`, for 24 duplicate events overall.
These explicit finite controls cost `B^(9/2+o(1))`, above the
`B^(5/2)` Pollard-rho proxy, and receive no attack credit.

No complete conditioned hash-to-curve transfer, full-rank theorem, reverse
signed FFE operator, identical target descent, generic-prime algorithm,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R158 tests: 16 passed.
- Full harness tests: 124 passed.
- Full ECDLP suite: 881 passed in 101.765 seconds.
- R158 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R158: 83 receipts, 1,553 bindings, 0 mismatches.
- V107 frontier preflights: 94 provided, 94 closed.
- R158 obligations: 16 of 26 passed; lane admission false.
- R158 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V107 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R158 producer: `c68c0a8644117133b7dfaa51564e93a44c2fa614e66c899524ae941a3673439b`
- R158 report: `bc0e0e864af69c8c03fd21b0bb360adceea7247daf940f5cc5cbf0e51ad30c6a`
- R158 frozen interface: `e278b905703403e30d223d611a36e7eb5cb857c4ef4cf50acab24a81eab8c8e5`
- R158 cost ledger: `eac09e1950a431ae71c675210eeda5bacb797bbc6aa83cc32ff7fc9b06497f22`
- R158 source replay: `055d2a9887db144eed8e4791270522aac82a4760d5fa96ed4dec4b67e9fcb968`
- R158 controls: `af9095b0b7cba03fc3b6dd454f99672a5e7999c4899a34b30d4f8d5dcc2e13de`
- R158 logs/descent: `8e1f858b3a49cf92c92c33a73ea9fb1c5133b8cd3eaa8f9939d60e96d74127af`
- R158 tests: `1f6134c23a8013c66d24a1d672a0e68faab615823872b9878a2af60f87b49582`
- R158 gate: `20077d3231e0c535af2fe7a4c435148de53fe1da300e1174d1959e89f12a203d`
- R158 parent: `6a88c2a881948d474248f15e91f826109bf97c46fc7c2cfe35f044f300530f89`
- Harness: `4965c430ebf3041f04a333a182b9b602a8371210e16a07d8bb586fda7a50b022`
- Harness tests: `323c616fdd2b72987b7016cbcf088855f04d872cbdb9932c7bdb478553410bb9`
- Report: `b543a8076493ddbb14d7c195fff70d20f056812da551c021beb29b29cd272d97`
- Note: `752b0276948ba51c5392cf4be1bd11352831355144fd9f6250762b8bf39b9cb3`
- Inventory: `74666ed7262ce05ebdff0bed6065cc318f03ae921d29fe36ee65e9b4a9f221fe`
- Replay plan: `da33bd8b8fc9dcb0de9a147fab45b0b8a1aa08f975a1040c6a869e3c5276084d`
