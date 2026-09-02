# P1436 autoresearch focus harness V103 result

Date: 2026-07-29

## Result

V103 binds R154 as the 90th closed frontier lane and routes the highest
priority action to
`s46_signed_quotient_random_rank_reverse_ffe_transfer`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R154 corrects the R153 rank target for the public identities

```text
ell(-C_j) = -ell(C_j).
```

Choosing one representative from each inversion pair turns every full
relation row `h` into the signed row

```text
hbar_j = h_j - h_(-j).
```

The public sign constraints have rank `|C|/2`, and all controls verify

```text
rank([sign constraints; H])
  = |C|/2 + rank(Hbar).
```

This halves the meaningful C-log dimension by a constant factor without
changing its `B^(3/4+o(1))` exponent. The eight actual R153 controls have
signed quotient ranks `[1, 0, 0, 0, 0, 0, 0, 0]`; none is full rank.

R154 also freezes 48 synthetic prime-cyclic controls before labels or
ranks are observed: two A inversion pairs, four through seven C inversion
pairs, occupancy multipliers `1,2,4,8`, and three seeds. All signed
relation identities and combined-rank formulas replay exactly.

Eleven synthetic controls attain full signed quotient rank. The
full-rank counts out of three trials, grouped by C-pair count and
increasing occupancy multiplier, are:

```text
4: 0, 0, 2, 2
5: 0, 0, 0, 2
6: 0, 0, 0, 2
7: 0, 0, 1, 2
```

This is finite evidence that occupancy explains the R153 rank failures.
It is not a random-rank concentration theorem or a transfer to
hash-to-curve elliptic decks. The controls use verifier labels and receive
no factor-log or attack credit.

R154 supplies no reverse signed marker FFE operator, hash-to-curve rank
transfer, candidate factor logs, or identical target descent. It consumes
no candidate DLP, root, count, marginal, rank, or source oracle. No
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R154 tests: 12 passed.
- Full harness tests: 120 passed.
- Full ECDLP suite: 820 passed.
- R154 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R154: 79 receipts, 1,475 bindings, 0 mismatches.
- V103 frontier preflights: 90 provided, 90 closed.
- R154 obligations: 16 of 26 passed; lane admission false.
- R154 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V103 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R154 producer: `8e3f3cf4947417f341d4b708258106e11eb391170eb956d75992c5b0c6ebe6ad`
- R154 report: `16ee4f848ef89c54bda5a413fb05fefc6f6863600ff79f9f6f925e59bcc2d9ef`
- R154 frozen interface: `f6082904507a58393507311fe712f5be2413fea71d087d3abebf6fe278a0856e`
- R154 cost ledger: `bd7ac6ba5efbdd9156b2ce0054bbdae29a19760a1b4323012d2d1c91320bbdbb`
- R154 source replay: `87576bb1e92529815938dc1e8a1520a50e6e850d09ee28500df55d0375a236ee`
- R154 controls: `4eef913ace45f47337f1362af946fe8eb26f23167b738f79f785583f1ada66d6`
- R154 logs/descent: `2302f2ea00ec127108817631f79961a4577dbbb0be912d11de8f0fd86343e631`
- R154 tests: `8b230718808612747c50d7cee4261c1f42482e235350b1040126f7f7ef345e64`
- R154 gate: `905b7ac86fb1e444e0d40252f0107dbc65293de9823ec69a9e171a39da14ab27`
- R154 parent: `46ae102679ddc3acaa35a3c7750bb1b5f2c6f504cd0d5292b826c99a2d3db7a4`
- harness: `ef7c9c9b09048d16231712ab5511d21e1ddebd67f94d62ac53f5f7f60205caf6`
- harness tests: `02c8842b0c79cc5df7eda225d6a734de6e9f06b7ca9d28dd2602d082fbd184f7`
- report: `279996b9f96c77f738f31e09de9fbc32de95155ce98e5718d6275656579259b0`
- note: `ee57deb0b3e2a3bb5db12109b6fb756a74230c7657a9e6eb204451c1fe2150db`
- inventory: `01793fca4faf996e166e8314d84d45b9fc9e9a69a29a42ce2d20728ec08cb6ba`
- replay plan: `721a89ab81c097536acdb1fb7704f32afc76e83024256d3709e696600789b01f`
