# P1436 autoresearch focus harness V104 result

Date: 2026-07-29

## Result

V104 binds R155 as the 91st closed frontier lane and routes the highest
priority action to
`s47_convolution_tanner_contiguity_reverse_ffe`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R155 extracts every relation target with exactly one unordered C6 source.
After dividing by its public nonzero A6 shift multiplicity and ordered
permutation multiplicity, the full row is

```text
h_b = m_b - 1[b=a].
```

The R154 sign quotient gives

```text
hbar_j
  = m_(C_j) - m_(-C_j)
    - 1[a=C_j] + 1[a=-C_j].
```

Each singleton row therefore has at most seven nonzero signed columns.
All 48 finite controls normalize exactly from their aggregate rows; the
maximum observed support is six.

The singleton matrices attain full rank in exactly the same 11 controls
as the aggregate R154 matrices. Multi-source aggregation is not
responsible for the finite transition. Coverage is necessary but not
sufficient: 17 controls cover every signed column yet remain rank
deficient. Every row set is closed under negation, and deficient controls
can have dozens of rows but only a few projectively distinct rows.

For an independent uniform sparse-support comparator, a fixed column is
uncovered with probability `product_i(1-k_i/n)`. At fixed width and a
constant number of rows per column, the expected number of uncovered
columns remains linear in `n`. High-probability coverage instead requires
`n(log n+omega(1))` total incidences.

With `n=B^(3/4+o(1))`, logarithmic oversampling changes no B exponent. A
reverse batch may be repeated `O(log B)` times while remaining
`B^(5/4+o(1))`; the conditional matrix-free solve remains
`B^(2+o(1))`.

R155 pins the primary papers arXiv:2301.09287 and arXiv:1906.05757. Their
sparse-rank models require independently uniform fixed-size supports or a
random Tanner graph with prescribed degrees. The M6 rows are correlated
endpoint-collision images of shared A/C decks. No coupling, contiguity,
or direct rank theorem is supplied.

R155 supplies no hash-to-curve rank transfer, reverse signed marker FFE
operator, candidate factor logs, or identical target descent. It consumes
no candidate DLP, root, count, marginal, rank, or source oracle. No
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R155 tests: 14 passed.
- Full harness tests: 121 passed.
- Full ECDLP suite: 835 passed.
- R155 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R155: 80 receipts, 1,496 bindings, 0 mismatches.
- V104 frontier preflights: 91 provided, 91 closed.
- R155 obligations: 17 of 28 passed; lane admission false.
- R155 breakthrough, rho-improvement, and Shoup-improvement claims: false.
- V104 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R155 producer: `a8c59626ebd38c87c7bc19c8c98cae15a548ce489ba3f39b6eb2387ababd5c00`
- R155 report: `11e69758caceba9926e8d1d046aef71f56ed075ab2d2766988eb8ecc258a7bac`
- R155 frozen interface: `f3a0f229c3449bf32e26e0c832eb36cd5911ebb1357e005f5e7fc9a5d195155b`
- R155 cost ledger: `9c7ac01885694dc51ed6450e20a6c44d4425d088a00e3cdfdf73f19bff84b977`
- R155 source replay: `3d8c2281175f3584a83c8502aa379a2222a9d0b95526f33a77a48f5f5f05570b`
- R155 controls: `1b9f0096c3fad2565e3af956b8a52be04877afabbfa24e063039a43211bb4530`
- R155 logs/descent: `307739c16b7f2d695679648dc91f79c84f564c69ac670305654d63cf088abb2d`
- R155 tests: `0c5dbd2eb8cded7a05bd9bea690d4dad17c2525dce0ef5dc1030a03143e3f509`
- R155 gate: `f558cd9a526471ad51e076256934bdea34a7fd2c5d41f878ae6fdd39eb797875`
- R155 parent: `66d36dd4a801f8b621c90415cbef3886562a8ca1660744b3771e3ba6ac011971`
- XORSAT rank PDF: `862fe28d87041f8c52444cc37703196191dcd1cd181c2c4ee37fef4d7cb78a52`
- sparse-rank PDF: `82275b37845a5d2f03648cd2c6189d0ff2931cb1d39b70e2c5dfa2f7adc1769d`
- harness: `3feac5bcce2223af75e3a6955fcd89761ac6779831397e398d18489b2c438110`
- harness tests: `3597ff788e3878547dc9d4937c999fac770f720100ceb9748b9d29fd02e67d04`
- report: `03b8f83fa1cd5efcdb417ef618641c61d8f840615f83d6e7746d13932c2f617a`
- note: `ac350fc336649a7b7ecdbc1e37b9fb5a2c679f75171c1c8e8968c222cb5c82e4`
- inventory: `332ef075423327a9ededf5ef7d2af642b8c6a90782a513edcab246f11f20794b`
- replay plan: `7783234e1b05418385eca0ba9c97e560862a527c3ba0a92c29e993dd6e352605`
