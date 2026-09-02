# P1436 autoresearch focus harness V82 result

Date: 2026-07-29

## Result

V82 binds R133 as the 69th closed frontier lane and routes the highest
priority action to
`s25_five_mode_low_slp_frobenius_torus_c5_selector`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R133 pins Zander Kelley's *Roots of Sparse Polynomials over a Finite
Field* (arXiv:1602.00208) and adapts the proof of Theorem 2.3 to the
prime-order torus subgroup. For a nonzero represented `t`-mode function on
the order-`q` subgroup, the largest root coset has size one and

```text
number of roots <= 2 q^(1 - 1/(t-1)).
```

A balanced C5 color requires `q^(3/4+o(1))` accepted roots. Consequently,
one through four represented extension-field modes are excluded
deterministically. Five modes are the first count not excluded by the
root exponent; no five-mode predicate is constructed.

R133 also freezes a separate uniform-random-deck model. Every color
contains its atoms' pure fifth powers, and fifth powering permutes the
prime-order subgroup. If `t-1 <= log2(q)/4`, every fixed represented
predicate has root density at most `1/8`. Projective support and
coefficient counting gives the four-color union bound

```text
4 T 36^T q^(3T) 8^(-m),
T = 1 + floor(log2(q)/4),
```

where `m` is the smallest color size. At the campaign's asymptotic deck
scale this tends to zero. This result is explicitly model-bound and does
not transfer to the deterministic structured factor base.

All eight actual pairing controls replay every color's pure-fifth sources
exactly in `Fp2`; the fifth powers are distinct and lie in their required
color acceptance sets. The fixtures have minimum active color size one,
so every numerical union bound is above one and receives no probability
or asymptotic credit.

R133 closes only one through four represented modes deterministically and
the small logarithmic range under the frozen random model. Five-mode
predicates, larger-polylog represented support, high-expansion low-SLP
predicates, multi-predicate Frobenius-coordinate DAGs, nonzero-value
tests, and general circuits remain open. No source index, relation-rank
construction, factor logs, identical target descent, Pollard-rho
improvement, Shoup improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R133 tests: 10 passed.
- Full harness tests: 99 passed.
- Full ECDLP suite: 573 passed.
- R133 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R133: 58 receipts, 1043 bindings, 0 mismatches.
- V82 frontier preflights: 69 provided, 69 closed.
- R133 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V82 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- Kelley pinned PDF: `250daacc1b157043cd9c7345036eb332fc69fee8e48a12ba81eddcc2b91995b7`
- R133 producer: `b53985696fb7a6b270b7e3cdffa50f50b6801cdc0569a567abf1486064d52bc7`
- R133 report: `5b02982dfdce90645db52e36ba6063a40a6476307ec6a269d3c0c4311d64e8d3`
- R133 frozen interface: `28e2827b522efd71b2abaebca377b169b9744263541499bd9a262eaf9d4f911b`
- R133 cost ledger: `7bc714f7c6b763c9347f78d094374fbf26442473bbdd58978e261de8494485c0`
- R133 source replay: `958baefed1027c0bf58d6ff666af3fad2c6546d9624acdc5bef9b8aac9c09dda`
- R133 controls: `5bf39fb2ec030c9300f1913b09d0d28800d0330d2eca26478ad1a322d3503938`
- R133 logs/descent: `757bb97a1adb867fb35455a17f822d825537cc59a9029459ad0c75f4e96a3e7a`
- R133 tests: `e5a350fa03fbf254f930ce9c52f1c36a648d234cba8617e7e6c8b533d6ab2891`
- R133 gate: `bfeb6cad6781ef2b63b36821b5ca4d02a7424353a29b99ed53aa974c580f5ffe`
- R133 parent: `236725e81e4a05451b7ea2ad5d0ef6ae51ed019d9916e7b8d0e9be5aaa4ebceb`
- harness: `5820e2ca7c2cdff5dc8ec59c39ea20bca41ab6df1d4940563a6fd47e8e820881`
- harness tests: `ab5af1c79b1a8c27391392bca800e5c0b896dab729c14bb88ec850bf3e943cd3`
- report: `e73aeb45ca7303834c3c67bb8d12a78a4ff4cfeb643578b740d10671cdff6c10`
- note: `d2713715fbc713359efd4378ade39651a5139acf1c05e99ea08a35365a5ff920`
- inventory: `0fdeaa27798ed432fc6bf749b9e9ea62fed562484da46b5169c2f9c279442b11`
- replay plan: `068963a6fbac1c805e3346f406017124f6cfbe3bd56e6d1aea56667854903fbf`
