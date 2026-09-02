# P1436 autoresearch focus harness V80 result

Date: 2026-07-29

## Result

V80 binds R131 as the 67th closed frontier lane and routes the highest
priority action to
`s23_lacunary_order2_torus_c5_selector_predicate`. The report remains
`DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R131 applies a field-independent polynomial root bound directly in the
actual `ord_q(characteristic)=2` pairing fields. A nonzero Laurent
polynomial with consecutive modes

```text
a, a+1, ..., a+s-1
```

has at most `s-1` distinct nonzero roots. Multiplication by `X^(-a)`
reduces this to the ordinary degree bound, so no complex-to-finite-field
transfer is required.

For the balanced four-color selector, one color accepts

```text
binom(n+4,5) - binom(n-m+4,5) - m*binom(n-m+3,4)
  = Theta(n^5)
  = B^(15/4+o(1))
```

targets. A dense consecutive-mode zero predicate therefore requires a
`B^(15/4+o(1))` serialized block and sequential arbitrary-target work,
missing both the `B^(9/4+o(1))` setup cap and the polylogarithmic query
cap. An explicit product tree over the accepted roots has the same
failure.

All eight actual pairing-deck controls pass. Their thirty active color
sets have exact combinatorial counts, distinct target values, nonzero
Vandermonde determinants, exact dense-annihilator zero sets, and zero
rejection. Every positive source also yields an R129-optimal within-color
C2 branch and a complementary C3 source whose product replays the target.
Verifier source labels receive no locator or asymptotic credit.

The result is deliberately scoped. Polynomial degree is not circuit size,
and the argument does not close lacunary finite-field modes, high-degree
low-SLP predicates, multi-predicate shared DAGs, or non-Fourier selectors.
No source index, relation-rank construction, factor logs, identical target
descent, Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough
is claimed.

## Verification

- Focused R131 tests: 10 passed.
- Full harness tests: 97 passed.
- Full ECDLP suite: 551 passed.
- R131 clean rerun: all six JSON outputs byte-identical.
- Harness clean rerun against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R131: 56 receipts, 1004 bindings, 0 mismatches.
- V80 frontier preflights: 67 provided, 67 closed.
- R131 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V80 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R131 producer: `d312f28885677339632ec9517c8529e327a63fa7e8aca5d6a67d6a4f71e3adf9`
- R131 report: `a2ffed2c599fd1003dce13a3a360d7ac49f983ab92a439f1023a8abe0cb2efcb`
- R131 frozen interface: `46775815377c1cadbc095735b4f1fe9cd64167ea7b933664ae5892e5bc1b8ded`
- R131 cost ledger: `d18d3317aa2952b604c880d44447cef0d7bbc2c01b33a28b4131889fdfea8965`
- R131 source replay: `80c0a9d57247a981a55ec104cfb8f933e44b2bc610db945aaa89a7b473068f9b`
- R131 controls: `c9b874db6c10f6108b1179bcddd7e2e5dcc398cec6eaa871649912f07e20fb2a`
- R131 logs/descent: `9388946d26f2e311fd712feaaed9e6905e4e5ecbe5ceed554084d6e57fd39b36`
- R131 tests: `50c4fb705854bb11de19096716295d05ba54b035e6dc8ef675c112219481cb3a`
- R131 gate: `e2b714358fa386b23f8c9bdf75ab6d6971d080209d0e45885b4ba4fbb623f11d`
- R131 parent: `ca65e6e86c3aeae074b13d80774b08c18743932776e8b546409417c40dca6e46`
- harness: `288b928010d3e5650fac739a0d2aa9df4f85615b26629b25b4fc183d8f49576d`
- harness tests: `35280ea58cf8019e33031ff0fc9b6f34c0cef902f99f223195a97ed95deb6821`
- report: `6ad6bf486ab02e486eb78d61367db9e79098a364fb9c98055f520d260f8123e8`
- note: `1ea5ad42e869eaa3ae91f528520f0c81ea9e7356f49576d6445cdf950147b5e2`
- inventory: `3a17b4edfa990f4be14b5d41b2598153cd132010b3f6385b9567b31fa1953627`
- replay plan: `867df603ee1163e22bbb55d8474b24a20b376967a3698f9ac5edc35e2516baa0`
