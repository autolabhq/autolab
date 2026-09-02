# P1436 autoresearch focus harness V85 result

Date: 2026-07-29

## Result

V85 binds R136 as the 72nd closed frontier lane and routes the highest
priority action to
`s28_growing_support_low_slp_nonzero_torus_c5_selector`. The report
remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R136 proves an all-nonzero path-product dichotomy for deterministic
decision trees whose nodes test represented polynomial equality to zero.
If `f_1,...,f_d` occur on the all-nonzero path, then a point follows that
path exactly when

```text
P(z) = product_i f_i(z)
```

is nonzero. A rejecting leaf forces `P` to vanish on the complete
positive support. An accepting leaf forces the union of the node root
sets to cover the complete subgroup complement.

R135 rules out the first alternative on a structured C5 color when the
expanded product has at most six modes. Kelley-adapted root bounds make
the second alternative impossible for polylogarithmically many
fixed-support nodes, because the positive support has size
`q^(3/4+o(1))` and the complement has size `q-o(q)`. Exact trees must
therefore escape through expanded path-product support above six.

Under the frozen uniform-random-deck model only, the product threshold
becomes `(5-o(1))*log2(q)` with overwhelming probability. For fixed node
support this yields only an `Omega(log log q)` depth condition, which is
compatible with polylogarithmic query work. It is not a low-SLP or
general circuit lower bound.

Across the eight actual pairing controls, 12 two-atom colors replay five
linear factors whose product has exactly six modes and vanishes on five
of the six progression positives. The sixth positive and its verified
inverse empty target are both nonzero at every factor and therefore
follow the same path. All C5 sources and color acceptance checks replay
exactly. Singleton controls receive no asymptotic credit.

R136 closes only bounded-expanded-support zero-test DAGs. Growing-support
low-SLP paths, nonzero-value Frobenius-coordinate circuits, and
structured seven-plus-mode products remain open. No source index,
relation-rank construction, factor logs, identical target descent,
Pollard-rho improvement, Shoup improvement, or ECDLP breakthrough is
claimed.

## Verification

- Focused R136 tests: 10 passed.
- Full harness tests: 102 passed.
- Full ECDLP suite: 606 passed.
- R136 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R136: 61 receipts, 1,101 bindings, 0 mismatches.
- V85 frontier preflights: 72 provided, 72 closed.
- R136 breakthrough, Shoup-improvement, and rho-improvement claims: false.
- V85 source breakthrough, promotion, below-rho, and Shoup-pressure gates:
  false.

## Hashes

- R136 producer: `a0761c05909996a80b3322aec859fcde97b23d77ce7cf512ddfa0c888e3017d9`
- R136 report: `530b696871917a69ddef1e6a0a14a44421176439c0ccf7b9031ecf5a8a1b64ce`
- R136 frozen interface: `3a8587a33e176a7174dacf88ced1da243b0be9b77f214a1b41bb2ef3db4ccef0`
- R136 cost ledger: `30935ad0b297822f56dca0e69fe80952e51b92c61f32cdee79a43dde8d483d8e`
- R136 source replay: `4c4414fa1472eaec54f9b71eeb6defc47b75c3b278cdad66b7d14e5fa0fb1d4d`
- R136 controls: `bcd98529f44223cb07510094de08c0f234da14830410b67847ec1aef75bf216b`
- R136 logs/descent: `c38aaf5b7966f24273ffe76d6564f4e2cf9d845caa2d17715a535ce044209049`
- R136 tests: `c6dd14aeabda0f4b69b6d3f021b207d998214d63922178fab58220a8beac01e3`
- R136 gate: `1a7e631d374e9b49d8d7eaf3440bc31d95b8d28f8f73f5a3e520b1f461bcabc5`
- R136 parent: `a4d61b3f08e92de9b9276bc12d137a1ff1c1494e81d08d8432c6fea9ecaa72dc`
- harness: `b2f67780d6d55b5ad9bedc5ab016d5d5260795fc9ea8ce5ae8f0bb981ef57736`
- harness tests: `05cf928af690f6a7d9b4301442eec47e97482b497ee07da9c8d40c312b81c6a6`
- report: `57d60e9cd14cf89fdff25c9f5bcc1c6ec8460d45099533b97bd399fff7445235`
- note: `8503010076717cfa62c8e05b03e17c1b3dcf74562607d90e9d56900a530069ac`
- inventory: `302e4787f13c31a4a99707c0d9c5cfef95a2bbeff4bcfa82368bc70b3880a15d`
- replay plan: `c524f85f8a16361d2f7b2f9014a30cc0bfd1fc057fb1c7bf30d62c2ca2248ff6`
