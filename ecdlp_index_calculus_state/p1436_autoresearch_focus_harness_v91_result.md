# P1436 autoresearch focus harness V91 result

Date: 2026-07-29

## Result

V91 binds R142 as the 78th closed frontier lane and routes the highest
priority action to `s34_algebraic_sextic_character_composition`. The
report remains `DIAGNOSTIC_ONLY_WITHHOLD_PROMOTION`.

R142 tests the simplest genuinely adaptive use of the R141 labels. Each
internal node chooses one unused nonidentity parameter from the frozen
pairing deck, evaluates

```text
chi_z(x) = T_z(x)^q in mu_6,
```

and branches on the exact value. An accept leaf is valid only if it
contains no inverse-empty target and one C2 pair occurs in every positive
five-source multiset reaching it. A reject leaf may contain
inverse-empty targets only.

An exhaustive dynamic program minimizes leaf count, then maximum depth,
node count, and parameter index. Repeating a parameter on one path
cannot refine that path, so the program covers every deterministic tree
in the frozen grammar.

Five of eight actual controls retain an inadmissible cell even after all
deck parameters are queried. Those controls have no exact tree in the
grammar. The remaining three controls have exact optimal trees with

```text
minimum leaves: 33, 360, 645
maximum depths:   3,   5,   5
```

Every control either has no exact tree or exceeds its finite
`floor(B^(9/4))` leaf comparator. Every accepted positive in the three
exact trees replays to a shared C2 pointer, a concrete C3 certificate,
the exact target product, and the original sorted five-source multiset.
Every inverse-empty target reaches a reject leaf.

This is an exact finite negative only for direct deck-landmark trees.
It does not cover arbitrary Mobius parameters, arithmetic combinations
of character values, compiled algebraic circuits, RAM, cell probes, or
structured asymptotic factor-base families. The finite cap comparison
receives no asymptotic credit.

A random-source comparator predicts why the direct grammar behaves this
way: two independent uniform five-subsets share any C2 pair with
probability at most `200/(B*(B-1))`. Under an additional independent
random-signature model, routable cells approach singletons and direct
state approaches `B^5`. This is not a theorem about the actual
structured labels.

Nearby shifted-character literature supplies context, not a transfer:
van Dam and Hallgren give efficient quantum shifted-character algorithms,
while Bourgain, Garaev, Konyagin, and Shparlinski study classical hidden
shifted powers. R142 claims neither a reduction nor a classical lower
bound.

The surviving route is an algebraic composition law across the C2/C3
split, or a broader parameter compiler with proved state below
`B^(9/4+o(1))`. No inside-cap source index, relation-rank construction,
factor logs, identical target descent, Pollard-rho improvement, Shoup
improvement, or ECDLP breakthrough is claimed.

## Verification

- Focused R142 tests: 11 passed.
- Full harness tests: 108 passed.
- Full ECDLP suite: 673 passed.
- R142 clean replay: all six JSON outputs byte-identical.
- Harness clean replay against the seed-1432001 collision source: note
  byte-identical; three JSON outputs identical after removing their
  `generated_at` timestamps.
- Parent audit R76-R142: 67 receipts, 1,218 bindings, 0 mismatches.
- V91 frontier preflights: 78 provided, 78 closed.
- R142 obligations: 16 of 27 passed; lane admission false.
- R142 breakthrough, rho-improvement, and Shoup-improvement claims:
  false.
- V91 source breakthrough, promotion, below-rho, and Shoup-pressure
  gates: false.

## Hashes

- R142 producer: `d99409f519e96a9c62644bddc8262d9e535ec55e8fdd4a70ee828a5486af2eef`
- R142 report: `2a6b812448fcf1e24aac0a7d85f56e2fc0cc9a26de5aaf73b42a1d8ae58b8122`
- R142 frozen interface: `b3b1a96ec46f8617b249f366b3fb77bc64bdde4d5188b47e293033a63a82a51e`
- R142 cost ledger: `71ee44314fd109bf2854a659d4a80356d5b084602f1f2e5a7113e2018ba706d4`
- R142 source replay: `b285ea1f2883f16b53f30b00aa6793d84f55b8397200297ec7c227c5e3083a10`
- R142 controls: `bf452fbfc523fa2469c70da68e25439d8fe79fe7a550a12f8cc6736d6b73d59a`
- R142 logs/descent: `d5a35abfa58dc93a8a5bd6d2d315a186c53a8e435d637084abe02d106f63b81f`
- R142 tests: `43723e93867533406eb45ab41cdfedf9f1476f8879018b32e8a106be7935dcb0`
- R142 gate: `c4663f680cd2816d721b0d8020c4c5ced9479fc40b4b417701ebc2e060cbdae4`
- R142 parent: `4ab045100abb9537baa6c2e89cbfbd82809f6adff420837dcadce1224dc3d6d7`
- harness: `8236c2a793aa574b9ffb604ef20e7f1c6d70324ce41df585cf27b7231a979ed1`
- harness tests: `f94d5a9b9efe9d9c29b9653c6fef1288c4acabf3032e871976fdc3c3ccd84fe8`
- report: `1471d54c3e25571f71f13e233ad30ef29fa3d97f3f832c9199026d367c42ca17`
- note: `574baafd9e83264f647dfc3bd8e468cdb9d629c8040d18c7da1014eabd1d1db3`
- inventory: `134bff0c54f258bb5268241142b1adbaeba02e7bed4eab3998ad7a38174ed5fe`
- replay plan: `4d2d58f52d981580562fe65c3e96ed211be92cbeb2f5ef07c68bc231f140f625`
