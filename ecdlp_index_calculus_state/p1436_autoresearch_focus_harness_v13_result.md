# P1436 autoresearch focus harness v13 result

## Status

`HARNESS_ENRICHMENT_PASS / R68_GATE_REPLAYED / PRODUCER_INPUT_MISSING / NONPROMOTING`

The alphaXiv post is now captured from the visible X post text rather than a
near-verbatim paraphrase. The report also records logical baseline/variant
lineage adapted from the alphaXiv OpenResearch harness, without claiming that
git branches or runs were materialized.

## Summation/FFE gate

The R68 information-conservation probe replayed byte-identically to the stored
report:

- replay/stored SHA-256:
  `5c064a36630c052093ebb7bd3b4429d37f836982ba097eacbad2f2ebe1bb6b51`;
- 4,161 core product-quotient rows retain rank 22, equal to the factor-row
  difference rank;
- 6,709 target-expanded quotient rows retain rank 23, again with zero added
  rank; and
- the uniform model does not beat rho.

The harness therefore gives product/summation quotient rows zero independent
information credit. A summation/FFE route is admissible only with exact
hash-bound payloads and a scalar-blind new-factor-row source enumerator whose
measured cost is strictly below direct pair-complement enumeration.

## Collision ABI

`p1436_collision_to_rank_routing_ablation.py` now validates
`ecdlp.p1436_collision_record.v1`, compiles exact coefficient and augmented
matrices for all/cross/within-shift variants, and recomputes ranks modulo the
curve order. Arbitrary nonempty lists no longer count as exact replay.

The current source artifact is a smoke fixture with SHA-256
`41589c3db13d160f793d5dbdfd55724795b331c58fff1f7343a5b65af3883f7f`.
It contains summaries but no raw collision records:

- exact configurations: 0/3;
- exact matrix variants: 0/9;
- routing status: `missing_exact_inputs`; and
- relation-matrix status: `missing_exact_inputs`.

## Verification

- task test discovery: 56/56 passed;
- Python compilation: passed;
- R68 replay hash matched the stored report exactly;
- focus report schema: `ecdlp.p1436_autoresearch_focus_report.v6`;
- routing schema: `ecdlp.p1436_collision_to_rank_routing_ablation.v2`; and
- relation-matrix schema: `ecdlp.p1436_relation_matrices.v2`.

## Boundary

This is stronger harness and obstruction evidence, not a generic-prime ECDLP
breakthrough. The original P1436 collector is absent from this authoritative
worktree, so collision-to-rank replay cannot advance until that producer emits
the required exact collision ABI.

Exactly one next action: restore or locate the original P1436 collector and add
`ecdlp.p1436_collision_record.v1` emission at the point where exact residual
collisions are converted into signed factor rows.
