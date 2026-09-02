# P1436 autoresearch focus harness v2 result

## Status

`HARNESS_ENRICHMENT_PASS / NO_P1436_CAMPAIGN_RESULT / NONPROMOTING`

The harness now incorporates the process guidance from the alphaXiv post at
`https://x.com/askalphaxiv/status/2076737985559822734`: select a bounded set
of decisive experiments, defer peripheral branches explicitly, and resolve
non-blocking ambiguities deterministically with an auditable record.

## Changes

- Every selected focus item has a hypothesis, decisive test, falsifier, and
  required artifact list.
- The report records selected and deferred experiments, priority mass, and the
  gap at the focus-budget boundary.
- The report records deterministic resolutions for oracle ties, absent fixed
  routes, synthetic controls, missing target descents, audit binding, and source
  claim authority.
- Missing target-descent evidence is now `untested` and blocks promotion.
- The report schema is `ecdlp.p1436_autoresearch_focus_report.v2`.

## Verification

`python3 -m unittest -v tasks/ecdlp_index_calculus/tests/test_p1436_autoresearch_focus_harness.py`
passes `8/8` tests. Python compilation passes for the harness and test module.
All eight critical-experiment templates contain a hypothesis, decisive test,
falsifier, and required artifact list.

## Bound Artifacts

- harness SHA-256: `0b1c5ffa882707741a7de06fc7aaab85206e43b7c0c3c53cbc0708e078f859e7`
- tests SHA-256: `b8a8f3226ceb0582c039b6d1c6c6e0816a36791b9e58026ad15196c2a8ffe0b2`
- contract SHA-256: `b88972f3ef353f95db036b4b1ba97b5345a670d59f21b7f6af3fe0e3d059455b`

## Campaign Boundary

The default P1436 producer artifact
`p1436_large_prime_residual_collision_collector_after_p1435_probe.json` is
absent in both this worktree and `/Volumes/Volume/autolab`. No fixture-derived
campaign claim is published. This result establishes harness behavior only; it
does not support or reject P1436 arithmetic, rank, descent, cost, or a
better-than-rho ECDLP claim.
