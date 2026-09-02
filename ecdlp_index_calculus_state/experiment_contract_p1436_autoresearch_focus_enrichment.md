# Experiment Contract: P1436 autoresearch focus enrichment

## Purpose

Add a diagnostic layer to P1436 that distinguishes residual information that
exists from information that is usable by the end-to-end ECDLP pipeline.  The
adaptation is inspired by alphaXiv Autoresearch and the knowing-versus-using
evaluation in arXiv:2607.08393.

The process steering is also bound to the alphaXiv post at
`https://x.com/askalphaxiv/status/2076737985559822734?s=46`: keep the loop focused
on a few critical experiments, spend less effort on peripheral branches, and
resolve non-blocking ambiguity without pushing it back to the operator. The known
tweet text for this source is embedded as a bounded `source_summary`/`tweet_text`
snapshot plus explicit source-intake metadata and operator-interrupt policy.

The post's referenced alphaXiv Autoresearch surface identifies
`https://github.com/alphaXiv/openresearch-cli` as its local harness. Its
baseline/variant experiment-tree pattern is adapted here as logical lineage:
the exact P1436 source hash is the immutable root and every selected or deferred
probe is an explicitly parented node. This report does not claim that a git
branch, run, or remote experiment was materialized; that requires a separate
receipt bound to the root hash.

## Natural Route

The preregistered `random_hash_mask1` configuration remains the only route
eligible for an unqualified result.  Its stages are:

1. exact residual events,
2. exact cross-shift collisions,
3. nonzero signed factor rows,
4. full RHS-compatible anchored rank,
5. independently verified factor logarithms,
6. separate prospective target descent,
7. total work below the P1436 rho and fitted-exponent gates,
8. multiscale Shoup-pressure residual checks on exact+verified full-route points.

The rho gate means charged total work strictly below Pollard rho itself. A
reported ratio below `11 * rho` is diagnostic only and cannot satisfy this gate.
The multiscale exponent is fitted as `log(charged operations)` against
`log(group order)` and must be strictly below `0.5`; fitting against bit length
or fitting a rho-normalized ratio does not test the claimed exponent.

## Routing Interventions

The tweet guidance is encoded as `methodology.tweet_guidance` in the harness
output so each run explicitly records bounded critical-experiment focus,
deterministic non-blocking ambiguity resolution, and explicit peripheral-branch
deferral.
Each field in that guidance record also includes a short rationale note in the
human-facing output so non-causal decisions are explicit and auditable.

Existing frozen shift/source/mask configurations may be ranked after the run to
locate a routing failure.  The best such configuration is an oracle diagnostic,
not promotion evidence.  `mixed_balanced_stride_mask1` is the fixed heuristic
intervention.  Any claimed intervention improvement must be frozen on
development evidence and transferred unchanged to every prospective curve.

This is an analogy with, not an implementation of, the paper's self-patching.
The paper replaces an anchor representation inside the same model execution;
P1436 reruns a collector under a different frozen configuration.  The harness
therefore labels every comparison `matched_configuration_ablation` and forbids a
causal self-patching claim.

For each exact comparable cell, rank headroom is reported as

`(fixed_rank_fraction - natural_rank_fraction) /
 (oracle_rank_fraction - natural_rank_fraction)`.

The ratio is also recomputed after aggregating the target policy by curve split.
The paper's reported 58--75% range is LLM evidence only and is not imported as
an ECDLP threshold.  A fixed route is marked as positive prospective transfer
only when all prospective target-policy cells are exact and available, no cell
regresses in rank, and at least one cell improves.

The target-policy gain is paired by curve with every exact available hash-policy
replay.  This is the analogue of the paper's irrelevant-patching control: if the
`two_map_union` fixed-minus-natural rank gain does not exceed the mean matched
hash gain, a generic scheduler effect has not been ruled out.  Even a positive
coordinate-specific excess remains diagnostic-only.

Each natural-route cell also receives a deterministic synthetic-uniform
occupancy control with the same number of accepted residual events and shifts.
It may test collision supply only.  Synthetic residual labels are never given
factor rows, logarithms, or target-descent credit.

Exact collision-to-rank replay uses
`ecdlp.p1436_collision_record.v2`. Every collision record must contain a unique
edge ID, left and right shift labels, a full relation coefficient vector, the
relation RHS, a Boolean relation-admission decision, and positive exactness flags
for the source equation and residual equality. The admission decision must equal
whether the anchored unknown row is nonzero, matching the legacy collector. The
ablation compiler:

1. checks record and summary collision counts;
2. checks cross-shift and within-shift counts;
3. preserves duplicate admitted rows and excludes collector-rejected zero rows;
4. recompiles all-edge, cross-shift-only, and within-shift-only matrices;
5. recomputes coefficient and augmented ranks modulo the curve order; and
6. requires the all-edge admitted-row count, rejected-row count, and ranks to
   match the collector summary.

A nonempty list is not enough. ABI-invalid rows, count mismatches, rank
mismatches, and summary-only payloads remain non-exact and cannot support a
collision-to-rank claim.

## Focus Budget

The report emits at most three next probes.  Exactness failures take precedence,
followed by collision supply, collision-to-rank routing, fixed-route transfer,
factor-log verification, target descent, and end-to-end cost.

Every selected item carries a hypothesis, decisive test, falsifier, and required
artifacts.  Lower-ranked candidates are retained as explicitly deferred work,
with selected/deferred counts and selected priority mass recorded in the report.
Deferred work must not be opened until a selected experiment is falsified,
completed, or blocked by exactness.

## Summation/FFE Admission

The R68 information-conservation receipt is a hard lane boundary. A
product-section quotient or summation-polynomial relation is assigned zero new
factor-log information beyond the fixed-sum factor rows used to construct it.
The harness therefore does not treat a summation/FFE marker, an opaque payload
label, or a product quotient as a new relation channel.

An exact summation/FFE payload must be either a structured object with a schema
and SHA-256 binding or an external payload reference with a companion SHA-256.
Even then, the lane remains blocked until a structured discovery contract
provides:

1. a named scalar-blind source enumerator;
2. positive new and independent factor-row counts;
3. measured source operations;
4. the direct pair-complement operation count; and
5. a SHA-256-bound replay artifact.

The measured source cost must be strictly below direct pair-complement cost.
Passing this gate admits an exact replay only. Promotion still requires
target-uniform transfer, full independent rank, verified logs, target descent,
multiscale cost evidence below the Shoup-pressure gate, and independent audit.

The next report also records source-fidelity and guidance-compliance metadata:
tweet source id/query/status, tweet text hash when text is embedded, whether tweet
text was embedded, required ambiguity
resolution IDs required by the guidance policy, explicit compliance booleans for
bounded selection, deferred-branch handling, and non-blocking ambiguity
recording, plus an explicit operator-interrupt alignment ledger.

Each run also emits a single `next_action` record that selects the highest-priority
focus candidate and lists its decisive test, falsifier, and required artifacts.
That action is the one unambiguous next step for the AutoLab loop.

## Ambiguity Resolution

The harness records deterministic decisions instead of silently guessing:

- tied oracle routes are ordered by configuration name and remain post-hoc;
- a missing fixed route falls back to the natural route and is marked unavailable;
- synthetic residuals remain occupancy-only because they have no EC source;
- configuration ablations are never described as causal self-patching;
- the paper's numeric headroom range is never used as an ECDLP gate;
- absent target descent is `untested`, not success, and blocks promotion;
- absent or hash-mismatched independent audit evidence blocks promotion; and
- post-processing never creates a breakthrough claim absent from the source.
- logical experiment lineage never claims a branch or run was materialized;
- product/summation quotient rows receive zero information credit under R68;
- each ambiguity resolution includes whether the ambiguity is non-blocking or
  promotion-blocking (`uncertainty_class`) and whether operator interrupt is
  required for promotion.
- each full natural-route cell now includes a major-result replication matrix
  across replay, collision, rank, descent, and cost stages so we can report where
  the natural pipeline fails, is blocked, or stays untested.

## Promotion Boundary

No post-hoc intervention can promote P1436.  Promotion requires the natural
route to pass every stage, the original P1436 coordinate-versus-hash and cost
gates, and an independent audit whose recorded source hash matches the exact
P1436 probe JSON.

## Reproduction

```bash
python3 tasks/ecdlp_index_calculus/p1436_autoresearch_focus_harness.py \
  ecdlp_index_calculus_state/p1436_large_prime_residual_collision_collector_after_p1435_probe_for_harness_v4_smoke.json \
  --note-url "https://x.com/askalphaxiv/status/2076737985559822734?s=46"
```
