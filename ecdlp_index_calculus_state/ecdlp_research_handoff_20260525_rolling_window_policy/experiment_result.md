# Experiment Result

Added and ran:

`tasks/ecdlp_index_calculus/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_future_witness_rolling_window_policy_probe.py`

Primary artifact:

`ecdlp_index_calculus_state/future_witness_rolling_228_299.json`

The run materialized salts 228-299 once for both fixed targets, then scored 65
overlapping 8-salt windows by public features before auditing verifier-backed
two-equation witnesses.

Summary:

- Windows scanned: 65
- Positive windows with at least one target witness: 29
- Positive windows with both targets witnessed: 13
- Strict threshold learned from 228-235 selected 2 windows, both true positives,
  with 0 false positives
- Public ranking was stronger than the strict threshold: the top 8 ranked
  windows were all two-target positives
- First unseen two-target positive by public rank: salts 250-257, rank 3,
  best verified ops/rho 0.64
- Another unseen two-target cluster: salts 280-289, with best verified
  ops/rho 0.624

Interpretation:

Same-term-signature multiplicity across rows is now a credible public routing
signal for compact witnesses. It is not merely event density: the disjoint
236-243 negative has events but no same-signature repetition, while the best
unseen positives have repeated signatures on both targets.

This is still a selector/routing result, not a full general ECDLP index
calculus break. Row materialization is charged, and the two-equation witnesses
are still fixed-target compact witnesses rather than a proven scalable hit
stream.

## Fresh Seed Transfer

Extended the probe with explicit seed override knobs:

- `--challenge-seed-prefix`
- `--row-seed-prefix`
- `--scout-seed-prefix`
- `--filter-seed-prefix`

This lets us separate row schedule transfer from challenge/scout/filter
transfer.

Naive fresh `--seed ecdlp-frontier-signed-dual-sieve-v1-fresh-transfer-a`
was a controlled negative for below-rho/all-target transfer:

- `future_witness_fresh_a_250_257.json`: one-target positive only,
  best verifier pair `ops/rho = 1.12`
- `future_witness_fresh_a_282_289.json`: one-target positive only,
  best verifier pair `ops/rho = 1.008`
- `future_witness_fresh_a_236_243.json`: one-target positive only,
  best verifier pair `ops/rho = 1.08`

Then the row schedule was frozen to the original baseline while
challenge/scout/filter used the fresh seed:

- `future_witness_fresh_challenge_a_rowfixed_250_257.json`: zero verified
  windows; public threshold selected one false positive
- `future_witness_fresh_challenge_a_rowfixed_282_289.json`: zero verified
  windows; public threshold selected one false positive
- `future_witness_fresh_challenge_a_rowfixed_236_243.json`: zero verified
  windows; public threshold selected one false positive

Finally, row, scout, and filter schedules were all frozen to baseline and only
the shared challenge seed was changed:

- `future_witness_fresh_challenge_only_a_rowscoutfilterfixed_250_257.json`:
  zero verified windows; public threshold selected one false positive
- `future_witness_fresh_challenge_only_a_rowscoutfilterfixed_282_289.json`:
  zero verified windows; public threshold selected one false positive
- `future_witness_fresh_challenge_only_a_rowscoutfilterfixed_236_243.json`:
  zero verified windows; public threshold selected one false positive

In all controlled fresh-challenge runs, public same-signature structure
survived on only one target: `targets_with_events = 1`,
`targets_with_same_signature_pair = 1`, and `same_signature_pair_count = 28`.
The all-target verifier-backed signal did not transfer.

Extended this into a small challenge-seed stability matrix:

`ecdlp_index_calculus_state/ecdlp_research_handoff_20260525_rolling_window_policy/challenge_seed_stability_matrix.md`

With row/scout/filter fixed, the original baseline seed had all-target
positives on `228-235`, `250-257`, and `282-289`. Fresh-transfer-a had zero
positives and a one-target public signature count of 28 on all three windows.
Fresh-transfer-b had zero positives and no public events. Fresh-transfer-c had
zero positives and a one-target public signature count of 56 on all three
windows. The surviving one-target events for fresh-transfer-a and
fresh-transfer-c were on `22050.cf1@11731`; `67.a1@9803` did not produce the
matching event stream.

Updated interpretation:

The rolling-window policy is a real diagnostic selector for the original
shared-challenge seed, but it is not yet a challenge-invariant ECDLP index
calculus route. The failure persists even when row/scout/filter schedules are
held fixed, so the next lever is the algebraic challenge dependence of the
same-term signatures themselves.

## Signature Inspection

Added compact public event signatures to the rolling probe output, then
generated:

- `future_witness_signature_inspect_baseline_250_257.json`
- `future_witness_signature_inspect_fresh_a_250_257.json`
- `ecdlp_index_calculus_state/ecdlp_research_handoff_20260525_rolling_window_policy/signature_inspection_250_257.md`

For the same 250-257 window, baseline had both targets active and an
all-target verified witness (`ops/rho = 0.64`). Fresh-transfer-a with only the
challenge changed had zero verified windows. It produced a dense one-target
signature family on `22050.cf1@11731`:
`[56, [10, 10, 13, 13], "2+2"]`, but no events on `67.a1@9803`.

This sharpens the next search target: avoid ranking dense one-target
repetition as if it were cross-target algebraic compatibility.
