# Experiment Result: priority recurrence through 6271

## Claim or task

Consume the live direct/rank/scout frontier through `6271`, refresh the
accepted-form replay stack, and audit whether the priority-column bridge has
become a repeated public-carrier signal.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-CARRIER-RECURRENCE / CHECKPOINT.

## Fresh mounted state

Direct certificates, rank-scorer outputs, and support scouts were complete
through `6271` for this checkpoint.  The process-list check was unavailable in
this sandbox because `pgrep` could not access `sysmond`, so running-job state is
not claimed here.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_priority_recurrence_audit.py
tasks/ecdlp_index_calculus/low_term_total2_family_replay_planner.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6271_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6271_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6271_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6271_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6271_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6271_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6271_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6271_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_priority_recurrence_6271/
```

## Direct audit through 6271

The bridge audit now has `77` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       19
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    27
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 29
```

Aggregate counts:

```text
rank-gain certificates                  21
accepted-missing certificates           46
accepted-missing rank gains             19
accepted priority-column certificates    3
selected priority-column certificates   77
```

Fresh rows after `6255`:

```text
6256 top_k=16 salts 166/167 forms [1,5], [11,15]
  accepted missing/priority column 15
  rank_gain 2, unique_gain 3, cost 0.75182482x rho

6270 top_k=4 salts 166/173 forms [0,2], [3,5]
  accepted missing column 0
  rank_gain 0, unique_gain 2, cost 0.75912409x rho
```

`6256` is the important row: the priority-column bridge repeated again and
added rank gain `2`.

## Priority recurrence

Accepted priority-column 15 rank-gain rows are now:

```text
6117 top_k=16 salts 167/170 form [11,15]       rank_gain 1 unique_gain 1
6228 top_k=16 salts 167/176 form [11,15]       rank_gain 1 unique_gain 1
6256 top_k=16 salts 166/167 forms [1,5],[11,15] rank_gain 2 unique_gain 3
```

This gives:

```text
priority_positive_count             3
priority_positive_rank_gain_total   4
priority_positive_unique_gain_total 5
selected_priority_no_rank_controls  56
```

The new `low_term_total2_priority_recurrence_audit.py` found a stable public
carrier but not a stable positive salt rule.  The stable public tokens are the
top-k16 full-support carrier and related selected-support tokens:

```text
selector_topk=mode_low_term_support_total5|16
selector_topk_support=mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
topk_support=16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
selected_has_13_and_15
```

The tempting salt hypothesis does not survive controls:

```text
salt_has=167 positive_count 3, control_count 7, score -0.79953567
stable_salt_tokens []
```

So the current honest claim is public-carrier recurrence, not salt-rule
prediction.

## Accepted-form families

The contrast summary through `6271`:

```text
[10,13] rank_gain_total 9, accepted_missing_rank_gain_count 7
[11,13] rank_gain_total 7, accepted_missing_rank_gain_count 5
[2,4]   rank_gain_total 7, accepted_missing_rank_gain_count 4
[10,14] rank_gain_total 6, accepted_missing_rank_gain_count 4
[1,5]   rank_gain_total 6, accepted_missing_rank_gain_count 2
[11,15] rank_gain_total 4, accepted_missing_rank_gain_count 3
[9,11]  rank_gain_total 3, accepted_missing_rank_gain_count 3
[8,11]  rank_gain_total 2, accepted_missing_rank_gain_count 2
```

`[10,13]` remains the strongest broad family.  `[11,15]` is now the strongest
priority-column family and has three independent transfers: `6117`, `6228`,
and `6256`.

## Public selector capture

The frozen public selector mined from `5984..6023` promotes `113` holdout
candidates through `6271`.  Against direct evidence:

```text
direct-audit matches                  29
direct rank-gain matches              14
accepted-missing rank-gain matches    12
unmatched candidates                  84
```

The broad top-k16 carrier now has:

```text
candidates                            71
direct matches                        11
direct rank-gain matches              10
accepted-missing rank-gain matches     8
```

The stricter `5984..6119` forward miner still promotes zero holdout candidates.

## Replay queues

Priority recurrence queue top transfers:

```text
6163,6255,6183,6121,6244,6247,6202,6143,6169,6250,6208,6165
```

Family planner combined top transfers:

```text
6268,6183,6244,6128,6225,6204,6189,6163,6255,6121,6247,6250
```

The new `6268` family-planner leader is driven by the `[1,5]` side family and
shared `[10,14]` salt pair, not by a direct priority label.

## Interpretation

The campaign now has two distinct toy signals:

```text
1. [10,13] accepted-form preservation remains the strongest broad rank-gain
   family.
2. [11,15] accepted priority-column 15 is now a repeated top-k16/full-support
   public-carrier ridge, with three accepted priority positives.
```

The second signal is closer to the summation-polynomial/FFE motivation because
it repeatedly preserves the target priority column, but it is still only
toy-model direct evidence.  The immediate next step is exact-row replay of the
priority queue and a stronger public split that separates top-k16 priority
positives from top-k16 no-rank controls.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The stable public signal is a broad top-k16/full-support carrier, not a clean
  salt rule.
- Open queued rows are not progress until direct certificate export and rank
  audit match them.
