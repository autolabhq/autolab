# Experiment Result: noncontiguous tail through 6303

## Claim or task

Consume the newly visible `6296..6303` direct/rank/scout evidence without
pretending the frontier is contiguous, and preserve a precise bridge job for
the missing `6288..6295` direct/rank batch.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
NONCONTIGUOUS-FRONTIER-CHECKPOINT.

## Fresh mounted state

The `6280..6303` gap audit shows a noncontiguous complete tail:

```text
common complete ranges       6280_6287, 6296_6303
contiguous common end        6287
latest common complete end   6303
missing any range            6288_6295
noncontiguous complete tail  6296_6303
```

The missing range is not absent everywhere: scout has `6288_6295`, but direct
certificates and rank scorers do not.  Any `6288..6295` rows in open candidate
lists are scout-only until those two producer artifacts arrive.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_frontier_gap_audit.py
ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_6303_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6303_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6303_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6303_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6303_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6303_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6303_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6303_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6303_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_noncontiguous_tail_6303/
```

## Direct audit through 6303

The bridge audit has `82` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       22
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    29
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 29
```

Aggregate counts:

```text
rank-gain certificates                  24
accepted-missing certificates           51
accepted-missing rank gains             22
accepted priority-column certificates    4
selected priority-column certificates   82
```

Fresh noncontiguous-tail rows:

```text
6297 top_k=7 salts 173/163 forms [3,5], [9,11]
  accepted missing column 9
  rank_gain 0, unique_gain 2, cost 0.78832117x rho

6298 top_k=16 salts 165/164 form [10,14]
  accepted missing columns 10,14
  rank_gain 1, unique_gain 1, cost 0.75912409x rho

6302 top_k=4 salts 173/172 form [5,6]
  accepted missing column 6
  rank_gain 0, unique_gain 0, cost 0.75912409x rho
```

`6298` is the only new rank-gain certificate in the noncontiguous tail, and it
reinforces the `[10,14]` family.

## Accepted-form families

The family ranking after consuming the `6296..6303` tail is:

```text
[10,14] rank_gain_total 11, accepted_missing_rank_gain_count 7
        transfers 6060,6151,6153,6164,6272,6280,6298
[10,13] rank_gain_total 9, accepted_missing_rank_gain_count 7
        transfers 6031,6181,6189,6202,6204
[1,5]   rank_gain_total 9, accepted_missing_rank_gain_count 3
        transfers 6055,6151,6174,6256,6280
[11,15] rank_gain_total 7, accepted_missing_rank_gain_count 4
        transfers 6117,6228,6256,6280
[11,13] rank_gain_total 7, accepted_missing_rank_gain_count 5
[2,4]   rank_gain_total 7, accepted_missing_rank_gain_count 4
[9,11]  rank_gain_total 3, accepted_missing_rank_gain_count 3
[8,11]  rank_gain_total 2, accepted_missing_rank_gain_count 2
```

`[10,14]` is still the rank-total leader.  The priority family `[11,15]` is
unchanged by the noncontiguous tail; its newest positive is still `6280`.

## Priority recurrence

Accepted priority-column 15 rank-gain rows remain:

```text
6117 top_k=16 salts 167/170 form [11,15] rank_gain 1 unique_gain 1
6228 top_k=16 salts 167/176 form [11,15] rank_gain 1 unique_gain 1
6256 top_k=16 salts 166/167 forms [1,5],[11,15] rank_gain 2 unique_gain 3
6280 top_k=16 salts 161/175 forms [1,5],[10,14],[11,15] rank_gain 3 unique_gain 13
```

Priority recurrence audit:

```text
priority_positive_count              4
priority_positive_rank_gain_total    7
priority_positive_unique_gain_total 18
selected_priority_no_rank_controls  58
```

Stable public tokens remain the broad top-k16 full-support carrier, not a clean
salt split:

```text
selector_topk=mode_low_term_support_total5|16
selector_topk_support=mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
topk_support=16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
selected_has_13_and_15
```

## Public selector capture

The frozen public selector mined from `5984..6023` promotes `126` holdout
candidates through `6303`.  Against direct evidence:

```text
direct-audit matches                  33
direct rank-gain matches              17
accepted-missing rank-gain matches    15
unmatched candidates                  93
```

The broad top-k16 public carrier now has:

```text
candidates                            80
direct matches                        14
direct rank-gain matches              13
accepted-missing rank-gain matches    11
```

The stricter `5984..6119` calibration still promotes no hard selector-tier
holdout candidates.  It only emits the same `10` broad-atomic diagnostics at
`6280`, with `6` direct-verified and `3` shared-verified posthoc rows.

## Replay queues

Family planner combined top transfers:

```text
6297,6268,6183,6244,6128,6225,6204,6189,6148,6184,6163,6255
```

Per-family heads:

```text
[10,13] 6297,6225,6204,6189,6163,6255,6293,6121
[11,15] 6148,6184,6163,6255,6183,6293,6121,6244
[10,14] 6183,6244,6128,6268,6165,6248,6148,6184
[9,11]  6297,6163,6255,6183,6293,6121,6244,6247
[8,11]  6163,6255,6183,6293,6121,6244,6247,6143
[1,5]   6268,6148,6184,6163,6255,6183,6293,6121
```

Priority recurrence queue top transfers:

```text
6163,6255,6183,6293,6121,6244,6247,6202,6143,6169,6250,6208
```

`6293` is scout-only until the missing direct/rank bridge is produced.  Do not
count it as a direct or rank event yet.

## Interpretation

There are two separable work streams now:

1. Bridge the mounted artifact gap by producing direct and rank artifacts for
   `6288_6295`.
2. Continue the mechanism line where `6298` strengthens `[10,14]`, while `6280`
   remains the best coupled `[1,5]/[10,14]/[11,15]` priority-row witness.

The current best algorithmic direction is still a public top-k16/full-support
carrier with posthoc accepted-form structure, followed by a second-stage split
that tries to separate rank-gain rows from selected-priority no-rank controls.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The `6303` checkpoint is not a contiguous frontier advance until `6288_6295`
  direct and rank artifacts exist.
- Open queued rows are not progress until direct certificate export and rank
  audit match them.
