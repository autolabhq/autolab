# Experiment Result: noncontiguous tail through 6343

## Claim or task

Consume the live direct/rank/scout tail through `6343`, keep the `6288_6295`
producer gap explicit, and refresh the accepted-form/priority recurrence stack
after the new `6342` priority-column rank-gain row.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-CARRIER-RECURRENCE / NONCONTIGUOUS-FRONTIER-CHECKPOINT.

## Fresh mounted state

The recent-window gap audit still shows a complete but noncontiguous tail:

```text
common complete ranges       6280_6287, 6296_6303, 6304_6311, 6312_6319, 6320_6327, 6328_6335, 6336_6343
contiguous common end        6287
latest common complete end   6343
missing any range            6288_6295
noncontiguous complete tail  6296_6303, 6304_6311, 6312_6319, 6320_6327, 6328_6335, 6336_6343
```

The `6288_6295` scout artifact is visible, but the direct-certificate and
rank-scorer artifacts for that range are still missing.  This checkout has
consumer/audit scripts for those mounted artifacts; I did not find the local
producer script that emits the missing direct/rank pair.

## Artifacts

```text
ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6343_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6343_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_noncontiguous_tail_6343/
```

## Direct audit through 6343

The bridge audit has `95` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       27
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    30
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 36
```

Aggregate counts:

```text
rank-gain certificates                  29
accepted-missing certificates           57
accepted-missing rank gains             27
accepted priority-column certificates    6
selected priority-column certificates   95
```

Fresh rows after `6327`:

```text
6332 top_k=12 salts 172/174 form [3,5]
  saturated collapse, rank_gain 0, cost 0.75182482x rho

6338 top_k=7 salts 172/170 form [2,4]
  saturated collapse, rank_gain 0, cost 0.79562044x rho

6338 top_k=12 salts 170/172 form [2,4]
  saturated collapse, rank_gain 0, cost 0.79562044x rho

6342 top_k=16 salts 173/162 forms [1,5], [11,15]
  accepted missing column 15 and accepted priority column 15
  rank_gain 2, unique_gain 3, cost 0.83211679x rho
```

`6342` is the new movement: it adds the sixth accepted priority-column 15
rank-gain row and gives the priority line a second post-6280 recurrence after
`6314`.

## Accepted-form families

The family ranking after `6343` is:

```text
[10,14] rank_gain_total 12, accepted_missing_rank_gain_count 8
        transfers 6060,6151,6153,6164,6272,6280,6298,6320
[10,13] rank_gain_total 11, accepted_missing_rank_gain_count 9
        transfers 6031,6181,6189,6202,6204,6314
[1,5]   rank_gain_total 11, accepted_missing_rank_gain_count 4
        transfers 6055,6151,6174,6256,6280,6342
[2,4]   rank_gain_total 10, accepted_missing_rank_gain_count 7
        transfers 6031,6164,6189,6314
[11,15] rank_gain_total 10, accepted_missing_rank_gain_count 6
        transfers 6117,6228,6256,6280,6314,6342
```

The leader remains `[10,14]`, but `6342` makes the coupled priority family more
interesting: `[1,5]` and `[11,15]` now both have double-digit rank-gain totals.

## Priority recurrence

Accepted priority-column 15 rank-gain rows:

```text
6117 top_k=16 salts 167/170 form [11,15] rank_gain 1 unique_gain 1
6228 top_k=16 salts 167/176 form [11,15] rank_gain 1 unique_gain 1
6256 top_k=16 salts 166/167 forms [1,5],[11,15] rank_gain 2 unique_gain 3
6280 top_k=16 salts 161/175 forms [1,5],[10,14],[11,15] rank_gain 3 unique_gain 13
6314 top_k=16 salts 174/175 forms [2,4],[11,15] rank_gain 1 unique_gain 3
6342 top_k=16 salts 173/162 forms [1,5],[11,15] rank_gain 2 unique_gain 3
```

Priority recurrence audit:

```text
priority_positive_count              6
priority_positive_rank_gain_total   10
priority_positive_unique_gain_total 24
selected_priority_no_rank_controls  66
```

Stable public tokens are still broad:

```text
selector_topk=mode_low_term_support_total5|16
selector_topk_support=mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
topk_support=16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
selected_has_13_and_15
```

Stable salt tokens remain empty.

## Public selector capture

The frozen public selector mined from `5984..6023` promotes `148` holdout
candidates through `6343`.  Against direct evidence:

```text
direct-audit matches                  39
direct rank-gain matches              21
accepted-missing rank-gain matches    19
unmatched candidates                 109
```

The broad top-k16 public carrier now has:

```text
candidates                            96
direct matches                        18
direct rank-gain matches              16
accepted-missing rank-gain matches    14
```

The stricter `5984..6119` calibration remains diagnostic only: `10`
broad-atomic holdout candidates at `6280`, `6` direct-verified and `3`
shared-verified posthoc, with no promoted selector-tier holdout rows.

## Replay queues

Family planner combined top transfers:

```text
6128,6287,6297,6268,6189,6183,6244,6165,6248,6225,6204,6240
```

Priority recurrence queue:

```text
6165,6248,6163,6255,6183,6293,6121,6244,6247,6240,6270,6128
```

Transfers in `6288..6295`, including `6293`, remain scout-only until the
missing direct/rank bridge is produced.

## Interpretation

The strongest current line is now the priority recurrence:

```text
mode_low_term_support_total5 top_k=16 full support
accepted priority column 15
repeated forms including [1,5] and [11,15]
```

`6280` is still the best coupled row, but `6314` and `6342` make the recurrence
harder to dismiss as a singleton.  The next useful algorithmic step is a
second-stage public split that separates the six priority positives from the 66
selected-priority no-rank controls, while preserving the below-rho direct cost.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The `6343` checkpoint is not a contiguous frontier advance until `6288_6295`
  direct and rank artifacts exist.
- The public carrier is broad; no clean salt rule has been validated.
- Open queued rows are not progress until direct certificate export and rank
  audit match them.
