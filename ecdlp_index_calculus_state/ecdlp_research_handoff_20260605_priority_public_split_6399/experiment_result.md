# Experiment Result: priority public split through 6399

## Claim or task

Refresh the priority public split against the newly arrived `6384..6399`
direct/rank/scout tail and supersede the `6383` handoff.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / PUBLIC-SPLIT-SUBCARRIER /
BROAD-CARRIER-VALIDATED / NONCONTIGUOUS-FRONTIER-CHECKPOINT.

## Fresh mounted state

Direct/rank/scout common evidence now reaches `6392_6399`, but the frontier is
still noncontiguous:

```text
contiguous common end        6287
latest common complete end   6399
missing recent ranges        6288_6295, 6344_6351, 6352_6359
new complete tail            6384_6391, 6392_6399
```

The `6384..6399` tail is valid evidence, but not a contiguous frontier advance.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_priority_public_split_miner.py
ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_public_split_miner_5984_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6399_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6399_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_priority_public_split_6399/
```

## New tail rows

Fresh rows after `6383`:

```text
6384 top_k=7 salts 165/164 forms [2,4], [8,11], [9,11]
  accepted missing columns 8,9, rank_gain 0

6386 top_k=7 salts 173/170 form [8,11]
  accepted missing column 8, rank_gain 0

6388 top_k=12 salts 172/164 form [8,11]
  accepted missing column 8, rank_gain 0

6394 top_k=16 salts 172/173 forms [10,14], [11,15]
  accepted missing columns 10,14,15
  accepted priority column 15
  rank_gain 2, unique_gain 6
```

The important row is `6394` top-k16.  It is a new accepted priority-column 15
rank-gain row and it satisfies the broad `selected_has=13` carrier, but its
salt max is `173`, so `salt_max_mod4=1`.  It does not validate the stricter
`salt_max_mod4=3 AND selected_has=13` subcarrier.

## Priority split refresh

The split miner result through `6399`:

```text
priority positives                         8
selected-priority no-rank controls        75
selected-priority other rank-gain rows    25
priority rank_gain_total                  14
priority unique_gain_total                33
```

Full-recall carrier:

```text
selected_has=13
positive_count 8 / 8
control_count 2 / 75
precision 0.8
controls: 6087,6319
```

Robust zero-control subcarrier:

```text
salt_max_mod4=3 AND selected_has=13
positive_count 4 / 8
control_count 0 / 75
precision 1.0
recall 0.5
positive transfers: 6256,6280,6314,6378
```

This is a broad-carrier validation event, not a strict-subcarrier validation
event.  The `6378` validation point from the `6383` handoff remains inside the
zero-control subcarrier; `6394` lowers subcarrier recall from `4/7` to `4/8`
without adding a control.

## Family movement

Accepted-form contrast through `6399`:

```text
[10,14] rank_gain_total 14, transfers 6060,6151,6153,6164,6272,6280,6298,6320,6394
[11,15] rank_gain_total 14, transfers 6117,6228,6256,6280,6314,6342,6378,6394
[1,5]   rank_gain_total 14, transfers 6055,6151,6174,6256,6280,6342,6368,6378
[10,13] rank_gain_total 12, transfers 6031,6181,6189,6202,6204,6314,6378
[2,4]   rank_gain_total 11, transfers 6031,6164,6189,6314,6378
```

`6394` ties `[10,14]` and `[11,15]` with `[1,5]` at rank-gain total `14`.
The priority-positive `[11,15]` line now has eight accepted priority-column
hits.

## Public selector capture

The frozen public selector mined from `5984..6023` now promotes `178` holdout
candidates through `6399`:

```text
direct-audit matches                  48
direct rank-gain matches              25
accepted-missing rank-gain matches    22
unmatched candidates                 130
```

The broad top-k16 full-support carrier has:

```text
candidates                           113
direct matches                        21
direct rank-gain matches              19
accepted-missing rank-gain matches    16
```

The soft row-geometry scorer through `6399` still has forward candidates:

```text
holdout candidates                   200
direct verified posthoc hits          21
top transfers                         6121,6163,6183,6202,6244,6247,6255,6293,6298,6320,6367,6382
```

## Interpretation

The priority line is now split into two useful work orders:

```text
stage 1: selected_has=13 captures all accepted priority positives with two no-rank controls
stage 2: salt_max_mod4=3 captures 6256,6280,6314,6378 with zero no-rank controls
```

`6394` strengthens stage 1 and keeps stage 2 alive.  A future row with
`salt_max_mod4=3 AND selected_has=13` that is accepted-priority rank gain would
promote the subcarrier; a future no-rank row with the same public tokens would
demote it.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The zero-control subcarrier covers `4/8` positives, not the full recurrence.
- The frontier is still noncontiguous because `6288_6295` and `6344_6359`
  direct/rank are missing.
- Open queued rows are work orders until direct certificate export and rank
  audit match them.
