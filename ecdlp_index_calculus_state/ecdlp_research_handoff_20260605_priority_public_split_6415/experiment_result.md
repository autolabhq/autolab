# Experiment Result: priority public split through 6415

## Claim or task

Extend the priority public split and accepted-form family stack through the
new complete `6400..6415` direct/rank/scout tail.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / PUBLIC-SPLIT-SUBCARRIER /
NONPRIORITY-FAMILY-LEAD / NONCONTIGUOUS-FRONTIER-CHECKPOINT.

## Fresh mounted state

Direct/rank/scout common evidence now reaches `6408_6415`, but the frontier is
still noncontiguous:

```text
contiguous common end        6287
latest common complete end   6415
missing recent ranges        6288_6295, 6344_6351, 6352_6359
new complete tail            6400_6407, 6408_6415
```

The scout stream has a newer `6416_6423` file, but matching direct/rank files
were not present at this checkpoint.  The `6415` bridge is therefore the
latest complete direct/rank/scout evidence.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_priority_public_split_miner.py
ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_public_split_miner_5984_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6415_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6415_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_priority_public_split_6415/
```

## New tail rows

Fresh rows after `6399`:

```text
6402 top_k=16 salts 163/168 form [10,14]
  accepted missing columns 10,14
  rank_gain 1, unique_gain 1

6403 top_k=7/top_k=12 salts 170/172 form [9,11]
  accepted missing column 9, rank_gain 0

6404 top_k=4 salts 167/175 forms [0,2], [2,4], [3,5]
  accepted missing column 0, rank_gain 0

6405 top_k=16 salts 177/162 form [10,14]
  accepted missing columns 10,14
  rank_gain 1, unique_gain 1

6408 top_k=4 salts 172/173 form [5,6]
  accepted missing column 6, rank_gain 0

6413 top_k=4 salts 165/173 form [5,6]
  accepted missing column 6, rank_gain 0

6414 top_k=7/top_k=12 salts 173/163 forms [2,4], [9,11]
  accepted missing column 9, rank_gain 0
```

The important rows are `6402` and `6405`: both are top-k16 full-support
accepted-form rank gains for `[10,14]`, but neither accepts priority column
`15`.

## Priority split refresh

The priority split through `6415` is stable relative to `6399`:

```text
priority positives                         8
selected-priority no-rank controls        82
selected-priority other rank-gain rows    27
priority rank_gain_total                  14
priority unique_gain_total                33
```

Full-recall carrier:

```text
selected_has=13
positive_count 8 / 8
control_count 2 / 82
precision 0.8
controls: 6087,6319
```

Robust zero-control subcarrier:

```text
salt_max_mod4=3 AND selected_has=13
positive_count 4 / 8
control_count 0 / 82
precision 1.0
recall 0.5
positive transfers: 6256,6280,6314,6378
```

No new accepted-priority row arrived after `6394`; the subcarrier remains
zero-control and unchanged.

## Family movement

Accepted-form contrast through `6415`:

```text
[10,14] rank_gain_total 16, transfers 6060,6151,6153,6164,6272,6280,6298,6320,6394,6402,6405
[11,15] rank_gain_total 14, transfers 6117,6228,6256,6280,6314,6342,6378,6394
[1,5]   rank_gain_total 14, transfers 6055,6151,6174,6256,6280,6342,6368,6378
[10,13] rank_gain_total 12, transfers 6031,6181,6189,6202,6204,6314,6378
[2,4]   rank_gain_total 11, transfers 6031,6164,6189,6314,6378
```

`[10,14]` is now the leading repeated accepted-form family.  This matters for
the index-calculus work order because the strongest fresh evidence is no
longer only the priority-column recurrence; it is a repeated nonpriority
accepted-form family under the same broad top-k16 full-support geometry.

## Public selector capture

The frozen public selector mined from `5984..6023` now promotes `188` holdout
candidates through `6415`:

```text
direct-audit matches                  52
direct rank-gain matches              27
accepted-missing rank-gain matches    24
unmatched candidates                 136
```

The broad top-k16 full-support carrier has:

```text
candidates                           119
direct matches                        23
direct rank-gain matches              21
accepted-missing rank-gain matches    18
```

The soft row-geometry scorer through `6415` still has forward candidates:

```text
holdout candidates                   200
direct verified posthoc hits          23
top transfers                         6121,6163,6183,6244,6247,6255,6293,6298,6320,6367,6382,6411
```

## Interpretation

The priority work order is stable:

```text
stage 1: selected_has=13 captures all accepted priority positives with two no-rank controls
stage 2: salt_max_mod4=3 captures 6256,6280,6314,6378 with zero no-rank controls
```

The family work order strengthened:

```text
[10,14] now has 11 accepted-missing rank-gain certificates and rank_gain_total 16
fresh validating transfers: 6402,6405
```

The next useful test is whether future direct/rank tails continue to produce
top-k16 full-support `[10,14]` rank gains, and whether any such rows also
activate accepted priority column `15` or the `salt_max_mod4=3` priority
subcarrier.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The zero-control priority subcarrier covers `4/8` positives, not the full
  recurrence.
- `[10,14]` is a repeated accepted-form family, not yet a closed relation
  collection or descent policy.
- The frontier is still noncontiguous because `6288_6295` and `6344_6359`
  direct/rank are missing.
- Open queued rows are work orders until direct certificate export and rank
  audit match them.
