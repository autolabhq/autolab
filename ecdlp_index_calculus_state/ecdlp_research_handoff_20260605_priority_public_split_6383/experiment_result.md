# Experiment Result: priority public split through 6383

## Claim or task

Validate the `6343` priority public split against the newly arrived
`6360..6383` direct/rank tail.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / PUBLIC-SPLIT-SUBCARRIER /
LIVE-TAIL-VALIDATED / NONCONTIGUOUS-FRONTIER-CHECKPOINT.

## Fresh mounted state

Direct/rank/scout common evidence now reaches `6376_6383`, but the frontier is
still noncontiguous:

```text
contiguous common end        6287
latest common complete end   6383
missing recent ranges        6288_6295, 6344_6351, 6352_6359
new complete tail            6360_6367, 6368_6375, 6376_6383
```

The `6360..6383` tail is valid evidence, but not a contiguous frontier advance.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_priority_public_split_miner.py
ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_6383_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6383_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_public_split_miner_5984_6383_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6383_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6383_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6383_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_priority_public_split_6383/
```

## New tail rows

Fresh rows after `6343`:

```text
6367 top_k=7 salts 169/171 form [9,11]
  accepted missing column 9, rank_gain 0

6368 top_k=16 salts 163/173 form [1,5]
  rank_gain 1 without accepted missing column

6371 top_k=4 salts 173/174 form [3,5]
  saturated collapse, rank_gain 0

6371 top_k=7 salts 166/174 forms [3,5], [8,11]
  accepted missing column 8, rank_gain 0

6378 top_k=7 salts 171/175 forms [2,4], [9,11], [10,13]
  accepted missing columns 9,10,13
  rank_gain 1, unique_gain 5

6378 top_k=16 salts 165/171 forms [1,5], [11,15]
  accepted missing column 15 and accepted priority column 15
  rank_gain 2, unique_gain 3

6381 top_k=7/top_k=12 salts 161/174 forms [2,4], [8,11]
  accepted missing column 8, rank_gain 0
```

The important row is `6378` top-k16: it is a new accepted priority-column 15
rank-gain row, and it satisfies the `salt_max_mod4=3 AND selected_has=13`
subcarrier mined at `6343`.

## Priority split validation

The split miner result through `6383`:

```text
priority positives                         7
selected-priority no-rank controls        72
selected-priority other rank-gain rows    25
priority rank_gain_total                  12
priority unique_gain_total                27
```

Full-recall carrier:

```text
selected_has=13
positive_count 7 / 7
control_count 2 / 72
precision 0.77777778
controls: 6087,6319
```

Robust zero-control subcarrier:

```text
salt_max_mod4=3 AND selected_has=13
positive_count 4 / 7
control_count 0 / 72
precision 1.0
recall 0.57142857
positive transfers: 6256,6280,6314,6378
```

This is a real validation event: `6378` was not in the `6343` split-miner input
and lands exactly in the zero-control subcarrier.

## Family movement

Accepted-form contrast through `6383`:

```text
[1,5]   rank_gain_total 14, transfers 6055,6151,6174,6256,6280,6342,6368,6378
[10,13] rank_gain_total 12, transfers 6031,6181,6189,6202,6204,6314,6378
[10,14] rank_gain_total 12, transfers 6060,6151,6153,6164,6272,6280,6298,6320
[11,15] rank_gain_total 12, transfers 6117,6228,6256,6280,6314,6342,6378
[2,4]   rank_gain_total 11, transfers 6031,6164,6189,6314,6378
```

`[1,5]` has overtaken `[10,14]` by rank total, and `[11,15]` now has seven
accepted priority-column hits.

## Public selector capture

The frozen public selector mined from `5984..6023` now promotes `173` holdout
candidates through `6383`:

```text
direct-audit matches                  45
direct rank-gain matches              24
accepted-missing rank-gain matches    21
unmatched candidates                 128
```

The broad top-k16 full-support carrier has:

```text
candidates                           110
direct matches                        20
direct rank-gain matches              18
accepted-missing rank-gain matches    15
```

## Interpretation

The priority line has moved from “interesting recurrence” to “validated
subcarrier work order”:

```text
stage 1: selected_has=13 captures all accepted priority positives with two no-rank controls
stage 2: salt_max_mod4=3 captures 6256,6280,6314,6378 with zero no-rank controls
```

The next direct/rank tails should be scored first for this subcarrier.  A
future no-rank row satisfying the same public tokens demotes the rule; another
accepted priority row satisfying it promotes the rule from local subcarrier
toward a replay policy.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The zero-control subcarrier covers `4/7` positives, not the full recurrence.
- The frontier is still noncontiguous because `6288_6295` and `6344_6359`
  direct/rank are missing.
- Open queued rows are work orders until direct certificate export and rank
  audit match them.
