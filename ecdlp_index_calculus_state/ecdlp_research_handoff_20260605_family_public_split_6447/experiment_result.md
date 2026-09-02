# Experiment Result: family public split through 6447

## Claim or task

Turn the leading `[10,14]` accepted-form recurrence into a public split and
fresh-tail replay test, then refresh the low-term total2 stack through the
latest complete direct/rank/scout evidence.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / FAMILY-PUBLIC-SPLIT /
NONPRIORITY-FAMILY-LEAD / NONCONTIGUOUS-FRONTIER-CHECKPOINT.

## Fresh mounted state

Direct/rank/scout common evidence now reaches `6440_6447`, but the frontier is
still noncontiguous:

```text
contiguous common end        6287
latest common complete end   6447
missing recent ranges        6288_6295, 6344_6351, 6352_6359, 6416_6423, 6432_6439
new complete tail            6424_6431, 6440_6447
```

The scout stream has `6448_6455`, but matching direct/rank files were not
present at this checkpoint.  The latest complete bridge evidence is therefore
`6447`.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_family_public_split_miner.py
ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_6447_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6447_probe.json
ecdlp_index_calculus_state/low_term_total2_family_public_split_miner_10_14_5984_6447_replay6416_probe.json
ecdlp_index_calculus_state/low_term_total2_family_public_split_miner_11_15_5984_6447_replay6416_probe.json
ecdlp_index_calculus_state/low_term_total2_family_public_split_miner_1_5_5984_6447_replay6416_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_public_split_miner_5984_6447_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6447_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6447_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6447_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6447_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6447_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6447_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_family_public_split_6447/
```

## New tail rows

Fresh rows after `6415`:

```text
6430 top_k=16 salts 170/172 forms [0,5], [1,5], [10,14]
  accepted missing columns 0,10,14
  rank_gain 2, unique_gain 6

6442 top_k=12 salts 161/167 forms [9,11], [11,13]
  accepted missing columns 9,13
  rank_gain 1, unique_gain 2

6447 top_k=7 salts 165/173 forms [9,11], [10,13]
  accepted missing columns 9,10,13
  rank_gain 1, unique_gain 2
```

`6430` is the important validation point: it is another top-k16 full-support
`[10,14]` rank-gain row, and it arrived after the `6415` handoff.

## Family split

The `[10,14]` family split through `6447`:

```text
target-family positives              12
selected-family no-rank controls     85
selected-family other rank-gain      26
rank_gain_total                      18
unique_gain_total                    41
positive transfers                   6060,6151,6153,6164,6272,6280,6298,6320,6394,6402,6405,6430
```

Full-recall carrier:

```text
selected_has=13
positive_count 12 / 12
control_count 2 / 85
precision 0.85714286
```

Robust zero-control subcarrier:

```text
salt_min_mod4=2 AND selected_has=13
positive_count 3 / 12
control_count 0 / 85
precision 1.0
recall 0.25
positive transfers: 6164,6405,6430
```

Prefix replay from `<6416` to `>=6416`:

```text
train positives       11
replay positives       1
replay transfer        6430
replay controls        3
status                 PREFIX_HIGH_RECALL_CARRIER_VALIDATED_ON_TAIL
```

The pre-6416 high-recall carrier rules, including `selected_has=13` and
top-k16 full support, selected the fresh `6430` `[10,14]` positive with zero
replay controls.  No pre-6416 robust zero-control subcarrier validated on the
tail; the current salt subcarrier is a full-fit local split that must be tested
on later direct/rank evidence.

## Comparison lines

The `[11,15]` family remains the priority recurrence:

```text
target-family positives              8
rank_gain_total                     14
best zero-control subcarrier         salt_max_mod4=3 AND selected_has=13
subcarrier positive transfers        6256,6280,6314,6378
```

The `[1,5]` family is not a comparable selected-support split:

```text
target-family positives              6
positive selected-family count       0
claim_status                         FAMILY_PUBLIC_SPLIT_HAS_SELECTED_CARRIER_MISMATCH
```

Every `[1,5]` positive is missing at least one family column from the public
selected support, so its apparent zero-control rules are not comparable to the
`[10,14]` selected-family controls.

## Contrast stack

Accepted-form contrast through `6447`:

```text
[10,14] rank_gain_total 18, transfers 6060,6151,6153,6164,6272,6280,6298,6320,6394,6402,6405,6430
[1,5]   rank_gain_total 16, transfers 6055,6151,6174,6256,6280,6342,6368,6378,6430
[11,15] rank_gain_total 14, transfers 6117,6228,6256,6280,6314,6342,6378,6394
[10,13] rank_gain_total 13, transfers 6031,6181,6189,6202,6204,6314,6378,6447
[2,4]   rank_gain_total 11, transfers 6031,6164,6189,6314,6378
```

The frozen public selector now promotes `200` holdout candidates, with:

```text
direct-audit matches                  53
direct rank-gain matches              29
accepted-missing rank-gain matches    26
```

The top-k16 full-support promoted family has:

```text
candidates                           127
direct matches                        24
direct rank-gain matches              22
accepted-missing rank-gain matches    19
```

## Interpretation

`[10,14]` is now the strongest selected-support-aligned accepted-form family.
It has a broad public carrier that repeatedly survives fresh direct/rank tails,
and it has a small current zero-control salt subcarrier that is worth testing
when `6448+` direct/rank arrives.

This is still not a completed index-calculus speedup.  It is a sharper
relation-harvesting work order: prioritize top-k16 full-support rows with
`selected_has=13`, then track whether `[10,14]` or the
`salt_min_mod4=2 AND selected_has=13` subcarrier keeps producing below-rho
rank gain.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The `[10,14]` zero-control subcarrier covers `3/12` positives, not the full
  recurrence.
- The pre-6416 replay validates the broad carrier, not the current salt
  subcarrier.
- The frontier is still noncontiguous because five direct/rank ranges are
  missing.
- Open queued rows are work orders until direct certificate export and rank
  audit match them.
