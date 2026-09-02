# Experiment Result: noncontiguous tail through 6327

## Claim or task

Consume the live direct/rank/scout tail through `6327`, keep the frontier gap
explicit, and refresh the accepted-form/priority recurrence stack after the new
`6314` and `6320` rank-gain rows.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-CARRIER-RECURRENCE / NONCONTIGUOUS-FRONTIER-CHECKPOINT.

## Fresh mounted state

The recent-window gap audit shows a complete but noncontiguous tail:

```text
common complete ranges       6280_6287, 6296_6303, 6304_6311, 6312_6319, 6320_6327
contiguous common end        6287
latest common complete end   6327
missing any range            6288_6295
noncontiguous complete tail  6296_6303, 6304_6311, 6312_6319, 6320_6327
```

The `6288_6295` scout artifact is present, but the direct-certificate and
rank-scorer artifacts for that range are still missing.  The `6327` evidence
below is valid as noncontiguous tail evidence, not as a contiguous frontier
advance.

## Calibration fix

`low_term_total2_accepted_form_public_feature_miner.py` now restricts
calibration labels to bridge certificates whose support report is inside the
requested calibration window.  Without this guard, later bridge certificates
leak into the rule-mining calibration set and collapse the frozen selector
diagnostic when newer tail files arrive.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_frontier_gap_audit.py
tasks/ecdlp_index_calculus/low_term_total2_accepted_form_public_feature_miner.py
ecdlp_index_calculus_state/low_term_total2_frontier_gap_audit_6280_6327_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6327_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6327_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6327_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6327_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6327_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6327_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6327_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6327_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_noncontiguous_tail_6327/
```

## Direct audit through 6327

The bridge audit has `91` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       26
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    30
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 33
```

Aggregate counts:

```text
rank-gain certificates                  28
accepted-missing certificates           56
accepted-missing rank gains             26
accepted priority-column certificates    5
selected priority-column certificates   91
```

Fresh rows after `6303`:

```text
6307 top_k=4 salts 175/169 form [3,5]
  saturated collapse, rank_gain 0, cost 0.81021898x rho

6311 top_k=12 salts 165/164 form [8,11]
  accepted missing column 8
  rank_gain 0, unique_gain 0, cost 0.75182482x rho

6314 top_k=7 salts 174/175 forms [2,4], [10,13]
  accepted missing columns 10,13
  rank_gain 1, unique_gain 2, cost 0.75912409x rho

6314 top_k=12 salts 175/174 forms [2,4], [10,13]
  accepted missing columns 10,13
  rank_gain 1, unique_gain 2, cost 0.75912409x rho

6314 top_k=16 salts 174/175 forms [2,4], [11,15]
  accepted missing column 15 and accepted priority column 15
  rank_gain 1, unique_gain 3, cost 0.79562044x rho

6319 top_k=16 salts 173/172 form [2,4]
  saturated collapse, rank_gain 0, cost 0.71532847x rho

6320 top_k=16 salts 165/164 form [10,14]
  accepted missing columns 10,14
  rank_gain 1, unique_gain 1, cost 0.75912409x rho

6321 top_k=4 salts 162/167 form [2,4]
  saturated collapse, rank_gain 0, cost 0.81751825x rho
```

`6314` is the new priority recurrence event: it adds the fifth accepted
priority-column 15 rank-gain row.  `6320` reinforces `[10,14]`.

## Accepted-form families

The family ranking after `6327` is:

```text
[10,14] rank_gain_total 12, accepted_missing_rank_gain_count 8
        transfers 6060,6151,6153,6164,6272,6280,6298,6320
[10,13] rank_gain_total 11, accepted_missing_rank_gain_count 9
        transfers 6031,6181,6189,6202,6204,6314
[2,4]   rank_gain_total 10, accepted_missing_rank_gain_count 7
        transfers 6031,6164,6189,6314
[1,5]   rank_gain_total 9, accepted_missing_rank_gain_count 3
        transfers 6055,6151,6174,6256,6280
[11,15] rank_gain_total 8, accepted_missing_rank_gain_count 5
        transfers 6117,6228,6256,6280,6314
[11,13] rank_gain_total 7, accepted_missing_rank_gain_count 5
[9,11]  rank_gain_total 3, accepted_missing_rank_gain_count 3
[8,11]  rank_gain_total 2, accepted_missing_rank_gain_count 2
```

`[10,14]` remains the rank-total leader.  `6314` is important because it
simultaneously strengthens `[10,13]`, `[2,4]`, and the priority `[11,15]`
line.

## Priority recurrence

Accepted priority-column 15 rank-gain rows:

```text
6117 top_k=16 salts 167/170 form [11,15] rank_gain 1 unique_gain 1
6228 top_k=16 salts 167/176 form [11,15] rank_gain 1 unique_gain 1
6256 top_k=16 salts 166/167 forms [1,5],[11,15] rank_gain 2 unique_gain 3
6280 top_k=16 salts 161/175 forms [1,5],[10,14],[11,15] rank_gain 3 unique_gain 13
6314 top_k=16 salts 174/175 forms [2,4],[11,15] rank_gain 1 unique_gain 3
```

Priority recurrence audit:

```text
priority_positive_count              5
priority_positive_rank_gain_total    8
priority_positive_unique_gain_total 21
selected_priority_no_rank_controls  63
```

Stable public tokens are still the broad carrier:

```text
selector_topk=mode_low_term_support_total5|16
selector_topk_support=mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
topk_support=16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
selected_has_13_and_15
```

Stable salt tokens remain empty.

## Public selector capture

The frozen public selector mined from `5984..6023` promotes `139` holdout
candidates through `6327`.  Against direct evidence:

```text
direct-audit matches                  37
direct rank-gain matches              20
accepted-missing rank-gain matches    18
unmatched candidates                 102
```

The broad top-k16 public carrier now has:

```text
candidates                            89
direct matches                        17
direct rank-gain matches              15
accepted-missing rank-gain matches    13
```

The stricter `5984..6119` calibration still has no promoted selector-tier
holdout candidates.  It emits the same `10` broad-atomic diagnostics at `6280`,
with `6` direct-verified and `3` shared-verified posthoc rows.

## Replay queues

Family planner combined top transfers:

```text
6297,6287,6268,6183,6244,6128,6225,6204,6189,6148,6184,6163
```

Per-family heads:

```text
[10,13] 6297,6287,6225,6204,6189,6163,6255,6293
[11,15] 6287,6148,6184,6163,6255,6183,6293,6121
[10,14] 6183,6244,6128,6268,6165,6248,6148,6163
[9,11]  6297,6163,6255,6183,6293,6121,6244,6247
[8,11]  6163,6255,6183,6293,6121,6244,6247,6143
[1,5]   6268,6148,6184,6163,6255,6183,6293,6121
```

Priority recurrence queue top transfers:

```text
6163,6255,6183,6293,6121,6244,6247,6143,6202,6169,6250,6208
```

Transfers in `6288..6295`, including `6293`, are scout-only until the missing
direct/rank bridge is produced.

## Interpretation

The best line is now a three-part queue:

1. Fill `6288_6295` direct/rank artifacts so the frontier can become
   contiguous.
2. Probe the `6314` mechanism: it is the first new priority-column hit after
   `6280` and links `[2,4]`, `[10,13]`, and `[11,15]`.
3. Keep replaying top-k16 `[10,14]` rows, now reinforced by both `6298` and
   `6320`.

The current algorithmic hypothesis is still a broad public top-k16/full-support
carrier with accepted-form structure discovered after direct certification.
The next useful improvement is a second-stage public split that separates the
five priority positives from the 63 selected-priority no-rank controls.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The `6327` checkpoint is not a contiguous frontier advance until `6288_6295`
  direct and rank artifacts exist.
- The public carrier is broad; no clean salt rule has been validated.
- Open queued rows are not progress until direct certificate export and rank
  audit match them.
