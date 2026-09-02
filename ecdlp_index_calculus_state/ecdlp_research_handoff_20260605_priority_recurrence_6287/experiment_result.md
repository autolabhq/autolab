# Experiment Result: priority recurrence through 6287

## Claim or task

Consume the live direct/rank/scout frontier through `6287`, refresh the
accepted-form replay stack, and update the priority-column recurrence audit
after the new `6280` multi-form hit.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-CARRIER-RECURRENCE / CHECKPOINT.

## Fresh mounted state

Direct certificates and rank-scorer outputs were complete through `6287`; scout
files were visible beyond that, through `6295`, but direct/rank were the limiting
frontier for this checkpoint.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_priority_recurrence_audit.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6287_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6287_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6287_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6287_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6287_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6287_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6287_probe.json
ecdlp_index_calculus_state/low_term_total2_priority_recurrence_audit_5984_6287_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_priority_recurrence_6287/
```

## Direct audit through 6287

The bridge audit now has `79` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       21
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    27
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 29
```

Aggregate counts:

```text
rank-gain certificates                  23
accepted-missing certificates           48
accepted-missing rank gains             21
accepted priority-column certificates    4
selected priority-column certificates   79
```

Fresh rows after `6271`:

```text
6272 top_k=16 salts 171/173 form [10,14]
  accepted missing columns 10,14
  rank_gain 1, unique_gain 1, cost 0.72262774x rho

6280 top_k=16 salts 161/175 forms [1,5], [10,14], [11,15]
  accepted missing columns 10,14,15
  accepted priority column 15
  rank_gain 3, unique_gain 13, cost 0.85401460x rho
```

`6280` is the key movement: it couples the `[1,5]`, `[10,14]`, and `[11,15]`
families in one accepted priority-column row.

## Accepted-form families

The family ranking changed:

```text
[10,14] rank_gain_total 10, accepted_missing_rank_gain_count 6
        transfers 6060,6151,6153,6164,6272,6280
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

`[10,14]` now leads by rank-gain total.  `[11,15]` is no longer a weak
priority side-note; it has four accepted priority-column hits and total
unique-gain `18`.

## Priority recurrence

Accepted priority-column 15 rank-gain rows:

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
selected_priority_no_rank_controls  56
```

Stable public tokens remain the broad carrier:

```text
selector_topk=mode_low_term_support_total5|16
selector_topk_support=mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
topk_support=16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
selected_has_13_and_15
```

Stable salt tokens are still empty.  For example:

```text
salt_has=167 positive_count 3, control_count 7, score -0.87453567
```

## Public selector capture

The frozen public selector mined from `5984..6023` promotes `117` holdout
candidates through `6287`.  Against direct evidence:

```text
direct-audit matches                  31
direct rank-gain matches              16
accepted-missing rank-gain matches    14
unmatched candidates                  86
```

The broad top-k16 public carrier now has:

```text
candidates                            75
direct matches                        13
direct rank-gain matches              12
accepted-missing rank-gain matches    10
```

The stricter `5984..6119` calibration still has no promoted selector-tier
holdout candidates, but it now emits `10` broad-atomic holdout candidates all at
`6280`, with `6` direct-verified and `3` shared-verified posthoc rows.  That is
a useful diagnostic for `6280`, not yet a hard public rule.

## Replay queues

Priority recurrence queue top transfers:

```text
6163,6255,6183,6121,6244,6247,6202,6143,6169,6250,6208,6165
```

Family planner combined top transfers:

```text
6268,6183,6244,6128,6225,6204,6189,6148,6184,6163,6255,6121
```

Per-family heads:

```text
[10,14] 6183,6244,6128,6268,6165,6248,6148,6184
[11,15] 6148,6184,6163,6255,6183,6121,6244,6247
[1,5]   6268,6148,6184,6163,6255,6183,6121,6244
[10,13] 6225,6204,6189,6163,6255,6121,6247,6250
```

## Interpretation

The strongest direction is now a coupled accepted-form/priority carrier:

```text
mode_low_term_support_total5 top_k=16 full support
target-side form cluster [1,5], [10,14], [11,15]
```

The `6280` row is the first row in this slice that looks like more than an
isolated family repeat: it bridges the side family `[1,5]`, the new rank-total
leader `[10,14]`, and the priority family `[11,15]` in one accepted
priority-column certificate.  The next experiment should replay exact top-k16
rows that look like this public carrier and then try to split the broad carrier
from its selected-priority no-rank controls.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The stable public signal is still broad top-k16/full-support, not a clean
  salt rule.
- Open queued rows are not progress until direct certificate export and rank
  audit match them.
