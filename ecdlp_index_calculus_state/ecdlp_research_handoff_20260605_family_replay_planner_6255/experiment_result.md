# Experiment Result: family replay planner through 6255

## Claim or task

Consume the complete direct/rank/scout frontier through `6255`, refresh the
accepted-form selector/contrast stack, and add a public-token replay planner for
the repeated accepted-form families.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-REPLAY-QUEUE / CHECKPOINT.

## Fresh mounted state

Direct certificates, rank-scorer outputs, and support scouts were complete
through `6255` for this checkpoint.  Rank/scout/direct files for `6240..6255`
arrived during the run; they were consumed before the checkpoint was written.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_family_replay_planner.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6255_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6255_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6255_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6255_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6255_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6255_probe.json
ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_6255_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_family_replay_planner_6255/
```

## Direct audit through 6255

The bridge audit now has `75` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       18
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    26
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 29
```

Aggregate counts:

```text
rank-gain certificates                  20
accepted-missing certificates           44
accepted-missing rank gains             18
accepted priority-column certificates    2
selected priority-column certificates   75
```

Rows added after `6239` were controls:

```text
6240 top_k=4  salts 166/173  forms [0,2], [3,5]
  accepted missing column 0, rank_gain 0, unique_gain 2

6243 top_k=12 salts 170/174  forms [8,11], [9,11]
  accepted missing columns 8,9, rank_gain 0, unique_gain 2

6245 top_k=4/7/12 salts 163/173  form [2,4]
  saturated-collapse controls, rank_gain 0

6254 top_k=7 salts 168/171 forms [2,4], [8,11]
  accepted missing column 8, rank_gain 0, unique_gain 2
```

No new positive rank-gain transfer appeared after `6228`.

## Accepted-form families

The contrast summary stayed stable under the added controls:

```text
[10,13] rank_gain_total 9, accepted_missing_rank_gain_count 7
        transfers 6031,6181,6189,6202,6204
[11,13] rank_gain_total 7, accepted_missing_rank_gain_count 5
[2,4]   rank_gain_total 7, accepted_missing_rank_gain_count 4
[10,14] rank_gain_total 6, accepted_missing_rank_gain_count 4
[9,11]  rank_gain_total 3, accepted_missing_rank_gain_count 3
[8,11]  rank_gain_total 2, accepted_missing_rank_gain_count 2
[11,15] rank_gain_total 2, accepted_missing_rank_gain_count 2
        transfers 6117,6228
```

The important change from the older checkpoint is that `[11,15]` is no longer a
singleton: `6228` repeated the priority-column accepted-form pattern.

## Frozen-selector cross-audit

The frozen selector mined from `5984..6023` promotes `108` holdout candidates
through `6255`.  Against direct evidence:

```text
direct-audit matches                  28
direct rank-gain matches              13
accepted-missing rank-gain matches    11
unmatched candidates                  80
```

The two public carrier families remain:

```text
family mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  candidates 66
  direct matches 10
  direct rank-gain matches 9
  accepted-missing rank-gain matches 7

family mode_low_term_support_total5|7|0,4,5,6,7,10,11,14,15
  candidates 42
  direct matches 18
  direct rank-gain matches 4
  accepted-missing rank-gain matches 4
```

The stricter `5984..6119` calibration still promotes zero holdout candidates.

## Family replay planner

Added `low_term_total2_family_replay_planner.py`.  It uses direct labels only to
seed and evaluate repeated families, while open-candidate ordering uses public
row tokens, selected support, salts, frozen selector score, and soft geometry
score.

Planner summary:

```text
claim_status FAMILY_REPLAY_PLANNER_HAS_REPEATED_TARGET_FAMILIES_AND_OPEN_QUEUE
open_candidate_count 207
open_queue_count 24
repeated_target_family_count 5
```

Combined top open transfers:

```text
6183,6244,6128,6225,6204,6189,6163,6255,6121,6247,6250,6165
```

Per-family top transfer queues:

```text
[10,13] 6225,6204,6189,6163,6255,6121,6247,6250
[11,15] 6163,6255,6183,6121,6244,6247,6143,6202
[10,14] 6183,6244,6128,6165,6248,6163,6255,6121
[9,11]  6163,6255,6183,6121,6244,6247,6143,6202
[8,11]  6163,6255,6183,6121,6244,6247,6143,6202
```

An open candidate is unmatched by exact row identity; another row at the same
transfer may already have a direct certificate.  The intended next action is
direct-certificate export for the exact queued rows.

## Interpretation

The best line remains accepted-form preservation around `[10,13]`, with related
`[9,11]` and `[8,11]` rows at the `6202` positive cluster.  The priority-column
path is stronger than before because `[11,15]` repeated at `6228`, but it is
still only two toy direct certificates.  The post-6239 controls did not weaken
the main signal enough to change the queue.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- Direct accepted-form rank gain is a toy-model certificate, not a finished
  index-calculus algorithm.
- Open queued rows are not progress until direct certificate export and rank
  audit match them.
