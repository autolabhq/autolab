# Experiment Result: accepted-form contrast checkpoint through 6223

## Claim or task

Consume the live direct/rank/scout frontier through `6223`, refresh the public
selector and contrast audits, and leave an honest checkpoint while the mounted
AutoLab stream continues.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-SELECTOR-CAPTURE / CHECKPOINT.

## Fresh mounted state

Mounted support scouts, direct relation certificates, and rank-scorer outputs
were complete through `6223` for this checkpoint.  The stream has been moving
continuously, so this is a checkpoint, not a terminal claim.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_accepted_form_contrast_miner.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6223_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6223_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6223_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6223_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6223_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6223_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_accepted_form_contrast_6223/
```

## Direct audit through 6223

The bridge audit now has `67` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       17
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    23
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 25
```

Aggregate counts:

```text
rank-gain certificates                  19
accepted-missing certificates           40
accepted-missing rank gains             17
accepted priority-column certificates    1
selected priority-column certificates   67
```

The fresh rank-gain rows after `6175` are:

```text
6181 top_k=7  salts 165/172  forms [9,11], [10,13]
  rank_gain 1, unique_gain 3, cost 0.78832117x rho

6186 top_k=16 salts 173/171  form [0,5]
  rank_gain 1, unique_gain 1, cost 0.72262774x rho

6189 top_k=12 salts 173/174  forms [2,4], [10,13]
  rank_gain 1, unique_gain 2, cost 0.75912409x rho

6202 top_k=7 and top_k=12 salts 168/164
  forms [8,11], [9,11], [10,13]
  each rank_gain 1, unique_gain 6, cost 0.78832117x rho

6204 top_k=12 salts 165/163  form [10,13]
  rank_gain 1, unique_gain 1, cost 0.74452555x rho
```

The rows after `6207` were controls only:

```text
6208 top_k=4 salts 174/167  form [5,6], rank_gain 0
6216 top_k=7 salts 166/167  form [8,11], rank_gain 0
```

The priority-column event remains a singleton:

```text
6117 mode_low_term_support_total5 top_k=16 salts 167/170
  accepted priority column 15
  form support [11,15]
  rank_gain 1, unique_factor_relation_gain 1
  direct cost 0.72262774x rho
```

## Accepted-form families

The contrast miner now puts `[10,13]` at the front:

```text
[10,13] rank_gain_total 9, accepted_missing_rank_gain_count 7
        transfers 6031,6181,6189,6202,6204
[11,13] rank_gain_total 7, accepted_missing_rank_gain_count 5
[2,4]   rank_gain_total 7, accepted_missing_rank_gain_count 4
[10,14] rank_gain_total 6, accepted_missing_rank_gain_count 4
[9,11]  rank_gain_total 3, accepted_missing_rank_gain_count 3
[8,11]  rank_gain_total 2, accepted_missing_rank_gain_count 2
[11,15] rank_gain_total 1, accepted_missing_rank_gain_count 1
```

This is the most important movement in the slice: `[10,13]` has overtaken the
older `[10,14]` line as the strongest repeated accepted-form signal.

## Frozen-selector cross-audit

The frozen public selector mined only from `5984..6023` now promotes `91`
holdout candidates through `6223`.  Against direct evidence:

```text
direct-audit matches                  25
direct rank-gain matches              12
accepted-missing rank-gain matches    10
unmatched candidates                  66
```

The two public carrier families are:

```text
family mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  candidates 53
  direct matches 9
  direct rank-gain matches 8
  accepted-missing rank-gain matches 6

family mode_low_term_support_total5|7|0,4,5,6,7,10,11,14,15
  candidates 38
  direct matches 16
  direct rank-gain matches 4
  accepted-missing rank-gain matches 4
```

The strict `5984..6119` forward miner still finds no holdout candidates, so the
usable selector remains the older frozen carrier plus soft row-geometry scoring.

## Soft row-geometry scorer

The soft scorer trained through `6119` and scored through `6223` reports:

```text
candidate_count                 200
posthoc direct-verified rows      32
posthoc shared-verified rows      10
top score                  44.22075674
top transfers              6121,6129,6135,6143,6154,6163,6165,6169,6183,6202,6208,6209
```

The top open all-rank-gain contrast candidates are still full-support `top_k=16`
rows.  `6202` is now notable: lower-top-k exact certificates are positive, but
the top-k-16 public row remains open.

## Interpretation

The priority-column route is still singleton-bound at `6117`.  The productive
line is now nonpriority accepted-form preservation: `[10,13]`, with related
`[9,11]` and `[8,11]` forms, is repeatedly appearing in below-rho direct/rank
certificates.  This is still toy/model-bound direct-key evidence, but it is a
better work order than continuing to wait for a second accepted priority-15
event.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- Direct accepted-form rank gain is still a toy-model certificate, not a
  finished index-calculus algorithm.
- Open promoted rows are not progress until direct certificate export and rank
  audit match them.
- The priority-column accepted-form bridge has not repeated beyond `6117`.
