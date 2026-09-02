# Experiment Result: accepted-form contrast through 6183

## Claim or task

Consume the live direct/rank frontier through `6183`, then refresh the
accepted-form contrast and public-selector capture tests.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-SELECTOR-CAPTURE / FRONTIER-UPDATED.

## Fresh mounted state

Mounted support scouts, direct relation certificates, and rank-scorer outputs
now reach `6183`.  This supersedes the `6175` handoff as the current live
frontier.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_accepted_form_contrast_miner.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6183_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6183_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6183_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6183_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6183_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6183_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_accepted_form_contrast_6183/
```

## Direct audit through 6183

The bridge audit now has `55` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       12
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    18
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 23
```

Aggregate counts:

```text
rank-gain certificates                  14
accepted-missing certificates           30
accepted-missing rank gains             12
accepted priority-column certificates    1
selected priority-column certificates   55
```

The new accepted-missing rank gain after `6175` is:

```text
6181 mode_low_term_support_total5 top_k=7 salts 165/172
  accepted missing columns 9,10,13
  form supports [9,11], [10,13]
  rank_gain 1, unique_factor_relation_gain 3
  direct cost 0.78832117x rho
```

The other new rows are collapse controls:

```text
6178 mode_low_term_support_total5 top_k=4 salts 163/175
  form support [2,4], rank_gain 0

6183 mode_low_term_support_total5 top_k=4/7/12 salts 165/164
  form support [3,5], rank_gain 0
```

The priority-column event remains a singleton:

```text
6117 mode_low_term_support_total5 top_k=16 salts 167/170
  accepted priority column 15
  form support [11,15]
  rank_gain 1, unique_factor_relation_gain 1
  direct cost 0.72262774x rho
```

## Frozen-selector cross-audit

The frozen public selector mined only from `5984..6023` now promotes `68`
holdout candidates through `6183`.  Against direct evidence through `6183`:

```text
direct-audit matches                  20
direct rank-gain matches              10
accepted-missing rank-gain matches     8
unmatched candidates                  48
```

Two public carrier families are now visible:

```text
family mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  candidates 42
  direct matches 8
  direct rank-gain matches 7
  accepted-missing rank-gain matches 5

family mode_low_term_support_total5|7|0,4,5,6,7,10,11,14,15
  candidates 26
  direct matches 12
  direct rank-gain matches 3
  accepted-missing rank-gain matches 3
```

The second family captured the new `6181` accepted-missing rank gain.  It is
therefore a live public carrier, but still not a public proof of accepted-form
preservation.

## Contrast miner

The contrast miner now reports:

```text
accepted [10,14] rank-gain certs     4  transfers 6060,6151,6153,6164
accepted priority rank-gain certs    1  transfer 6117
accepted-missing rank-gain certs    12
selected-priority no-rank controls  41
rank-gain certificates              14
```

Leading form-rank summaries:

```text
[11,13] rank_gain_total 7, accepted_missing_rank_gain_count 5
[10,14] rank_gain_total 6, accepted_missing_rank_gain_count 4
[2,4]   rank_gain_total 6, accepted_missing_rank_gain_count 3
[10,13] rank_gain_total 5, accepted_missing_rank_gain_count 3
[9,11]  rank_gain_total 1, accepted_missing_rank_gain_count 1
[11,15] rank_gain_total 1, accepted_missing_rank_gain_count 1
```

The new `6181` row expands the nonpriority accepted-form line: it adds
`[9,11]` and reinforces `[10,13]`.  The `[10,14]` family remains repeated but
does not gain a fifth member in this window.

## Soft row-geometry scorer

The soft scorer trained through `6119` and scored through `6183` reports:

```text
candidate_count                 200
posthoc direct-verified rows      29
posthoc shared-verified rows      11
top score                  44.22075674
top transfers              6121,6129,6135,6143,6148,6150,6154,6163,6165,6169,6182,6183
```

The top open all-rank-gain contrast candidate remains:

```text
6163 mode_low_term_support_total5 top_k=16 salts 166/172
  contrast score 64.39383907
  direct cost estimate 0.68613139x rho
```

Fresh high-priority open rows include:

```text
6183 mode_low_term_support_total5 top_k=16 salts 164/165
  all-rank contrast score 61.0571676
  soft score 42.43819244
  direct cost estimate 0.68613139x rho

6182 mode_low_term_support_total5 top_k=16 salts 162/166
  soft score 35.41958237
  direct cost estimate 0.79562044x rho
```

The exact top-k-16 `6183` row is still open: the current direct certificates for
`6183` are top-k 4, 7, and 12 collapse controls.

## Interpretation

The priority-column path is still singleton-bound at `6117`.  The nonpriority
accepted-form bridge line strengthened again: `6181` is a fresh direct/rank
certificate below rho, captured by the frozen public selector, and it expands
the `[10,13]`/`[9,11]` accepted-form evidence.  The immediate work should focus
on exact top-k-16 replay/export for the open full-support rows, while keeping
public source-side selection separate from posthoc accepted-form labels.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- Direct accepted-form rank gain is still a toy-model certificate, not a
  finished index-calculus algorithm.
- Open promoted rows are not progress until direct certificate export and rank
  audit match them.
- The priority-column accepted-form bridge has not repeated beyond `6117`.
