# Experiment Result: accepted-form contrast through 6175

## Claim or task

Consume the live direct/rank frontier through `6175`, then separate repeated
accepted-form rank gain from priority-column acceptance and no-rank controls.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-SELECTOR-CAPTURE.

## Fresh mounted state

Mounted support scouts, direct relation certificates, and rank-scorer outputs
now reach `6175`.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_accepted_form_contrast_miner.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6175_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6175_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6175_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6175_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6175_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6175_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_accepted_form_contrast_6175/
```

## Direct audit through 6175

The bridge audit now has `50` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       11
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    18
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       2
SELECTED_MISSING_COLLAPSED_TO_SATURATED 19
```

Aggregate counts:

```text
rank-gain certificates                  13
accepted-missing certificates           29
accepted-missing rank gains             11
accepted priority-column certificates    1
selected priority-column certificates   50
```

The newest accepted-missing rank gain is:

```text
6164 mode_low_term_support_total5 top_k=16 salts 162/173
  accepted missing columns 0,10,14
  form supports [0,5], [2,4], [10,14]
  rank_gain 2, unique_factor_relation_gain 6
  direct cost 0.78832117x rho
```

The newest rank-gain control is:

```text
6174 mode_low_term_support_total5 top_k=16 salts 162/174
  classification RANK_GAIN_WITHOUT_ACCEPTED_MISSING_COLUMN
  form support [1,5]
  rank_gain 1, unique_factor_relation_gain 1
  direct cost 0.79562044x rho
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

The frozen public selector mined only from `5984..6023` now promotes `63`
holdout candidates through `6175`.  Against direct evidence through `6175`:

```text
direct-audit matches                  18
direct rank-gain matches               9
accepted-missing rank-gain matches     7
unmatched candidates                  45
```

The full-support `top_k=16` family is the strongest public family:

```text
family mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
candidates 39
direct matches 8
direct rank-gain matches 7
accepted-missing rank-gain matches 5
```

The broad family now includes the verified `6117`, `6151`, `6153`, `6164`, and
`6174` rank-gain rows.  It is not specific enough to prove accepted-form
preservation, but it is a durable carrier for rank-gain replays.

## Contrast miner

The contrast miner now reports:

```text
accepted [10,14] rank-gain certs     4  transfers 6060,6151,6153,6164
accepted priority rank-gain certs    1  transfer 6117
selected-priority no-rank controls  37
rank-gain certificates              13
```

Leading form-rank summaries:

```text
[11,13] rank_gain_total 7, accepted_missing_rank_gain_count 5
[10,14] rank_gain_total 6, accepted_missing_rank_gain_count 4
[2,4]   rank_gain_total 6, accepted_missing_rank_gain_count 3
[11,15] rank_gain_total 1, accepted_missing_rank_gain_count 1
```

For `[10,14]` versus selected-priority no-rank controls, the strongest public
tokens are the full-support `top_k=16` family and column-13 support features.
Direct diagnostics recover `form_support=10,14` and accepted columns `10,14`,
but those are posthoc labels, not allowed future-selection features.

## Soft row-geometry scorer

The soft scorer trained through `6119` and scored through `6175` reports:

```text
candidate_count                 200
posthoc direct-verified rows      25
posthoc shared-verified rows      11
top score                  44.22075674
top transfers              6121,6129,6135,6143,6148,6150,6151,6154,6163,6165,6169,6173
```

The top open all-rank-gain contrast candidate is now:

```text
6163 mode_low_term_support_total5 top_k=16 salts 166/172
  contrast score 65.35415632
  direct cost estimate 0.68613139x rho
```

## Interpretation

The priority-column path is still singleton-bound, but the accepted-form bridge
line is now much stronger.  `[10,14]` has repeated four times, twice in the new
frontier after the `6143` controls, and the frozen public selector captured the
new rank-gain rows.  The next useful step is to push exact full-support
`top_k=16` exports for high contrast-scored open rows and keep the accepted-form
labeling separate from public selection features.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- Direct accepted-form rank gain is still a toy-model certificate, not a
  finished index-calculus algorithm.
- Open promoted rows are not progress until direct certificate export and rank
  audit match them.
- The priority-column accepted-form bridge has not repeated beyond `6117`.
