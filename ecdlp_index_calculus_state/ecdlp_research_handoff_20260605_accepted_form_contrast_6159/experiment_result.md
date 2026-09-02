# Experiment Result: accepted-form contrast through 6159

## Claim or task

Consume the new direct/rank frontier through `6159`, then separate the
priority-column singleton from the repeated non-priority accepted-form rank
gains using public contrast features.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
PUBLIC-SELECTOR-CAPTURE.

## Fresh mounted state

Mounted support scouts, direct relation certificates, and rank-scorer outputs
now reach `6159`.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_accepted_form_contrast_miner.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6159_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6159_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6159_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6159_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6159_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_6159_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_accepted_form_contrast_6159/
```

## Direct audit through 6159

The bridge audit now has `46` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       10
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    17
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       1
SELECTED_MISSING_COLLAPSED_TO_SATURATED 18
```

Aggregate counts:

```text
rank-gain certificates                  11
accepted-missing certificates           27
accepted-missing rank gains             10
accepted priority-column certificates    1
selected priority-column certificates   46
```

The new `6144..6159` rank-gain certificates are:

```text
6151 mode_low_term_support_total5 top_k=16 salts 174/164
  accepted missing columns 10,14
  form supports [1,5] and [10,14]
  rank_gain 2, unique_factor_relation_gain 3
  direct cost 0.75912409x rho

6153 mode_low_term_support_total5 top_k=16 salts 173/171
  accepted missing columns 10,14
  form support [10,14]
  rank_gain 1, unique_factor_relation_gain 1
  direct cost 0.72262774x rho
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

The frozen public selector mined only from `5984..6023` now promotes `51`
holdout candidates through `6159`.  Against direct evidence through `6159`:

```text
direct-audit matches                  14
direct rank-gain matches               7
accepted-missing rank-gain matches     6
unmatched candidates                  37
```

The full-support `top_k=16` family is now the strongest public family:

```text
family mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
candidates 32
direct matches 6
direct rank-gain matches 5
accepted-missing rank-gain matches 4
```

That family includes the verified `6117`, `6151`, and `6153` rows, so the
selector is no longer only carrying one priority-column hit.  It is capturing a
repeated accepted-form rank-gain surface.

## Contrast miner

The new contrast miner separates public tokens from direct-only diagnostic
tokens.  Its direct summaries show:

```text
accepted [10,14] rank-gain certs     3  transfers 6060,6151,6153
accepted priority rank-gain certs    1  transfer 6117
selected-priority no-rank controls  35
```

The leading rank-gain form supports are:

```text
[11,13] rank_gain_total 7, accepted_missing_rank_gain_count 5
[3,5]   rank_gain_total 4, accepted_missing_rank_gain_count 4
[10,14] rank_gain_total 4, accepted_missing_rank_gain_count 3
[11,15] rank_gain_total 1, accepted_missing_rank_gain_count 1
```

For `[10,14]` versus selected-priority no-rank controls, the strongest public
tokens are not accepted-form labels; they are the full-support `top_k=16`
family plus column-13 support features.  Direct diagnostic tokens correctly
recover `form_support=10,14` and accepted columns `10,14`, but those are
posthoc and must not be used as future selectors.

## Soft row-geometry scorer

The soft scorer trained through `6119` ranked the new verified rows high:

```text
6151 mode_low_term_support_total5 top_k=16 score 34.13624518
6153 mode_low_term_support_total5 top_k=16 score 32.07415914
```

The unexported scout/exact work order `6121 top_k=16 salts 174/176` remains the
top soft row at `43.20313790`.

## Interpretation

The priority-column path is still not repeated, but the broader accepted-form
bridge is now stronger: the frozen public selector and soft row geometry both
capture fresh direct rank gains after the `6143` control window.  The most
productive next step is to stop treating all full-support rows as equivalent
and mine the contrast between `[10,14]` repeated gains, `[11,15]` priority
singleton, and no-rank selected-priority controls.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- Direct accepted-form rank gain is still a toy-model certificate, not a
  finished index-calculus algorithm.
- Open promoted rows are not progress until direct certificate export and rank
  audit match them.
- The priority-column accepted-form bridge has not repeated beyond `6117`.
