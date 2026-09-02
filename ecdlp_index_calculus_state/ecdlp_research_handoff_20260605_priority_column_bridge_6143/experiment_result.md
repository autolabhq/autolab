# Experiment Result: priority-column bridge through 6143

## Claim or task

Extend the priority-column bridge check after AutoLab advanced direct
certificates, rank scorers, and support scouts to `6136..6143`.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
NEGATIVE-CONTROL-REFINEMENT.

## Fresh mounted state

Mounted support scouts, direct relation certificates, and rank-scorer outputs
now reach `6143`.

The exact top work order from the `6119` pass remains unexported:

```text
6121 mode_low_term_support_total5 top_k=16 salts 174/176
```

The newly exported `6128..6143` direct rows are controls, not new progress.

## Artifacts

```text
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6143_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6143_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6143_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6143_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6143_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_priority_column_bridge_6143/
```

## Direct audit through 6143

The bridge audit now has `41` passing direct certificates, all below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN        8
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    15
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       1
SELECTED_MISSING_COLLAPSED_TO_SATURATED 17
```

Aggregate counts:

```text
rank-gain certificates                 9
accepted-missing certificates         23
accepted-missing rank gains            8
accepted priority-column certificates  1
selected priority-column certificates 41
```

The only accepted priority-column certificate is still `6117`:

```text
transfer 6117
selector mode_low_term_support_total5
top_k 16
row salts 167/170
selected support 0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
form support [11,15]
accepted missing columns 15
accepted priority columns 15
rank_gain 1
unique_factor_relation_gain 1
direct cost 0.72262774x rho
public_key_verified true
```

New direct controls past `6119`:

```text
6129 mode_low_term_support_total5 top_k=12 salts 169/172
  collapse to [2,4] and [3,5], rank_gain 0, unique_gain 2

6134 mode_low_term_support_total5 top_k=4 salts 169/171
  collapse to [3,5], rank_gain 0, unique_gain 0

6143 mode_low_term_support_total5 top_k=4 salts 174/167
  accepted missing column 6 via [5,6], rank_gain 0, unique_gain 0
```

These controls all select priority column `15`, but none accept it and none add
rank.

## Frozen-selector cross-audit

The original public selector mined only from `5984..6023` still promotes
holdout rows through `6143`.  Cross-auditing its promoted candidates against
direct evidence through `6143` gives:

```text
promoted candidates                  45
direct-audit matches                 12
direct rank-gain matches              5
accepted-missing rank-gain matches    4
unmatched candidates                 33
```

This is unchanged in the positive direction from the `6119` pass.  The new
`6128..6143` promoted rows do not add rank-gain matches.

The main full-support family now has:

```text
family mode_low_term_support_total5|16|0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
candidates 27
direct matches 4
direct rank-gain matches 3
accepted-missing rank-gain matches 2
```

It includes `6121`, `6128`, `6129`, `6135`, and `6143` as promoted forward
rows.  Their exact promoted `top_k=16` rows still lack direct matches, so they
remain work orders rather than verified failures.

## Strict forward check

Training strict public rules on direct-audited labels through `6119` and
applying them to scout rows through `6143` yields no forward candidates:

```text
strict public rules       2
strict exact row rules   10
6120..6143 scout reports 140
forward candidates        0
```

The strict rule path is still too brittle.

## Soft row-geometry scorer

The soft scorer trained through `6119` and scored `6120..6143` reports:

```text
candidate_count                 120
posthoc direct-verified rows       9
posthoc shared-verified rows       6
top score                  43.20313790
top transfers              6121,6128,6129,6135,6143
```

The top row remains the unexported exact work order:

```text
6121 mode_low_term_support_total5 top_k=16 salts 174/176
score 43.20313790
scout cost estimate 0.68613139x rho
selected support 0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

The next high-scoring promoted row is:

```text
6143 mode_low_term_support_total5 top_k=16 salts 167/174
score 40.01079708
scout cost estimate 0.64963504x rho
selected support 0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

The actual exported `6143` direct certificate was not this top-k-16 row; it was
a `top_k=4` accepted-column-6 no-rank control.

## Interpretation

The positive signal remains narrow and useful: `6117` is still the first and
only accepted priority-column bridge.  The new direct rows show that selecting
priority column `15` is cheap and common, but accepting it through a surviving
form is rare.  The campaign should now focus on reproducing the `[11,15]`
accepted-form event, not on generic full-support promotion.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The top `6121` and `6143` `top_k=16` rows are scout-only exact work orders
  until direct certificate export and rank audit arrive.
- The result is direct-key evidence, not yet shared-product/source-charged
  evidence.
- The `6128..6143` direct extension added controls, not new rank gains.
