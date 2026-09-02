# Experiment Result: row-geometry soft scorer to 6095

## Claim or task

Refresh the direct bridge audit after new mounted AutoLab artifacts appeared,
then add a softer public row-key geometry scorer for scout-only rows where the
strict zero-nonpositive miner has no forward candidates.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / WORK-ORDER-RANKING.

## Fresh mounted state

Mounted support scouts now reach `6088..6095`.  Direct relation certificates
and rank-scorer outputs now reach `6080..6087`.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_row_geometry_soft_scorer.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6087_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6095_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6087_to_6088_6095_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6087_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6087_to_6088_6095_probe.json
```

## Refreshed direct audit

The bridge audit through `6087` now has `31` passing direct certificates, all
below rho:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       7
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    9
RANK_GAIN_WITHOUT_ACCEPTED_MISSING      1
SELECTED_MISSING_COLLAPSED_TO_SATURATED 14
```

The new `6080..6087` direct window added useful controls but no new rank gain.
At transfer `6087`, accepted forms expose missing columns `6` and `8`, but the
rank scorer reports `rank_gain=0`; the `top_k=16` row collapses to `[3,5]`.

## Frozen-selector cross-audit

The original frozen public selector trained through `6023` now proposes `25`
promoted candidates through `6095`.  In the direct-audited range `6024..6087`,
the cross-audit reports:

```text
promoted candidates      22
direct-audit matches     10
direct rank-gain matches 4
accepted-missing gains   3
```

This preserves the previous positive result but does not improve it with the
new `6087` controls.  The direct-audited stream over the same range has `6`
rank-gain certificates, so the frozen selector still captures `4/6` rank-gain
rows and `3/5` accepted-missing rank-gain rows.

## Strict forward check

Training strict public rules on all direct-audited labels through `6087` and
applying them to scout-only `6088..6095` yields no forward candidates:

```text
strict public rules       3
strict exact row rules    8
6088..6095 scout reports  40
forward candidates        0
```

The strict evidence is still too brittle for the immediate next replay queue.

## Soft row-geometry scorer

The new scorer uses smoothed public token likelihoods over selector/top-k,
selected support, cost bucket, and row-key salt geometry.  It reports scores as
work-order rankings only; posthoc verification fields are not used as features.

Prefix validation trained only on `5984..6023` and scored direct-audited
`6024..6087`:

```text
validation rows                  24
accepted-missing rank gains       5
top-10 accepted-missing gains     3
top-10 total rank gains           4
top-10 controls/collapses         6
```

This is not a zero-false-positive selector, but it does surface a useful mix of
rank-gain positives and fresh controls.

The full scorer trained through `6087` ranks scout-only `6088..6095` rows.  The
highest-priority direct export/replay work order is:

```text
transfer 6094
selector mode_low_term_support_total5
top_k 16
salts 173/176
selected support 0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
score 14.50438699
direct cost 0.72262774x rho
```

Secondary rows for the same source salts:

```text
6094 mode_cost_low_term_support_total3 top_k=16 score 10.22709380 cost 0.67153285x rho
6094 mode_low_term_support_total3      top_k=16 score 10.22709380 cost 0.67153285x rho
```

The best alternate transfer is:

```text
6090 mode_low_term_support_total5 top_k=16 salts 171/173 score 8.96027681 cost 0.72262774x rho
```

The frozen promoted selector also flags:

```text
6090 mode_low_term_support_total5 top_k=7 salts 171/173 cost 0.75912409x rho
```

That row has a negative soft score, so it is a useful control for the compact
`top_k=7` family rather than the top replay target.

## Interpretation

The campaign now has a cleaner next step: direct-certificate export and rank
scoring for `6094` first, then `6090` as an alternate/control.  The strongest
soft features are still row-key/salt geometry plus the broad `top_k=16`
column-13 support family, not a strict public accepted-form rule.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- Soft scores are not proof of accepted-form rank gain.
- Scout-only rows require direct certificate export and rank audit before they
  count as progress.
- Direct-key evidence is still not shared-product/source-charged evidence.
