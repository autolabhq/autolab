# Experiment Result: priority-column bridge through 6119

## Claim or task

Refresh the direct bridge audit after new mounted AutoLab direct certificates
reached `6119`, then check whether the frozen public selector and the softer
row-geometry scorer explain the first accepted priority-column bridge.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / VERIFIED-DIRECT-CERT-SIGNAL /
WORK-ORDER-RANKING.

## Fresh mounted state

Mounted support scouts now reach `6120..6127`.  Direct relation certificates
and rank-scorer outputs now reach `6112..6119`.

The previous soft work order, `6094` with alternate `6090`, was not exported by
the direct certificate stream.  There is therefore no direct rank outcome for
those transfers in this pass.

## Artifacts

```text
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6119_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6127_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6119_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6119_to_6120_6127_probe.json
ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_6127_probe.json
ecdlp_index_calculus_state/ecdlp_research_handoff_20260605_priority_column_bridge_6119/
```

## Refreshed direct audit

The bridge audit through `6119` has `38` passing direct certificates, all below
rho.  It reports:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN        8
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    14
RANK_GAIN_WITHOUT_ACCEPTED_MISSING       1
SELECTED_MISSING_COLLAPSED_TO_SATURATED 15
```

Aggregate counts:

```text
rank-gain certificates                 9
accepted-missing certificates         22
accepted-missing rank gains            8
accepted priority-column certificates  1
selected priority-column certificates 38
```

The new priority-column certificate is:

```text
transfer 6117
selector mode_low_term_support_total5
top_k 16
row salts 167/170
selected support 0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
form support [11,15]
accepted missing columns 15
accepted priority columns 15
rank_before 9
rank_after 10
rank_gain 1
unique_factor_relation_gain 1
direct cost 0.72262774x rho
public_key_verified true
```

This is the first certificate in this bridge family where the priority column
`15` is both selected and accepted by the surviving form support.

## Frozen-selector cross-audit

The original public selector mined only from `5984..6023` still promotes
holdout rows through `6127`.  Cross-auditing its promoted candidates against
direct evidence through `6119` gives:

```text
promoted candidates                  38
direct-audit matches                 12
direct rank-gain matches              5
accepted-missing rank-gain matches    4
unmatched candidates                 26
```

The promoted set includes `6117` as a `selector_topk_support` candidate in the
full-support `top_k=16` family, with direct rank gain `1` and direct cost
`0.72262774x` rho.  This matters because the priority-column hit was captured
by a frozen public selector instead of being only an after-the-fact direct
certificate.

The same frozen selector also promotes scout-only `6121`:

```text
transfer 6121
selector mode_low_term_support_total5
top_k 16
row salts 174/176
promotion tier selector_topk_support
selected support 0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
scout cost estimate 0.68613139x rho
```

## Strict forward check

Training strict public rules on all direct-audited labels through `6119` and
applying them to scout-only `6120..6127` yields no forward candidates:

```text
strict public rules       2
strict exact row rules   10
6120..6127 scout reports 20
forward candidates        0
```

The strict rule path is therefore too brittle for the immediate next replay
queue.

## Soft row-geometry scorer

The full soft scorer trained through `6119` has `8` positive rank-gain rows and
`30` negative/control rows.  Prefix validation, trained only on `5984..6023`
and scored on direct-audited `6024..6119`, reports:

```text
validation rows                  31
accepted-missing rank gains       6
top-10 accepted-missing gains     3
top-10 total rank gains           4
top-10 transfers                  6031,6042,6043,6055,6056,6060,6087,6107,6117
```

The highest scout-only work order is again transfer `6121`:

```text
transfer 6121
selector mode_low_term_support_total5
top_k 16
row salts 174/176
selected support 0,2,3,4,5,6,7,8,9,10,11,12,13,14,15
soft score 43.20313790
scout cost estimate 0.68613139x rho
```

Top positive public tokens after training through `6119` include
`salt_pair_mod4=2,0`, `salt_gap_bucket=ge13`, and the broad full-support
`top_k=16` family around column `13`.  The `6117` hit adds the separate
`[11,15]` accepted priority-column form as the new direct target to preserve.

## Interpretation

The campaign now has a sharper candidate mechanism: the frozen public selector
can surface full-support `top_k=16` rows that sometimes preserve accepted
forms, and `6117` shows that this can reach the priority column `15` with a
real marginal rank gain.  The next direct export target is `6121`, because it
is promoted by both the frozen selector and the soft row-geometry scorer.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- This is not a proven large-field or FFE speedup.
- The `6121` rows are scout-only until direct certificate export and rank audit
  arrive.
- The result is direct-key evidence, not yet shared-product/source-charged
  evidence.
- `6094` and `6090` remain unverified because the direct stream skipped them.
