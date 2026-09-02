# Experiment Result: direct missing-column bridge audit through 6023

## Claim or task

Test whether the fresh direct-source bridge certificates after the branch
frontier can move missing factor columns into accepted relation forms, rather
than merely selecting source rows whose broad support includes those columns.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / SELECTOR-DIAGNOSTIC.

## Inputs

The duplicate-hit branch bank baseline is the local rollup through `6007`.
Mounted AutoLab direct-source certificates, selected-support scouts, and
factor-rank scorer outputs were read from `/Volumes/Volume/autolab`.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6023_probe.json
ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json
```

Mounted artifacts summarized by the audit:

```text
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_5984_5991_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_5992_5999_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_6000_6007_probe.json
ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_6016_6023_probe.json
ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json
ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json
```

## Result

Across `5984..6023`, the audit finds `7/7` passing direct-source certificates,
all below rho and all with selected supports touching missing branch-bank
columns.  Only the accepted relation forms count for rank progress:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN      2
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN   1
SELECTED_MISSING_COLLAPSED_TO_SATURATED 4
```

Accepted form supports across these certificates:

```text
[3,5]    x6
[2,4]    x1
[5,6]    x1
[11,13]  x1
[0,5]    x1
```

The two rank-gain positives are:

- Transfer `5996`, top-k `7`, direct cost `0.86131387x` rho.  Accepted forms
  include `[11,13]`; the factor-rank scorer reports rank gain `1` and unique
  factor-relation gain `2`.
- Transfer `6003`, top-k `16`, direct cost `0.75912409x` rho.  Accepted forms
  include `[0,5]`; the factor-rank scorer reports rank gain `1` and unique
  factor-relation gain `3`.

The fresh `6016..6023` certificates are useful negative controls:

- Transfer `6019`, direct cost `0.79562044x` rho.  Selected support includes
  missing columns and priority column `15`, but accepted forms collapse to
  `[3,5]`; rank gain `0`.
- Transfer `6023`, direct cost `0.74452555x` rho.  Selected support again
  includes missing columns and priority column `15`, but accepted forms collapse
  to `[3,5]`; rank gain `0`.

The `5984..6023` selected-support scouts contain `310` case reports.  All
`310` selected supports touch missing branch-bank columns, but `170` such cases
do not have an exported direct certificate in this audit horizon.  This makes
selected-support visibility much too broad as a stop rule.

## Mechanism notes

- Column `15` is selected often but appears in no accepted relation form in
  this horizon.
- One positive, transfer `5996`, accepts missing column `13` in `[11,13]` even
  though `13` is not present in its selected support set.  This shows that the
  accepted-form support can introduce a useful missing column that the selected
  support proxy did not expose.
- Transfer `5996` also has collapsed controls at top-k `4` and `12`; the
  positive top-k `7` certificate shares transfer and target context but uses a
  different row-key pair.

## Interpretation

The next public selector cannot be "selected support touches missing column" or
"selected support touches column 15"; both predicates are too permissive.  The
useful objective is accepted-form support prediction before paying full branch
setup or direct replay.  The strongest recent tokens to mine are the row-key
pair/top-k differences around transfer `5996` and the transfer `6003` bridge
that preserves column `0` in accepted forms.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- Direct-source certificates are not yet shared-product/source-charged relation
  streams.
- The rank-gain scorer uses the broader static/direct candidate baseline, not a
  completed branch-only factor bank.
- A selected missing column is not counted as progress unless the accepted
  relation form and rank scorer agree.
