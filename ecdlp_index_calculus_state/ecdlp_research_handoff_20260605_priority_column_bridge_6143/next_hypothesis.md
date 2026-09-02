# Next Hypothesis

## Hypothesis

The useful mechanism is not full-support selection by itself.  It is the rare
case where a full-support `top_k=16` row preserves an accepted form involving
priority column `15`; the observed instance is `6117` with form support
`[11,15]`.

## Null hypothesis

`6117` is a singleton.  Later full-support promoted rows may keep selecting
column `15` but continue collapsing to `[3,5]`, `[2,4]`, or accepted no-rank
forms that do not involve the priority column.

## Immediate command path

When direct certificates move past `6143`, refresh the bridge audit:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --state-dir /Volumes/Volume/autolab/ecdlp_index_calculus_state \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --priority-columns 15 \
  --start-min 5984 \
  --end-max <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json
```

Then cross-audit the frozen selector:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_promoted_candidate_rank_audit.py \
  --miner ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6143_probe.json \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --audit-start 6024 \
  --audit-end <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json
```

If support scouts extend before direct certificates, refresh the soft scorer:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_row_geometry_soft_scorer.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6143_probe.json \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --train-start 5984 \
  --train-end 6143 \
  --prefix-end 6023 \
  --validation-start 6024 \
  --validation-end 6143 \
  --holdout-start 6144 \
  --holdout-end <latest_scout_end> \
  --candidate-limit 120 \
  --top-k-summary 12 \
  --out ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6143_to_6144_<latest_scout_end>_probe.json
```

## Priority replay queue

```text
1. 6121 mode_low_term_support_total5 top_k=16 salts 174/176
   Frozen selector and soft scorer both promote it.  It remains unexported.

2. 6143 mode_low_term_support_total5 top_k=16 salts 167/174
   Soft score 40.01079708 and same full-support family.  The exported 6143
   top_k=4 row was a no-rank control, so the exact top_k=16 row is still open.

3. 6129 mode_low_term_support_total5 top_k=16 salts 163/169
   Soft score 39.81898244.  The exported 6129 top_k=12 row collapsed, so this
   is a control-neighborhood replay target.

4. Mine a contrast rule for the accepted-form event:
   positive = 6117 [11,15] priority-column accepted rank gain;
   controls = 6129/6134/6143 selected-priority no-rank/collapse rows.

5. Treat 6094/6090 as historical work orders only.
   They still have no direct outcome because the export stream skipped them.
```

## Promotion evidence

- A future exact `6121`, `6143`, or `6129` `top_k=16` direct export accepts
  priority column `15` and gains rank.
- A second accepted priority-column form appears, especially another support
  involving `11` and `15`.
- A contrast rule separates `[11,15]` preservation from the new
  `6129/6134/6143` no-rank controls using public row/support features.
- Shared-product/source-charged replay reproduces a priority-column bridge
  below rho.

## Demotion evidence

- Exact `top_k=16` replays keep collapsing or accepting non-priority columns.
- Future accepted priority-column hits fail to repeat beyond `6117`.
- Soft scores continue to rank direct controls above true accepted-priority
  events.
- Shared-product/source-charged replay remains absent.
