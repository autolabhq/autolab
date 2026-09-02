# Next Hypothesis

## Hypothesis

The full-support `top_k=16` public selector family can occasionally preserve an
accepted priority-column form.  The observed instance is `6117`, where the form
support `[11,15]` accepts priority column `15` and gives one new independent
factor relation below rho.

The next transfer to force through direct certificate export is `6121`, because
it is the only fresh `6120..6127` transfer promoted by the frozen selector and
it is also the top row-geometry work order.

## Null hypothesis

`6117` is a singleton direct-certificate artifact.  The `6121` rows may collapse
to saturated supports or accepted-missing-no-rank controls when rank-scored.

## Immediate command path

When direct certificates move past `6119`, refresh the bridge audit:

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
  --miner ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6127_probe.json \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --audit-start 6024 \
  --audit-end <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json
```

If support scouts extend before direct certificates, refresh the soft scorer:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_row_geometry_soft_scorer.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6119_probe.json \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --train-start 5984 \
  --train-end 6119 \
  --prefix-end 6023 \
  --validation-start 6024 \
  --validation-end 6119 \
  --holdout-start 6120 \
  --holdout-end <latest_scout_end> \
  --candidate-limit 80 \
  --top-k-summary 10 \
  --out ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_<latest_scout_end>_probe.json
```

## Priority replay queue

```text
1. 6121 mode_low_term_support_total5 top_k=16 salts 174/176
   frozen promotion selector_topk_support, soft score 43.20313790,
   scout cost 0.68613139x rho, full support including column 15.

2. 6121 mode_cost_low_term_support_total3 top_k=16 salts 174/176
   soft score 26.64929801, scout cost 0.67153285x rho,
   compact support 0,2,3,6,7,8,9,11,12,13.

3. 6121 mode_low_term_support_total3 top_k=16 salts 174/176
   soft score 26.64929801, scout cost 0.67153285x rho,
   same compact support as item 2.

4. 6121 mode_cost_hybrid_support_monic_b_total2 top_k=16 salts 174/176
   soft score 7.49931776, scout cost 0.66423358x rho,
   support 0,4,5,7,10,11,14,15 as a priority-column compact control.

5. 6094/6090 only if the direct stream later exports them.
   They were good previous work orders, but this pass has no direct evidence
   for them because the export stream skipped `6088..6095`.
```

## Promotion evidence

- A future direct window confirms `6121` as accepted-missing rank gain.
- `6121` specifically preserves a form involving column `15`, preferably
  another priority-column accepted form.
- The frozen selector continues to recover direct rank gains without retraining
  on the verified window.
- A shared-product/source-charged replay reproduces a priority-column bridge
  below rho.

## Demotion evidence

- `6121` collapses to saturated support or accepted-missing-no-rank.
- Future accepted priority-column hits fail to repeat beyond the singleton
  `6117`.
- Strict public rules keep producing zero forward candidates while soft scores
  merely chase broad selected-support features.
- Shared-product/source-charged replay remains absent.
