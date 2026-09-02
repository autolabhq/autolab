# Next Hypothesis

## Hypothesis

The current live branch should use a two-carrier selector:

```text
primary public carrier: full-support mode_low_term_support_total5 top_k=16
secondary public carrier: mode_low_term_support_total5 top_k=7 support 0,4,5,6,7,10,11,14,15
posthoc target: accepted-form preservation, especially [10,14], [10,13], and [11,15]
```

The `[10,14]` family has repeated on transfers `6060`, `6151`, `6153`, and
`6164`; the new `6181` row reinforces `[10,13]` and adds `[9,11]`; `[11,15]`
remains the priority-column singleton at `6117`.

## Null hypothesis

The public carriers are broad row-shape buckets.  They may keep finding
occasional rank gains, but they may not yield a public rule that distinguishes
accepted-form preservation from selected-priority no-rank controls.

## Immediate command path

When direct certificates move past `6183`, refresh the bridge audit:

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

Then rerun the frozen-selector miner, rank audit, soft scorer, and contrast
miner:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_accepted_form_public_feature_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6023_probe.json \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --calibration-start 5984 \
  --calibration-end 6023 \
  --holdout-start 6024 \
  --holdout-end <latest_direct_end> \
  --candidate-limit 200 \
  --out ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_promoted_candidate_rank_audit.py \
  --miner ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_<latest_direct_end>_probe.json \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --candidate-field promoted_holdout_candidates \
  --audit-start 6024 \
  --audit-end <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_row_geometry_soft_scorer.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --train-start 5984 \
  --train-end 6119 \
  --prefix-end 6023 \
  --validation-start 6024 \
  --validation-end 6119 \
  --holdout-start 6120 \
  --holdout-end <latest_direct_end> \
  --candidate-limit 200 \
  --top-k-summary 12 \
  --out ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_<latest_direct_end>_probe.json

python3 tasks/ecdlp_index_calculus/low_term_total2_accepted_form_contrast_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --promoted-audit ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json \
  --priority-columns 15 \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --open-start 6120 \
  --open-end <latest_direct_end> \
  --top-tokens 30 \
  --open-candidate-limit 24 \
  --out ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_<latest_direct_end>_probe.json
```

## Priority replay queue

```text
1. 6163 mode_low_term_support_total5 top_k=16 salts 166/172
   Top all-rank-gain contrast candidate; exact row still unmatched.

2. 6128 mode_low_term_support_total5 top_k=16 salts 162/173
   Top [10,14] contrast candidate; exact row still unmatched.

3. 6121 mode_low_term_support_total5 top_k=16 salts 174/176
   Persistent high soft and all-rank contrast candidate; exact row still unmatched.

4. 6148 mode_low_term_support_total5 top_k=16 salts 162/175
   High all-rank contrast candidate; exact row still unmatched.

5. 6165 mode_low_term_support_total5 top_k=16 salts 165/176
   High [10,14] and priority-singleton contrast candidate; exact row still open.

6. 6183 mode_low_term_support_total5 top_k=16 salts 164/165
   Fresh high soft and contrast row; current exact lower-top-k variants collapse.

7. 6182 mode_low_term_support_total5 top_k=16 salts 162/166
   Fresh soft-score row in the top-12 holdout queue.
```

## Promotion evidence

- A future exact `top_k=16` replay produces another `[10,14]` or `[10,13]`
  accepted-form rank gain.
- A future exact replay produces a second `[11,15]` accepted-priority hit.
- A public contrast rule separates repeated accepted-form positives from
  selected-priority no-rank controls without direct accepted-form tokens.
- Shared-product/source-charged replay reproduces an accepted-form bridge below
  rho.

## Demotion evidence

- Exact top-k-16 replays keep producing selected-priority no-rank controls.
- `[10,14]`, `[10,13]`, and `[9,11]` remain confined to the current windows.
- The contrast miner keeps ranking known control neighborhoods above future
  positives.
- Shared-product/source-charged replay remains absent.
