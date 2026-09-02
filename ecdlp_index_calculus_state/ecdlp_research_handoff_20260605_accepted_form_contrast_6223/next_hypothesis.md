# Next Hypothesis

## Hypothesis

The best next line is no longer priority-column singleton hunting.  It is a
nonpriority accepted-form family:

```text
public carriers:
  mode_low_term_support_total5 top_k=16 full support
  mode_low_term_support_total5 top_k=7 support 0,4,5,6,7,10,11,14,15

posthoc target:
  accepted-form preservation around [10,13], [9,11], [8,11], and [10,14]
```

The `[10,13]` family now has rank-gain total `9`, accepted-missing count `7`,
and transfers `6031,6181,6189,6202,6204`.

## Null hypothesis

The carrier features are broad row-shape buckets.  They may continue producing
occasional direct rank gains without yielding a public rule that predicts
accepted-form preservation before direct replay.

## Immediate command path

When direct certificates move past this checkpoint, refresh the same stack with
`<latest_direct_end>`:

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

Then rerun the frozen selector and contrast:

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
1. 6202 mode_low_term_support_total5 top_k=16 salts 168/170
   Lower top-k rows at the same transfer are accepted-form rank gains;
   exact top-k-16 row is still open.

2. 6204 mode_low_term_support_total5 top_k=16 salts 163/165
   Top-k-12 exact row is a [10,13] rank gain; exact top-k-16 row is open.

3. 6163 mode_low_term_support_total5 top_k=16 salts 166/172
   Highest all-rank-gain contrast score; exact top-k-16 row still unmatched.

4. 6121 mode_low_term_support_total5 top_k=16 salts 174/176
   Persistent high soft and all-rank contrast candidate.

5. 6148 and 6184 mode_low_term_support_total5 top_k=16 salts 162/175
   Repeated public row shape across transfers; exact row still unmatched.

6. 6209 mode_low_term_support_total5 top_k=16 salts 165/171
   Fresh high soft/contrast row after the positive 6202/6204 window.

7. 6223 mode_low_term_support_total5 top_k=16 salts 162/166
   Fresh promoted full-support row at the checkpoint end.
```

## Promotion evidence

- Exact top-k-16 replay turns `6202` or `6204` into another accepted-form rank
  gain.
- `[10,13]`, `[9,11]`, or `[8,11]` repeats in future direct/rank batches.
- A public contrast rule separates these positives from selected-priority
  no-rank controls without accepted-form labels.
- Shared-product/source-charged replay reproduces an accepted-form bridge below
  rho.

## Demotion evidence

- Exact top-k-16 replays keep producing selected-priority no-rank controls.
- `[10,13]` stops repeating and remains a local toy-window artifact.
- The contrast miner keeps ranking known controls above future positives.
- Shared-product/source-charged replay remains absent.
