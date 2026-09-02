# Next Hypothesis

## Hypothesis

The strongest next experiment is exact-row replay for public top-k rows that
look like the repeated accepted-form families before direct labels are known:

```text
primary form family:    [10,13]
priority form family:   [11,15]
supporting families:    [10,14], [9,11], [8,11]
public carriers:
  mode_low_term_support_total5 top_k=16 full support
  mode_low_term_support_total5 top_k=7 support 0,4,5,6,7,10,11,14,15
```

The `[10,13]` family remains the strongest toy signal with accepted-missing
rank-gain count `7`, rank-gain total `9`, unique-gain total `36`, and transfers
`6031,6181,6189,6202,6204`.

## Null hypothesis

The top-k16 full-support lane is a broad shape bucket.  It may rank many
below-rho direct rows without separating accepted-form rank gain from
accepted-missing/no-rank or saturated controls.

## Immediate command path

When direct/rank/scout files move past this checkpoint, refresh with
`<latest_direct_end>`:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --priority-columns 15 \
  --start-min 5984 \
  --end-max <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json
```

Then rerun the frozen selector, rank audit, contrast, and family planner:

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

python3 tasks/ecdlp_index_calculus/low_term_total2_family_replay_planner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --promoted-audit ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json \
  --soft-scorer ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6119_to_6120_<latest_direct_end>_probe.json \
  --families '10,13;11,15;10,14;9,11;8,11' \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --open-start 6120 \
  --open-end <latest_direct_end> \
  --open-candidate-limit 24 \
  --out ecdlp_index_calculus_state/low_term_total2_family_replay_planner_5984_<latest_direct_end>_probe.json
```

## Priority replay queue

```text
1. 6183 top_k=16 salts 164/165
   Combined planner leader; shares the [10,14] positive salt pair.

2. 6244 top_k=16 salts 164/165
   Fresh post-6239 open row with the same top public replay signature.

3. 6225 top_k=16 salts 163/165
   Top [10,13] family queue row; shares a [10,13] positive salt pair.

4. 6204 top_k=16 salts 163/165
   Same transfer as a positive [10,13] top-k12 row, exact top-k16 row open.

5. 6189 top_k=16 salts 173/174
   Same transfer as positive [10,13]/[2,4] direct certificates.

6. 6163 top_k=16 salts 166/172
   Persistent high soft/contrast candidate and top priority-family queue row.

7. 6255 top_k=16 salts 166/168
   Fresh post-6239 queue row in the top 8 for every target family.

8. 6121,6247,6250 top_k=16
   Remaining high-ranked exact-row replays before broadening the carrier search.
```

## Promotion evidence

- Exact queued rows direct-certify as accepted-missing rank gains below rho.
- `[10,13]` repeats outside the `6181..6204` cluster.
- `[11,15]` produces a third priority-column accepted-form rank gain.
- A public token split ranks future positives above the post-6239 controls
  `6240`, `6243`, `6245`, and `6254`.

## Demotion evidence

- Top-k16 replay keeps landing on accepted-missing/no-rank or saturated controls.
- `[10,13]` remains bounded to the same toy transfer window.
- `[11,15]` stays at exactly two direct certificates.
- Shared-product/source-charged replay remains absent.
