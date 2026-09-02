# Next Hypothesis

## Hypothesis

Public row-key salt geometry can act as a second-stage work-order ranker after
selector/top-k/support-family filtering.  The next replay target should be
`6094`, because it combines the broad `top_k=16` column-13 family with a
positive row-key geometry score.

## Null hypothesis

The soft score is mostly selected-support leakage and will replay into
saturated `[3,5]` or accepted-missing-no-rank controls, similar to `6087`.

## Immediate command path

When direct certificates move past `6087`, refresh the bridge audit:

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
  --miner ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6095_probe.json \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --audit-start 6024 \
  --audit-end <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json
```

If support scouts extend before direct certificates, refresh the soft scorer:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_row_geometry_soft_scorer.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6087_probe.json \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --train-start 5984 \
  --train-end 6087 \
  --prefix-end 6023 \
  --validation-start 6024 \
  --validation-end 6087 \
  --holdout-start 6088 \
  --holdout-end <latest_scout_end> \
  --candidate-limit 50 \
  --top-k-summary 10 \
  --out ecdlp_index_calculus_state/low_term_total2_row_geometry_soft_scorer_5984_6087_to_6088_<latest_scout_end>_probe.json
```

## Priority replay queue

```text
1. 6094 mode_low_term_support_total5 top_k=16 salts 173/176
2. 6094 mode_cost_low_term_support_total3 top_k=16 salts 173/176
3. 6094 mode_low_term_support_total3 top_k=16 salts 173/176
4. 6090 mode_low_term_support_total5 top_k=16 salts 171/173
5. 6090 mode_low_term_support_total5 top_k=7 salts 171/173 as compact-family control
```

## Promotion evidence

- `6094` direct replay yields accepted-missing rank gain against the branch-bank
  baseline.
- A future direct window confirms that positive soft scores outperform the
  `6087` accepted-missing-no-rank controls.
- Shared-product/source-charged replay reproduces the direct-key result below
  rho.
- The same row-key geometry pattern fires on a later scout-only window and
  survives direct audit.

## Demotion evidence

- `6094` collapses to `[3,5]` or accepted missing with no rank gain.
- The top soft scores remain dominated by broad `top_k=16` selected-support
  features without accepted-form preservation.
- Strict public rules continue to have no forward candidates.
- Shared-product/source-charged replay remains absent.
