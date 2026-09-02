# Next Hypothesis

## Hypothesis

The current live branch should move from priority-column singleton hunting to a
two-layer selector:

```text
public carrier: full-support mode_low_term_support_total5 top_k=16
posthoc target: accepted-form preservation, especially [10,14] and [11,15]
```

The `[10,14]` family has now repeated on transfers `6060`, `6151`, `6153`, and
`6164`, while `[11,15]` remains the priority-column singleton at `6117`.

## Null hypothesis

Full-support promotion is only a broad carrier.  It will keep finding occasional
rank gains but will not yield a public rule that distinguishes `[10,14]` or
`[11,15]` accepted-form preservation from selected-priority controls.

## Immediate command path

When direct certificates move past `6175`, refresh the bridge audit:

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

Then rerun the frozen-selector rank audit and contrast miner:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_promoted_candidate_rank_audit.py \
  --miner ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6175_probe.json \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
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
1. 6163 mode_low_term_support_total5 top_k=16 salts 166/172
   Top all-rank-gain contrast candidate, exact row still unmatched.

2. 6121 mode_low_term_support_total5 top_k=16 salts 174/176
   Persistent top soft row and high contrast candidate, still unexported exact row.

3. 6148 mode_low_term_support_total5 top_k=16 salts 162/175
   High all-rank-gain contrast candidate, still unexported exact row.

4. 6165 mode_low_term_support_total5 top_k=16 salts 165/176
   High [10,14] and priority-singleton contrast candidate, still open.

5. 6169 mode_low_term_support_total5 top_k=16 salts 164/169
   Re-enters the [10,14] contrast queue as a fresh exact row.
```

## Promotion evidence

- A future exact `top_k=16` replay produces another `[10,14]` accepted-form
  rank gain or a second `[11,15]` accepted-priority hit.
- A public contrast rule separates repeated `[10,14]` positives from
  selected-priority no-rank controls without direct accepted-form tokens.
- Shared-product/source-charged replay reproduces an accepted-form bridge below
  rho.

## Demotion evidence

- Exact top-k-16 replays keep producing selected-priority no-rank controls.
- `[10,14]` stops repeating and remains confined to the current window.
- The contrast miner keeps ranking known control neighborhoods above future
  positives.
- Shared-product/source-charged replay remains absent.
