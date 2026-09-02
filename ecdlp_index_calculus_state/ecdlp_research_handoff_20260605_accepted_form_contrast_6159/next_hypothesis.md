# Next Hypothesis

## Hypothesis

The full-support `top_k=16` selector is a useful carrier, but the actual
algorithmic lead is accepted-form preservation.  The repeated `[10,14]`
accepted-form rank gains give a larger training signal than the singleton
`[11,15]` priority-column hit, so the next branch should mine public row
geometry that distinguishes:

```text
positive family A: [10,14] accepted-form rank gains at 6060,6151,6153
positive family B: [11,15] accepted-priority singleton at 6117
controls: selected-priority no-rank/collapse rows, especially 6129,6134,6143,6150
```

## Null hypothesis

Full-support `top_k=16` promotion is too broad.  It captures some rank gains
but cannot separate `[10,14]` or `[11,15]` preservation from saturated/no-rank
controls using public information alone.

## Immediate command path

When direct certificates move past `6159`, refresh the bridge audit:

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

Then rerun the frozen-selector rank audit:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_promoted_candidate_rank_audit.py \
  --miner ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6159_probe.json \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --audit-start 6024 \
  --audit-end <latest_direct_end> \
  --out ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json
```

Rerun the contrast miner:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_accepted_form_contrast_miner.py \
  --bridge-audit ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest_direct_end>_probe.json \
  --promoted-audit ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_<latest_direct_end>_probe.json \
  --priority-columns 15 \
  --audit-start 5984 \
  --audit-end <latest_direct_end> \
  --open-start 6120 \
  --open-end <latest_direct_end> \
  --top-tokens 30 \
  --open-candidate-limit 20 \
  --out ecdlp_index_calculus_state/low_term_total2_accepted_form_contrast_miner_5984_<latest_direct_end>_probe.json
```

## Priority replay queue

```text
1. 6121 mode_low_term_support_total5 top_k=16 salts 174/176
   Top soft row, high contrast score, still exact-unexported.

2. 6148 mode_low_term_support_total5 top_k=16 salts 162/175
   High all-rank-gain contrast score and unexported exact row.

3. 6154 mode_low_term_support_total5 top_k=16 salts 164/166
   Same full-support family, still unexported exact row.

4. 6135 mode_low_term_support_total5 top_k=16 salts 164/169
   Highest [10,14]-contrast open score, exact row still unmatched.

5. Continue treating 6143 and 6129 as control neighborhoods unless their exact
   top_k=16 rows export; nearby lower-top-k exports were controls.
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
