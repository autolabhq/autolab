# Next Hypothesis

## Hypothesis

Accepted-form support can be predicted from public row-key/top-k/source
features better than from selected support alone.

The calibration target is to distinguish:

```text
positive: transfer 5996, top-k 7, accepted [11,13], rank gain 1
positive: transfer 6003, top-k 16, accepted [0,5], rank gain 1
negative: transfer 5996, top-k 4/12, accepted [3,5], rank gain 0
negative: transfer 6019/6023, accepted [3,5], rank gain 0
```

## Null hypothesis

The useful accepted-form support is not public-predictable from the current
source artifacts.  It may require verifier outcomes, direct replay labels, or
rowspace construction that cannot be moved before setup.

## Immediate command path

Refresh the audit when new direct-source certificates appear:

```bash
python3 tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py \
  --branch-frontier ecdlp_index_calculus_state/low_term_total2_branch_family_frontier_rollup_5480_6007_probe.json \
  --state-dir /Volumes/Volume/autolab/ecdlp_index_calculus_state \
  --certificates '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_direct_relation_equation_certificate_22050_col15_lowterm_support5_*_probe.json' \
  --rank-scorers '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_factor_rank_candidate_scorer_target67_22050_multibranch_plus_priority_hash6_plus_direct_col15_lowterm_support5_*_probe.json' \
  --support-scouts '/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json' \
  --priority-columns 15 \
  --start-min 5984 \
  --end-max <latest> \
  --out ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_<latest>_probe.json
```

## Promotion evidence

- A frozen public accepted-form predictor selects `[11,13]` or `[0,5]` style
  accepted supports before direct replay labels.
- The selector rejects `[3,5]` collapse cases at `6019` and `6023`.
- The selected direct/source rows remain below rho after source-generation or
  shared-product charges.
- A fresh target can map into the resulting factor bank.

## Demotion evidence

- The predictor only keys off exact transfer/salt replay.
- Positive and collapsed controls remain indistinguishable before direct replay.
- Accepted-form missing columns continue to appear only after outcome labels.
- Column `15` remains selected-only and never survives into accepted forms.
