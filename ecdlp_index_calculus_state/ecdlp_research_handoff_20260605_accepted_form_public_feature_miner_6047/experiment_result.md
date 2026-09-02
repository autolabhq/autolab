# Experiment Result: accepted-form public feature miner through 6047

## Claim or task

Mine public/source-side features from the direct missing-column bridge audit
through `6023`, then apply the frozen rules to support-scout holdout rows
`6024..6047`.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / SELECTOR-DIAGNOSTIC.

## Inputs

The calibration labels come from the local direct missing-column bridge audit:

```text
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6023_probe.json
```

Fresh holdout rows come from mounted selected-support scouts:

```text
/Volumes/Volume/autolab/ecdlp_index_calculus_state/low_term_total2_selected_leaf_term_support_scout_22050_col15_selector_expanded_*_probe.json
```

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_accepted_form_public_feature_miner.py
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6047_probe.json
```

## Result

Calibration has seven labeled direct-source certificates:

```text
positive_rank_gain             2
accepted_missing_dependent     1
saturated_collapse             4
```

The miner found `30` strict public rules and `4` exact row-key rules.  Exact
row-key rules are reported for auditability but are not used for holdout
promotion.

On `210` holdout support-scout reports covering `6024..6047`, public rules
produce:

```text
all holdout candidates       60
promoted holdout candidates  10
posthoc direct verified       4
posthoc shared verified       0
```

Promoted candidates are restricted to selector/top-k/support-family matches.
The useful promoted family is:

```text
selector = mode_low_term_support_total5
top_k    = 7
support  = 0,4,5,6,7,10,11,14,15
```

That family has four posthoc direct-key verified holdout rows:

```text
transfer 6028, cost 0.79562044x rho, salts 164/166
transfer 6031, cost 0.86131387x rho, salts 162/168
transfer 6042, cost 0.72262774x rho, salts 163/172
transfer 6043, cost 0.79562044x rho, salts 161/173
```

Two same-family promoted rows are not posthoc direct verified and should be
used as immediate controls:

```text
transfer 6030, cost 0.64963504x rho, salts 165/166
transfer 6040, cost 0.72262774x rho, salts 161/163
```

The broad `top_k=16` family with near-full support and column `13` scores high
under atomic tokens, but its promoted holdout rows at transfers `6028`, `6031`,
`6032`, and `6042` are posthoc direct false.  Treat that family as a diagnostic
for overly broad support predicates, not as a promoted replay target.

## Interpretation

This is progress beyond "selected support touches a missing column."  A frozen
public selector rule from the `5984..6023` calibration window finds a compact
holdout family with repeated posthoc direct-key verification through `6047`.

The result still does not prove accepted-form rank gain on holdout rows.  The
next gate is direct certificate export/replay for the four verified `top_k=7`
rows and rank-scorer measurement against the branch-bank baseline.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- Posthoc direct-key verification is reported but not used for rule ranking.
- Shared-product/source-charged verification remains absent in this holdout.
- Broad atomic rules such as `selected_has_13` are not sufficient promotion
  evidence.
