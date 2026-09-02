# Experiment Result: promoted public selector rank audit through 6071

## Claim or task

Refresh the direct missing-column bridge audit after new mounted AutoLab
certificates arrived, then test whether the previously frozen public selector
actually captured held-out accepted-form rank gain.

## Status

OBSERVATION / TOY-EVIDENCE / MODEL-BOUND / SELECTOR-DIAGNOSTIC.

## Fresh mounted state

The mounted support-scout stream now reaches `6072..6079`.  Direct relation
certificates and factor-rank scorer outputs now reach `6064..6071`.

## Artifacts

```text
tasks/ecdlp_index_calculus/low_term_total2_direct_missing_column_bridge_audit.py
tasks/ecdlp_index_calculus/low_term_total2_accepted_form_public_feature_miner.py
tasks/ecdlp_index_calculus/low_term_total2_promoted_candidate_rank_audit.py
ecdlp_index_calculus_state/low_term_total2_direct_missing_column_bridge_audit_5984_6071_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6023_to_6024_6079_probe.json
ecdlp_index_calculus_state/low_term_total2_promoted_public_selector_rank_audit_6024_6071_probe.json
ecdlp_index_calculus_state/low_term_total2_accepted_form_public_feature_miner_5984_6071_to_6072_6079_probe.json
```

## Refreshed direct audit

Across `5984..6071`, the refreshed direct audit finds `27/27` passing direct
certificates, all below rho.  The classification counts are:

```text
ACCEPTED_MISSING_COLUMN_RANK_GAIN       7
ACCEPTED_MISSING_COLUMN_NO_RANK_GAIN    6
RANK_GAIN_WITHOUT_ACCEPTED_MISSING      1
SELECTED_MISSING_COLLAPSED_TO_SATURATED 13
```

The accepted-form rank-gain count is no longer a two-row curiosity.  It now
includes held-out `6028`, `6031`, and `6060` evidence after the previous
handoff.

## Frozen-selector holdout audit

The public feature miner trained only on `5984..6023` and applied to
`6024..6079` produced `18` promoted selector/top-k/support candidates.  In the
direct-audited range `6024..6071`, the cross-audit found:

```text
promoted candidates in direct-audited range        17
direct audit matches                               8
direct rank-gain matches                           4
accepted-missing rank-gain matches                 3
posthoc direct-key verified promoted candidates    8
posthoc shared-product verified promoted candidates 0
```

The direct audit over the same `6024..6071` range has `20` certificates and
`6` direct rank-gain certificates, so the frozen selector captured `4/6`
rank-gain rows and `3/5` accepted-missing rank-gain rows in the exported
certificate stream.

## Family split

The compact family remains the strongest interpretable public selector:

```text
selector = mode_low_term_support_total5
top_k    = 7
support  = 0,4,5,6,7,10,11,14,15
```

Within `6024..6071`, that family has `8` promoted candidates, `6` direct audit
matches, and `2` accepted-missing rank-gain matches:

```text
6028 top_k=7 salts 164/166 -> rank_gain 1, accepted [11,13]
6031 top_k=7 salts 162/168 -> rank_gain 2, accepted [10,13], [11,13]
```

It also has direct-audited collapses:

```text
6042 top_k=7 salts 163/172 -> collapse [3,5]
6043 top_k=7 salts 161/173 -> collapse [3,5]
6056 top_k=7 salts 166/167 -> collapse [2,4]
6063 top_k=7 salts 161/166 -> collapse [3,5]
```

The broad `top_k=16` near-full-support family is mixed rather than dead.  It
has low direct-export coverage but includes:

```text
6060 top_k=16 salts 164/165 -> rank_gain 1, accepted [10,14]
6055 top_k=16 salts 162/173 -> rank_gain 1 without accepted missing column
```

This means `top_k=16` should be retained as a diagnostic branch, but it still
needs a narrower public discriminator before it deserves priority replay.

## Expanded-calibration forward check

Training the strict public-rule miner on all direct-audited labels through
`6071` and applying it only to scout-only `6072..6079` yielded no candidates:

```text
calibration positives        7
strict public rules          3
strict exact row-key rules   8
6072..6079 holdout reports   40
6072..6079 candidates        0
```

The only strict public rules that survived zero nonpositive examples are
salt-gap tokens:

```text
salt_gap=6
salt_gap=14
salt_gap=15
```

That demotes the broad selector/top-k/support predicates from strict promotion
rules to candidate-family generators.  The next selector needs a second stage
using public row-key geometry, not selected support alone.

## Non-claims

- This is not target descent and not a deployed-curve speedup.
- Direct-key evidence is still not shared-product/source-charged evidence.
- The branch-bank baseline remains the local `5480..6007` rollup.
- Capturing rank-gain certificates is a selector advance, not a complete
  index-calculus algorithm.
