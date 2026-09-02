# ECDLP FFE External Signature Stress - 2026-05-25

## Scope

The current FFE quotient route closes the default 19 positive low-term total-2
signature cases, which materialize into 18 FFE surfaces. This handoff expands
the input signature bank by mining all public-leaf policy artifacts already in
the live campaign state, then identifies which extra cases are genuine new FFE
surfaces versus duplicates of already factored surfaces.

## Artifacts

- All-public-leaf signature source:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_all_public_leaf_sources.json`
- External-source audit:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_external_signature_source_audit.py`
- External-source audit output:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_external_signature_source_audit.json`
- Narrow wide-stress-only signature source for the next Sage run:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_wide_external_only.json`
- Selected-surface Sage wrapper:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_sage_factor_surface_subset_probe.py`
- Charged public selector rollup:
  `/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_public_factor_selector_charged_rollup.py`
- Sage factorization over all public-leaf sources:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_all_public_leaf_sources.json`
- Public quadratic selector over all public-leaf Sage factors:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_all_public_leaf_sources.json`
- Charged selector rollup over all public-leaf Sage factors:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_all_public_leaf_sources_charged_rollup.json`
- Narrow Sage factorization and charged selector rollup for the four new wide
  external surfaces:
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_wide_external_subset.json`
  `/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_wide_external_subset_charged_rollup.json`

## Result

Mining all public-leaf artifacts increases the below-rho signature bank:

- default signature cases: 19
- all-public-leaf signature cases: 33
- external cases not in default bank: 14
- external surfaces: 6
- external surfaces already factored by the default Sage bank: 2
- external surfaces still requiring Sage factorization: 4

The six external cases from the broad 48/55 diagnostic reuse already factored
`67.a1@9803` transfer-55 surfaces:

- `67.a1@9803:uniform:256:salt204`
- `67.a1@9803:uniform:256:salt205`

The only genuinely new FFE surfaces come from the wide-stress transfer 9/10
window on `22050.cf1@11731`:

- `22050.cf1@11731|22050.cf1@11731:uniform:256:salt165|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:9:22050.cf1@11731`
- `22050.cf1@11731|22050.cf1@11731:uniform:256:salt174|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:9:22050.cf1@11731`
- `22050.cf1@11731|22050.cf1@11731:uniform:256:salt173|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:10:22050.cf1@11731`
- `22050.cf1@11731|22050.cf1@11731:uniform:256:salt174|ecdlp-frontier-signed-dual-sieve-v1:shared-transfer:10:22050.cf1@11731`

## Sage Factorization Result

The broad Sage run over all 33 all-public-leaf cases completed and replaced
the earlier quick-window uncertainty:

```sh
DOT_SAGE=/private/tmp/codex_sage_userdir \
PYTHONPYCACHEPREFIX=/tmp/autolab_pycache \
PYTHONPATH=/Volumes/Volume/autolab/tasks/ecdlp_index_calculus \
sage --python /Volumes/Volume/autolab/tasks/ecdlp_index_calculus/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_sage_factor_probe.py \
  --signature-source /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_all_public_leaf_sources.json \
  --out /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_all_public_leaf_sources.json
```

Sage 10.9 results:

- verified cases: 33/33
- factored surfaces: 22
- surfaces with a preserving Sage resultant factor: 22/22
- preserving factor root-scan below rho: 22/22
- preserving factor surface FFE below rho: 22/22
- full remainder FFE below rho: 0/22
- total Sage resultant factor candidates: 419
- best preserving root scan: 0.47445255 ops/rho
- best preserving surface FFE: 0.37956204 ops/rho
- best full remainder FFE: 1.0729927 ops/rho

The narrow wrapper also verified the four genuinely new wide-stress surfaces:

- requested surfaces: 4
- selected surfaces: 4
- missing requested surfaces: 0
- verified cases: 8/8
- preserving Sage factor root-scan below rho: 4/4
- full remainder FFE below rho: 0/4
- best preserving root scan: 0.47445255 ops/rho
- worst selected public charged path after rollup: 0.74452555 ops/rho

## Public Selector Result

The public quadratic selector recovered a preserving public factor for every
expanded surface and produced no false-positive source rows. Its built-in
charged quadratic fields remain empty for these synthetic signature sources
because they do not carry full surface records, so the follow-up charged rollup
joins selector rows back to the Sage generic-rho counts.

Charged rollup definition:

```text
charged ops/rho = (public selector eval ops + selected factor root-scan ops)
                  / Sage generic Pollard-rho steps
```

Expanded all-public-leaf bank:

- best charged policy: `summax_sage_low_constant_target_hash`
- charged surfaces below rho: 22/22
- preserving selections: 22/22
- false positives: 0
- mean charged ops/rho: 0.7825083
- max charged ops/rho: 0.968
- target split: `22050.cf1@11731` 14/14 below rho, `67.a1@9803` 8/8 below rho

Four new wide external surfaces:

- best charged policy: `global_fingerprint_hash`
- charged surfaces below rho: 4/4
- preserving selections: 4/4
- false positives: 0
- mean charged ops/rho: 0.69160584
- max charged ops/rho: 0.74452555

This is a stronger external-source validation than the prior 18-surface current
bank: true finite-field factorization found preserving resultant factors on all
22 expanded surfaces, and an already-registered public order keeps every
selector-plus-root-scan path below rho.

## Reproduction Commands

Narrow selected-surface Sage run:

```sh
DOT_SAGE=/private/tmp/codex_sage_userdir \
PYTHONPYCACHEPREFIX=/tmp/autolab_pycache \
PYTHONPATH=/Volumes/Volume/autolab/tasks/ecdlp_index_calculus \
/usr/local/bin/sage --python /Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_sage_factor_surface_subset_probe.py \
  --signature-source /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_wide_external_only.json \
  --audit-source /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_external_signature_source_audit.json \
  --out /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_wide_external_subset.json
```

Public selector and charged rollup for the expanded bank:

```sh
PYTHONPYCACHEPREFIX=/tmp/autolab_pycache \
PYTHONPATH=/Volumes/Volume/autolab/tasks/ecdlp_index_calculus \
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
/Volumes/Volume/autolab/tasks/ecdlp_index_calculus/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_low_term_total2_ffe_public_factor_quadratic_root_probe.py \
  --sage-factor-source /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_all_public_leaf_sources.json \
  --signature-source /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_all_public_leaf_sources.json \
  --out /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_all_public_leaf_sources.json

/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
/Users/adamburan/.codex/worktrees/258d/autolab/tasks/ecdlp_index_calculus/ffe_public_factor_selector_charged_rollup.py \
  --sage-factor-source /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_all_public_leaf_sources.json \
  --public-selector-source /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_all_public_leaf_sources.json \
  --out /Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_all_public_leaf_sources_charged_rollup.json
```

## Boundary And Next Work

This remains a candidate below-rho relation-family result, not a claimed
general ECDLP speedup. The current win is a public FFE-resultant factor
selection plus root-scan route on the expanded toy/frontier surface bank. Full
resultant remainders still do not beat rho, and the next validation must use
fresh rows or a larger independent holdout rather than only mining already
materialized public-leaf artifacts.

Best next step: promote the charged rollup into the live public selector probe
so the script reports selector-plus-root-scan ops directly, then generate fresh
wide-stress rows for both `22050.cf1@11731` and `67.a1@9803` and require the
same all-below-rho, all-preserving, false-positive-free result without reusing
this artifact-mined bank.
