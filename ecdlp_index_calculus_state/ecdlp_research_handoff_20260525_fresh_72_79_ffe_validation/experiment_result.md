# Fresh 72-79 FFE Validation Result

## Fresh Signature Stress

Command artifact:

```text
/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/frontier_signed_eval_cover_row_side_sketch_fresh_salt_filter_rescue_guarded_relation_harvester_static_bank_shared_challenge_salt_neighborhood_public_leaf_policy_low_term_total2_target_scoped_72_79_probe.json
```

Summary:

- calibration row verified count: 189 for `balanced`, `target_balanced`, and `target_recall_first`
- chosen row selector: `target_cap3_ow0_hw1_lw0_sw0_cw0_aw0`
- stress row verified count: 23
- stress leaf verified count: 5
- stress leaf below-rho count: 4
- best stress leaf ops/rho: 0.73722628
- `target_rho_first` produced no stress leaf positives in this window

The fresh signature source is:

```text
/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/low_term_total2_signature_fresh_72_79.json
```

It contains 12 policy-distinguished below-rho cases over two targets:

- `22050.cf1@11731`: transfer 76, leaf 90, row salts 162 and 168, best ops/rho 0.73722628
- `67.a1@9803`: transfers 74 and 77, leaves 16 and 8, row salts 205, 206, and 209, best ops/rho 0.96

The audit against the old 22-surface bank found 6 genuinely new FFE surfaces
and 0 already factored fresh surfaces.

## Sage Factorization

Command artifact:

```text
/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_sage_factor_fresh_72_79_subset.json
```

Sage 10.9 result:

- requested surfaces: 6
- selected surfaces: 6
- missing requested surfaces: 0
- verified cases: 12/12
- surfaces with preserving Sage factor candidates: 6/6
- preserving Sage factor surface-FFE below rho: 6/6
- preserving Sage factor root-scan below rho: 6/6
- preserving Sage factor full-remainder below rho: 1/6
- best preserving surface FFE ops/rho: 0.37956204
- best preserving root-scan ops/rho: 0.40875912
- best preserving full-remainder ops/rho: 0.97080292
- total Sage resultant factor candidates: 107

The single full-remainder below-rho case is
`22050.cf1@11731:uniform:256:salt168` at transfer 76. Treat it as a follow-up
lead, not yet a standalone breakthrough, because its best preserving
full-remainder candidate is root-empty for the selected leaf in this materialized
surface. The public factor/root-scan route on the same surface is still charged
and below rho.

## Public Selector Scoring

Fresh 72-79 charged rollup:

```text
/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_fresh_72_79_subset_charged_rollup.json
```

Result:

- best charged policy: `summax_sage_low_coeff_max_target_hash`
- charged surfaces below rho: 6/6
- preserving selections: 6/6
- false positives: 0
- mean charged ops/rho: 0.81610706
- max charged ops/rho: 0.88321168

Combined old-plus-fresh charged rollup:

```text
/Users/adamburan/.codex/worktrees/258d/autolab/ecdlp_index_calculus_state/ffe_public_factor_quadratic_root_all_public_leaf_plus_fresh_72_79_charged_rollup.json
```

Result:

- combined surfaces: 28
- best charged policy: `summax_sage_low_constant_target_hash`
- charged surfaces below rho: 28/28
- preserving selections: 28/28
- false positives: 0
- mean charged ops/rho: 0.78788738
- max charged ops/rho: 0.968
- target split: `22050.cf1@11731` 16/16 below rho, `67.a1@9803` 12/12 below rho

## Interpretation

This is a stronger fresh-window validation of the FFE quotient line. The public
row/leaf selector survived a new transfer window, Sage factored every new FFE
surface, and an already-registered public factor order kept all 28 old-plus-new
surfaces below rho under selector-plus-root-scan accounting.

This is still not a claimed general ECDLP speedup. The result remains a
candidate relation-family route on the current toy/frontier surface bank. The
next proof burden is either a second disjoint fresh window or a stronger
root/remainder recovery path that does not rely on selected-root hit scans.
