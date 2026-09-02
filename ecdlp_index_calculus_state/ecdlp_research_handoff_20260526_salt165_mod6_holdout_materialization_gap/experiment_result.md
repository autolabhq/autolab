# Salt165 Mod6 Holdout Materialization Gap

Date: 2026-05-26

Status: holdout selector exposure found; exact Sage replay blocked by missing witness specs. This is not a promoted ECDLP speedup.

## Frozen Selector

The selector frozen from the full-remainder sparsity miner was:

```text
target = 22050.cf1@11731
row_key = 22050.cf1@11731:uniform:256:salt165
transfer_index mod 6 = 0
```

## New Artifacts

- Coverage audit script:
  `tasks/ecdlp_index_calculus/ffe_full_remainder_salt165_holdout_coverage_audit.py`
- Exact-profile wrapper update:
  `tasks/ecdlp_index_calculus/ffe_sage_factor_exact_profile_subset_probe.py`
  now supports `--allow-materialization-errors` so failed exact-profile replays write JSON diagnostics.
- Coverage audit output:
  `ecdlp_index_calculus_state/ffe_full_remainder_salt165_mod6_holdout_coverage_audit_376_671.json`
- First exact-profile diagnostic attempts:
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_salt165_mod6_holdout_transfer378_376_383.json`
  - `ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_salt165_mod6_holdout_transfer420_416_423.json`

## Commands

Coverage audit:

```bash
/Users/adamburan/.cache/vllm-venvs/vllm-312/bin/python tasks/ecdlp_index_calculus/ffe_full_remainder_salt165_holdout_coverage_audit.py --out ecdlp_index_calculus_state/ffe_full_remainder_salt165_mod6_holdout_coverage_audit_376_671.json
```

Transfer 378 exact-profile diagnostic:

```bash
HOME=/private/tmp/codex-sage-home DOT_SAGE=/private/tmp/codex-dot-sage /usr/local/bin/sage --python tasks/ecdlp_index_calculus/ffe_sage_factor_exact_profile_subset_probe.py --allow-materialization-errors --signature-source ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_376_383.json --profile '22050.cf1@11731|378|16|fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0|mode_cost_low_term_support_total3|22050.cf1@11731:uniform:256:salt165|79' --profile '22050.cf1@11731|378|16|fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0|mode_low_term_support_total3|22050.cf1@11731:uniform:256:salt165|79' --profile '22050.cf1@11731|378|16|fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0|mode_cost_low_term_support_total4|22050.cf1@11731:uniform:256:salt165|65,79' --profile '22050.cf1@11731|378|16|fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0|mode_low_term_support_total4|22050.cf1@11731:uniform:256:salt165|79' --out ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_salt165_mod6_holdout_transfer378_376_383.json
```

Transfer 420 exact-profile diagnostic:

```bash
HOME=/private/tmp/codex-sage-home DOT_SAGE=/private/tmp/codex-dot-sage /usr/local/bin/sage --python tasks/ecdlp_index_calculus/ffe_sage_factor_exact_profile_subset_probe.py --allow-materialization-errors --signature-source ecdlp_index_calculus_state/low_term_total3_total4_public_bounded_full_selector_416_423.json --profile '22050.cf1@11731|420|12|fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0|mode_cost_low_term_support_total3|22050.cf1@11731:uniform:256:salt165|90' --profile '22050.cf1@11731|420|12|fixed_target_cap3_ow0_hw1_lw0_sw0_cw0_aw0|mode_cost_low_term_support_total4|22050.cf1@11731:uniform:256:salt165|90' --out ecdlp_index_calculus_state/ffe_sage_factor_exact_profiles_22050_salt165_mod6_holdout_transfer420_416_423.json
```

## Coverage Result

The frozen public selector appears in later selector artifacts. Across selector windows 376-671, the audit found:

| metric | value |
| --- | ---: |
| selector artifacts scanned | 60 |
| frozen-selector profiles | 96 |
| distinct selected transfers | 12 |
| neighbor-control profiles | 1720 |
| neighbor-control salts | 16 |
| exact attempts written | 2 |
| exact attempts materialized | 0 |

Selected transfer indices:

```text
378, 420, 438, 480, 540, 588, 594, 606, 612, 618, 630, 660
```

Leaf signatures among the 96 selected profiles:

| leaf signature | count |
| --- | ---: |
| `90` | 51 |
| `79` | 20 |
| `8,90` | 18 |
| `65,79` | 4 |
| `34,90` | 3 |

## Exact Replay Result

The first two post-boundary selected transfers did not materialize:

| transfer | requested profiles | materialized profiles | materialization errors | repeated error |
| ---: | ---: | ---: | ---: | --- |
| 378 | 4 | 0 | 4 | `candidate spec not found` |
| 420 | 2 | 0 | 2 | `candidate spec not found` |

Each failed profile also ended with `row_key_missing_from_replay_context`, because the fallback replay context had no available row keys for the selected `salt165` row.

## Interpretation

This is a useful demotion of the previous next step. The frozen selector is not merely absent; it recurs frequently in the future public selector bank. The blocker is narrower and more actionable: the current exact-profile Sage path cannot reconstruct the selected `salt165` row/leaf surface from the witness spec source, even when the public-bounded selector artifact already records the row profile.

That means the next algorithmic task is not more rule mining. It is a forced-witness materializer for public-bounded row profiles:

```text
selector artifact row_leaf_keys -> build row context directly -> Sage factor surface -> full-remainder FFE charge
```

Until that path exists, the selector cannot be validated or falsified on the 12 future transfer exposures.
