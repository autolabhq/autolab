# ECDLP Index-Calculus Frontier Handoff: Shared Public Subcarrier at 9695

Date: 2026-06-05

## Claim

The low-term total2 direct/rank frontier now supports a shared public subcarrier across several repeated accepted-form rank-gain families:

`salt_min_mod4=2 AND selected_has=13`

This is the cleanest current work-order rule for the summation-polynomial / FFE index-calculus campaign. It is not yet an end-to-end ECDLP speedup claim. It is a below-rho relation-export candidate backed by direct certificates and rank gains.

## Live Frontier

- Coverage audit: `low_term_total2_frontier_gap_audit_6280_9695_probe.json`
- Bridge audit: `low_term_total2_direct_missing_column_bridge_audit_5984_9695_probe.json`
- Latest complete direct/rank/scout block used here: `9688_9695`
- Contiguous complete prefix still stops at `6287`; the later frontier remains noncontiguous.

Bridge summary at `5984_9695`:

- Direct below-rho certificates: `938`
- Rank-gain certificates: `270`
- Accepted-missing-column rank gains: `240`
- Accepted-priority-column certificates: `788`

## Repeated Rank-Gain Families

From `low_term_total2_accepted_form_contrast_miner_5984_9695_probe.json`:

| Accepted form support | Rank-gain total | Accepted-missing rank gains | Unique relation gain |
| --- | ---: | ---: | ---: |
| `[11,15]` | 91 | 58 | 170 |
| `[10,14]` | 85 | 53 | 176 |
| `[0,5]` | 83 | 59 | 168 |
| `[1,5]` | 81 | 28 | 161 |
| `[2,4]` | 62 | 49 | 208 |
| `[11,13]` | 61 | 55 | 178 |

## Clean Public-Selected Carriers

The shared robust zero-control subcarrier appears in the clean public-selected families:

| Family | Positives | Rank-gain total | Unique gain | Robust tokens | Recall | Warning |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `[11,15]` | 58 | 91 | 170 | `salt_min_mod4=2`, `selected_has=13` | 0.34482759 | none |
| `[10,14]` | 53 | 85 | 176 | `salt_min_mod4=2`, `selected_has=13` | 0.24528302 | none |
| `[0,5]` | 59 | 83 | 168 | `salt_min_mod4=2`, `selected_has=13` | 0.27118644 | none |

These families all have:

- `positive_not_selected_family_count = 0`
- `selected_family_warning = null`
- zero selected-family no-rank controls under the robust rule

## Broken or Weaker Lines

The larger tail exposed several accepted-form-heavy families that should not be promoted directly as public-selected work-order rules yet:

- `[2,4]`: high unique gain (`208`) but selected-carrier mismatch; many accepted positives do not select both family columns first.
- `[11,13]`: selected-carrier mismatch; all positives in the split audit are not selected-family positives.
- Old strict public-feature promotion from `5984_6447` does not carry strongly to `6448_9695`: broad candidates still produce `17` posthoc direct hits at cap `200`, but strict promoted candidates are `0`.

## Interpretation

The current best mechanism is not a single accepted form such as `[10,14]`. It is a shared source-side carrier:

1. Public selection includes column `13`.
2. The row has `salt_min_mod4=2`.
3. The direct accepted form can then land in several productive families, especially `[11,15]`, `[10,14]`, and `[0,5]`.

This suggests the next algorithmic step should search for why that public carrier forces useful low-degree FFE relation material, rather than hard-coding one accepted support family.

