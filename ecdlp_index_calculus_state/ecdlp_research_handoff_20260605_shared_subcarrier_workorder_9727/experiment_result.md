# ECDLP Index-Calculus Work-Order Handoff: Shared Subcarrier Queue at 9727

Date: 2026-06-05

## Claim

The shared public subcarrier remains stable through the latest audited frontier, but the newest scout-only block does not contain a full shared-subcarrier row. The immediate direct-export queue is still the `9696_9703` hole, specifically transfer `9696`.

This remains a work-order claim, not an ECDLP speedup claim.

## Latest Frontier

- Coverage audit: `low_term_total2_frontier_gap_audit_6280_9727_probe.json`
- Bridge audit: `low_term_total2_direct_missing_column_bridge_audit_5984_9727_probe.json`
- Latest direct/rank/scout common block observed: `9712_9719`
- Scout-only block observed: `9720_9727`
- Existing direct/rank hole with shared-subcarrier rows: `9696_9703`

Bridge summary at `5984_9727`:

- Direct below-rho certificates: `942`
- Rank-gain certificates: `272`
- Accepted-missing-column rank gains: `242`
- Accepted-priority-column certificates: `790`

## Stable Public Carrier

The robust public subcarrier remains:

`salt_min_mod4=2 AND selected_has=13`

The clean selected-family lines are unchanged:

| Family | Positives | Rank-gain total | Unique gain | Robust rule |
| --- | ---: | ---: | ---: | --- |
| `[11,15]` | 58 | 91 | 170 | `salt_min_mod4=2`, `selected_has=13` |
| `[10,14]` | 53 | 85 | 176 | `salt_min_mod4=2`, `selected_has=13` |
| `[0,5]` | 59 | 83 | 168 | `salt_min_mod4=2`, `selected_has=13` |

## Work-Order Queue

Artifacts:

- `low_term_total2_shared_subcarrier_workorder_9696_9703_probe.json`
- `low_term_total2_shared_subcarrier_workorder_9720_9727_probe.json`
- `low_term_total2_shared_subcarrier_workorder_9696_9727_probe.json`

Combined queue summary from `9696_9727`:

- Scout rows: `270`
- Candidate rows with any relevant token/family match: `246`
- Full shared-subcarrier rows: `3`
- Full shared-subcarrier direct-missing rows: `3`
- Full shared-subcarrier transfers: `[9696]`

Top work-order row:

- Transfer: `9696`
- Range: `9696_9703`
- Row keys: `salt170`, `salt171`
- Selected support: `[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`
- Compatible families: `[11,15]`, `[10,14]`, `[0,5]`
- Direct status: `direct_certificate_missing`

The newest scout-only block `9720_9727` has family-compatible rows, but no full `salt_min_mod4=2 AND selected_has=13` row.

## Interpretation

The next concrete AutoLab move should not be a broad replay of every scout-only row. It should prioritize direct/rank export for the three transfer-`9696` shared-subcarrier rows first, especially the full-support `top_k=16` row that matches all three clean families.

