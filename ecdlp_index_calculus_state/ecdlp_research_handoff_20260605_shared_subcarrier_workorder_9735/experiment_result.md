# ECDLP Index-Calculus Work-Order Handoff: Shared Subcarrier Queue at 9735

Date: 2026-06-05

## Claim

The shared public subcarrier remains the best concrete queue rule:

`salt_min_mod4=2 AND selected_has=13`

At the latest audited boundary, it has both:

- direct-missing queue rows at transfer `9696`
- direct-exported comparator rows at transfer `9732`

This strengthens the work-order story without turning it into a solved ECDLP speedup claim.

## Latest Frontier

- Coverage audit: `low_term_total2_frontier_gap_audit_6280_9735_probe.json`
- Bridge audit: `low_term_total2_direct_missing_column_bridge_audit_5984_9735_probe.json`
- Latest audited direct/rank/scout block: `9728_9735`
- Still-missing direct/rank hole carrying the shared-subcarrier work order: `9696_9703`

Bridge summary at `5984_9735`:

- Direct below-rho certificates: `947`
- Rank-gain certificates: `273`
- Accepted-missing-column rank gains: `242`
- Accepted-priority-column certificates: `795`

## Stable Clean Families

The clean accepted-missing families are unchanged:

| Family | Positives | Rank-gain total | Unique gain | Robust public tokens |
| --- | ---: | ---: | ---: | --- |
| `[11,15]` | 58 | 91 | 170 | `salt_min_mod4=2`, `selected_has=13` |
| `[10,14]` | 53 | 85 | 176 | `salt_min_mod4=2`, `selected_has=13` |
| `[0,5]` | 59 | 83 | 168 | `salt_min_mod4=2`, `selected_has=13` |

The fresh `9728_9735` rank gain landed at transfer `9732` as `[1,5]` with `RANK_GAIN_WITHOUT_ACCEPTED_MISSING_COLUMN`, so it is evidence that the public carrier can trigger rank gain outside the clean accepted-missing families.

## Work-Order Queue

Artifact:

- `low_term_total2_shared_subcarrier_workorder_9696_9735_probe.json`

Combined queue summary:

- Scout rows: `340`
- Candidate rows with any relevant token/family match: `307`
- Full shared-subcarrier rows: `6`
- Full shared-subcarrier direct-exported rows: `3`
- Full shared-subcarrier direct-missing rows: `3`
- Direct-missing transfer: `9696`
- Direct-exported comparator transfer: `9732`

Primary missing row:

- Transfer: `9696`
- Range: `9696_9703`
- Row keys: `salt170`, `salt171`
- Selected support: `[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`
- Compatible clean families: `[11,15]`, `[10,14]`, `[0,5]`
- Direct status: `direct_certificate_missing`

Fresh comparator row:

- Transfer: `9732`
- Range: `9728_9735`
- Row keys: `salt174`, `salt176`
- Selected support: `[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]`
- Compatible clean families: `[11,15]`, `[10,14]`, `[0,5]`
- Direct result: `RANK_GAIN_WITHOUT_ACCEPTED_MISSING_COLUMN`
- Accepted form: `[1,5]`
- Rank gain: `1`

## Interpretation

The shared subcarrier looks broader than the original `[10,14]` family. The next test should export `9696_9703` and ask whether transfer `9696` behaves like:

1. a clean accepted-missing family hit (`[11,15]`, `[10,14]`, or `[0,5]`),
2. a broader rank-gain-without-accepted-missing hit like transfer `9732`, or
3. a no-rank/saturated control.

