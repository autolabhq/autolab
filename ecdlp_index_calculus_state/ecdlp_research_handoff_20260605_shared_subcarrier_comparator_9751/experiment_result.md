# ECDLP Index-Calculus Handoff: Shared-Carrier Comparator at 9751

Date: 2026-06-05

## Claim

The live low-term total2 frontier now separates into two useful public lanes:

1. Strict shared subcarrier: `salt_min_mod4=2 AND selected_has=13`
2. Broader full-support carrier: `selected_has=13`

The strict lane has exported rank-gain comparators and two direct-missing full-family targets. The broader lane produced fresh accepted-missing `[10,14]` and `[0,5]` rank gains at `9742` and `9750`.

This is still a relation-export work-order claim, not an end-to-end ECDLP speedup claim.

## Latest Frontier

- Coverage audit: `low_term_total2_frontier_gap_audit_6280_9751_probe.json`
- Bridge audit: `low_term_total2_direct_missing_column_bridge_audit_5984_9751_probe.json`
- Latest audited complete tail block: `9744_9751`
- Persistent missing block carrying the original full shared-subcarrier target: `9696_9703`

Bridge summary at `5984_9751`:

- Direct below-rho certificates: `955`
- Rank-gain certificates: `276`
- Accepted-missing-column rank gains: `245`
- Accepted-priority-column certificates: `803`

## Accepted-Form Movement

The new complete blocks added meaningful accepted-missing rank gain:

- `[10,14]`: now `55` positives, rank-gain total `88`, unique gain `183`; fresh transfers include `9742`, `9750`
- `[0,5]`: now `60` positives, rank-gain total `85`, unique gain `174`; fresh transfer `9750`
- `[11,15]`: unchanged at `58` positives, rank-gain total `91`, unique gain `170`

The robust zero-control token pair remains `salt_min_mod4=2 AND selected_has=13` for `[10,14]`, `[11,15]`, `[0,5]`, and priority recurrence.

## Comparator Audit

Artifact:

- `low_term_total2_shared_subcarrier_comparator_audit_9696_9751_probe.json`

Summary:

- Strict shared-subcarrier rows: `6`
- Direct-exported strict shared rows: `2`
- Direct-missing strict shared rows: `4`
- Exported rank-gain transfer: `9732`
- Missing full-family transfers: `9696`, `9739`

Full-family strict lane:

| Transfer | Status | Row salts | Selected support | Outcome |
| --- | --- | --- | --- | --- |
| `9696` | direct missing | `170,171` | full support minus column `1` | pending export |
| `9739` | direct missing | `166,172` | full support minus column `1` | pending export |
| `9732` | direct exported | `174,176` | full support minus column `1` | `[1,5]`, rank gain `1` |

The comparator at `9732` proves the strict public carrier can trigger rank gain, but it landed as `RANK_GAIN_WITHOUT_ACCEPTED_MISSING_COLUMN` through `[1,5]`, not as one of the clean accepted-missing families.

## Broader Selected-13 Lane

The fresh accepted-missing hits at `9742` and `9750` are full-support and `selected_has=13`, but they do not satisfy `salt_min_mod4=2`:

- `9742`: salts `171,173`, accepted form `[10,14]`, rank gain `1`
- `9750`: salts `161,170`, accepted forms `[0,5]`, `[1,5]`, `[10,14]`, rank gain `2`, unique gain `6`

This suggests `selected_has=13` is a broad public carrier, while the `salt_min_mod4=2` refinement steers toward a narrower rank-gain-without-accepted-missing lane.

## Next Work Order

Export direct/rank rows for the missing strict full-family targets:

1. Transfer `9696`, salts `170,171`
2. Transfer `9739`, salts `166,172`

Then compare their outcomes against:

- strict comparator `9732` (`[1,5]`, rank gain without accepted missing)
- broad selected-13 accepted-missing hits `9742` and `9750`

