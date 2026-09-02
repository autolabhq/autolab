# ECDLP Index-Calculus Handoff: Public Carrier Lanes at 9775

Date: 2026-06-05

## Claim

The low-term total2 public carrier has separated into two actionable lanes:

1. Broad lane: `selected_has=13`
2. Strict lane: `selected_has=13 AND salt_min_mod4=2`

The broad lane is currently the better export policy by observed full-family rank-gain evidence. The strict lane is a smaller high-precision diagnostic lane with one exported rank-gain comparator and three full-family direct-missing targets.

This remains relation-export evidence, not an end-to-end ECDLP speedup proof.

## Frontier

- Coverage audit: `low_term_total2_frontier_gap_audit_6280_9775_probe.json`
- Bridge audit: `low_term_total2_direct_missing_column_bridge_audit_5984_9775_probe.json`
- Latest common direct/rank/scout block: `9760_9767`
- Latest scout-only block included in support reports: `9768_9775`
- Persistent direct/rank file hole: `9696_9703`

Bridge summary at `5984_9775`:

- Direct below-rho certificates: `959`
- Rank-gain certificates: `277`
- Accepted-missing-column rank gains: `246`
- Accepted-priority-column certificates: `807`

## Accepted-Form Movement

From `low_term_total2_accepted_form_contrast_miner_5984_9775_probe.json`:

| Family | Positives | Rank-gain total | Unique gain | Fresh tail |
| --- | ---: | ---: | ---: | --- |
| `[11,15]` | 58 | 91 | 170 | unchanged |
| `[10,14]` | 55 | 88 | 183 | `9742`, `9750` |
| `[0,5]` | 61 | 86 | 175 | `9750`, `9754` |

The robust zero-control subcarrier remains `salt_min_mod4=2 AND selected_has=13` for `[10,14]`, `[11,15]`, `[0,5]`, and priority recurrence.

## Lane Comparison

Artifact:

- `low_term_total2_public_lane_comparison_rollup_9696_9775_probe.json`

Broad lane (`selected_has=13`):

- Shared rows: `54`
- Exported rows: `20`
- Exported rank-gain rows: `10`
- Exported rank-gain rate: `0.5`
- Full-family exported rows: `10`
- Full-family exported rank-gain rows: `5`
- Full-family exported rank-gain rate: `0.5`
- Rank-gain transfers: `9705`, `9732`, `9742`, `9750`, `9754`
- Missing full-family transfers: `9696`, `9699`, `9700`, `9701`, `9707`, `9713`, `9715`, `9719`, `9728`, `9729`, `9739`, `9743`, `9755`, `9761`, `9763`, `9767`, `9771`

Strict lane (`selected_has=13 AND salt_min_mod4=2`):

- Shared rows: `8`
- Exported rows: `2`
- Exported rank-gain rows: `2`
- Exported rank-gain rate: `1.0`
- Full-family exported rows: `1`
- Full-family exported rank-gain rows: `1`
- Full-family exported rank-gain rate: `1.0`
- Rank-gain transfer: `9732`
- Missing full-family transfers: `9696`, `9739`, `9763`

## Interpretation

The broad lane is now the better primary export policy: it has five exported rank-gain transfers and covers accepted-missing forms `[10,13]`, `[10,14]`, `[0,5]`, plus `[1,5]` rank gain without accepted missing.

The strict lane should remain a diagnostic subqueue. Its exported comparator at transfer `9732` produced `[1,5]` rank gain without accepted-missing-column gain, while its missing full-family targets are `9696`, `9739`, and `9763`.

## Next Export Queue

Primary broad-lane queue:

1. `9696`
2. `9699`
3. `9700`
4. `9701`
5. `9707`
6. `9713`
7. `9715`
8. `9719`
9. `9728`
10. `9729`

Strict diagnostic subqueue:

1. `9696`
2. `9739`
3. `9763`

Priority should go to rows that are both broad-lane and strict-lane when available, then to full-family broad-lane rows with `selected_has=13`.

