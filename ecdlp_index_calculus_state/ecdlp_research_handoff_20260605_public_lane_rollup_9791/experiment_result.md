# ECDLP Index-Calculus Handoff: Public Carrier Lanes at 9791

Date: 2026-06-05

## Claim

The public-carrier policy is now stable enough to steer AutoLab:

- Primary export lane: `selected_has=13`
- Diagnostic strict sublane: `selected_has=13 AND salt_min_mod4=2`

The broad lane remains the better export policy by observed full-family rank-gain evidence. The strict lane is smaller and cleaner, but it mainly diagnoses `[1,5]` rank-gain-without-accepted-missing behavior.

This is still relation-export evidence only, not a full ECDLP speedup proof.

## Frontier

- Coverage audit: `low_term_total2_frontier_gap_audit_6280_9791_probe.json`
- Bridge audit: `low_term_total2_direct_missing_column_bridge_audit_5984_9791_probe.json`
- Latest complete direct/rank/scout block: `9784_9791`
- Direct/rank-missing scout block still present: `9768_9775`
- Persistent direct/rank file hole: `9696_9703`

Bridge summary at `5984_9791`:

- Direct below-rho certificates: `962`
- Rank-gain certificates: `278`
- Accepted-missing-column certificates: `609`
- Accepted-missing-column rank gains: `246`

## Family Movement

Compared with `9775`, accepted-missing rank gain did not move. The fresh direct tail added one rank-gain-without-accepted-missing row:

- `[1,5]` gained transfer `9776`
- `[10,14]` remains at `55` positives, rank-gain total `88`, unique gain `183`
- `[0,5]` remains at `61` positives, rank-gain total `86`, unique gain `175`
- `[11,15]` remains at `58` positives, rank-gain total `91`, unique gain `170`

## Lane Rollup

Artifact:

- `low_term_total2_public_lane_comparison_rollup_9696_9791_probe.json`

Broad lane (`selected_has=13`):

- Shared rows: `60`
- Exported rank-gain rows: `12`
- Exported rank-gain rate: `0.5`
- Full-family exported rows: `12`
- Full-family exported rank-gain rows: `6`
- Full-family exported rank-gain rate: `0.5`
- Exported rank-gain transfers: `9705`, `9732`, `9742`, `9750`, `9754`, `9776`
- Missing full-family transfers: `9696`, `9699`, `9700`, `9701`, `9707`, `9713`, `9715`, `9719`, `9728`, `9729`, `9739`, `9743`, `9755`, `9761`, `9763`, `9767`, `9771`, `9790`

Strict lane (`selected_has=13 AND salt_min_mod4=2`):

- Shared rows: `8`
- Exported rank-gain rows: `2`
- Exported rank-gain rate: `1.0`
- Full-family exported rows: `1`
- Full-family exported rank-gain rows: `1`
- Full-family exported rank-gain rate: `1.0`
- Exported rank-gain transfer: `9732`
- Missing full-family transfers: `9696`, `9739`, `9763`

## Interpretation

The broad `selected_has=13` lane should be promoted as the primary export queue. It has repeated rank-gain outcomes across accepted forms `[10,13]`, `[10,14]`, `[0,5]`, and `[1,5]`.

The strict lane should remain attached as a sublabel on the broad queue. It is too sparse for the main export policy, but it has a perfect exported-rank-gain rate so far and should be tracked as a possible `[1,5]` / rank-gain-without-accepted-missing steering feature.

## Next Work Order

Primary broad-lane full-family export queue:

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
11. `9739`
12. `9743`
13. `9755`
14. `9761`
15. `9763`
16. `9767`
17. `9771`
18. `9790`

Strict diagnostic subset:

1. `9696`
2. `9739`
3. `9763`

