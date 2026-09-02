# ECDLP FFE Kernel Contract Handoff, 9959 Frontier

Date: 2026-06-05

## Result

The selected13 sharp lane has been lowered from a replay workorder into a strict kernel contract:

- `../low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9959_probe.json`

The contract status is `FFE_SHARP_LANE_KERNEL_CONTRACT_READY` with no failures.

It has also been lowered into typed ABI packets for a future native/FFE worker:

- `../low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9959_probe.json`
- `../low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9959_probe.h`

The packet manifest status is `FFE_SHARP_LANE_KERNEL_PACKET_MANIFEST_READY` with no failures.

## Verified Shape

- First-pass public groups: 18
- Second-pass row checks: 54
- Exact certificate checks: 6
- Inherited promotion checks: 12
- Direct/rank backfill groups: 12
- Exported direct rows: 18
- Missing direct rows: 36
- ABI packets: 18
- ABI row slots: 54
- Verifier target contexts: 1 verified target, `22050.cf1@11731`

Exact-positive transfers:

`9742, 9754, 9776, 9803, 9842, 9884`

Backfill transfers:

`9715, 9728, 9755, 9790, 9814, 9839, 9840, 9872, 9889, 9909, 9913, 9943`

The new tail addition in the sharp lane is transfer `9943`, represented as a full-family missing row `t9943_4c88ff834538` with public salts `167,175`.

Packet `17` carries this transfer with:

- first-pass id: `first_cff38c0711c53228`
- packet hash: `packet_a3c18ea956996b89`
- row offset: `51`
- target field: `GF(11731)`
- base order: `11779`
- generic rho steps: `137`
- full-family row hash: `row_05f942a16fb73283`

## Important Context

The current feature-lift promoted rule over the 9696-9959 validation window is broad:

`public_product_gate=False AND selected_has=13`

The fixed sharp lane remains useful because it preserves a repeatable exact-positive stream, but it is no longer the feature miner's top promoted rule for this window. The new assembly script patch avoids treating the name `sharp_promoted` as authoritative and instead derives lane targets by token matching.

## Non-Claims

This handoff does not claim a solved ECDLP instance, a completed index-calculus break, a summation-polynomial evaluation, or a Pollard-rho speedup. It is a reproducible contract for the next FFE/summation-polynomial worker to satisfy.
