# Next Hypothesis

## Hypothesis

If the FFE/summation-polynomial worker implements the first-pass public group contract and then reproduces the six exact row-level certificates, the same executor can be pointed at the 12 direct/rank backfill groups to convert at least one missing full-family row into a new exact certificate.

The highest-value immediate backfill target is transfer `9943` because it is newly exposed at the 9959 frontier and already has a full-family row in the strict contract:

- transfer: `9943`
- first-pass id: `first_cff38c0711c53228`
- group id: `sharp_group_a15d48c024d5`
- full-family row id: `t9943_4c88ff834538`
- salts: `167,175`
- row keys: `22050.cf1@11731:uniform:256:salt167`, `22050.cf1@11731:uniform:256:salt175`

## Required Worker Gate

The worker should emit the same fields and hashes as:

- `contract.first_pass_contract`
- `contract.second_pass_contract`
- `contract.exact_certificate_checks`
- `contract.inherited_promotion_checks`
- `contract.direct_rank_backfill_manifest`

Before implementing the arithmetic kernel, it should ingest the typed packet ABI:

- `../low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9959_probe.json`
- `../low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9959_probe.h`

The packet manifest verifies the target context for `22050.cf1@11731` and gives each row a `selected_support_mask`, family masks, promotion flags, and direct/rank export flags.

Promotion rule:

Inherited rows remain blocked until the worker produces exact row-level certificates. Same-salt bridge evidence is useful for prioritization only.

## Next Command

```sh
python3 tasks/ecdlp_index_calculus/low_term_total2_ffe_sharp_lane_kernel_contract.py \
  --workorder ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_replay_workorder_selected13_9696_9959_probe.json \
  --out ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9959_probe.json
```

```sh
python3 tasks/ecdlp_index_calculus/low_term_total2_ffe_sharp_lane_kernel_packet_manifest.py \
  --contract ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_kernel_contract_selected13_9696_9959_probe.json \
  --out ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9959_probe.json \
  --c-header-out ecdlp_index_calculus_state/low_term_total2_ffe_sharp_lane_kernel_packet_manifest_selected13_9696_9959_probe.h
```

The future FFE implementation should consume the packet manifest/header, not the broader validation manifest.
