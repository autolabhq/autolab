# Next Hypothesis

The sharp selected13 lane is now stable enough to make the next FFE step an
exact replay gate instead of another lane miner.

## Hypothesis

A lower-level FFE/summation-polynomial replay should reproduce the 6 exact
positive public first-pass groups:

```text
9742, 9754, 9776, 9803, 9842, 9884
```

After those pass, the same first-pass grouping should be used to backfill the
11 missing full-family transfers:

```text
9715, 9728, 9755, 9790, 9814, 9839, 9840, 9872, 9889, 9909, 9913
```

Success at this stage means exact row-level replay of bridge certificates and
clean direct/rank export on missing rows.  It does not mean ECDLP recovery.

## Execution Order

1. Consume `low_term_total2_ffe_sharp_lane_replay_workorder_selected13_9696_9935_probe.json`.
2. Replay `queues.exact_positive_replay_queue` first.
3. For each exact-positive group, reproduce the exact bridge certificate fields:
   row keys, selector, top-k, selected support, accepted missing columns, form
   supports, rank gain, and unique factor-relation gain.
4. Only after exact replay passes, attempt `queues.inherited_promotion_queue`.
5. Export direct/rank rows for `queues.direct_rank_backfill_queue`, starting
   with the 0.6350365-cost transfers 9728, 9755, 9790, 9872, 9909, and 9913.

## Stop Conditions

- Stop if any exact-positive group fails to reproduce its exact bridge
  certificate.
- Stop promoting inherited-positive rows if they cannot be tied to the same
  row-key, selector, top-k, and selected support.
- Stop claiming sharp-lane improvement if the 11 backfill transfers do not
  produce new exact direct/rank evidence.
