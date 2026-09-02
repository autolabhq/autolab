# Next Hypothesis

The selected13 sharp lane is now tight enough to use as the first concrete
FFE/summation-polynomial relation assembly stream.

## Hypothesis

If a fused FFE/summation-polynomial executor replays the public row-key salt
pairs in the `selected13_sharp_minmod3` lane, then the executor should reproduce
the 6 exact rank-gain transfers and should prioritize direct/rank export on the
8 full-family missing transfers:

```text
9715, 9728, 9755, 9790, 9814, 9839, 9840, 9872
```

The success criterion for the next stage is relation assembly efficiency only:
same public inputs, same direct/rank outputs, less wasted direct-rank work.  It
is not yet an ECDLP solve criterion.

## Work Order

1. Read `ecdlp_index_calculus_state/low_term_total2_ffe_assembly_manifest_selected13_9696_9887_probe.json`.
2. Select lane `selected13_sharp_minmod3`.
3. Replay the exact positives first:

```text
9742, 9754, 9776, 9803, 9842, 9884
```

4. Use the backfill queue for direct/rank export on the full-family missing
   transfers:

```text
9715, 9728, 9755, 9790, 9814, 9839, 9840, 9872
```

5. Promote only rows whose FFE replay produces exact row-level direct/rank
   certificates.  Keep transfer-level inherited evidence separate from exact
   row evidence.

## Comparator Baseline

The current comparator baseline for selected13 through 9887 is:

- Exported rank-gain transfers: 9705, 9732, 9742, 9750, 9754, 9776, 9803, 9820, 9828, 9842, 9849, 9860, 9880, 9884
- Missing full-family transfers: 9696, 9699, 9700, 9701, 9707, 9713, 9715, 9719, 9728, 9729, 9739, 9743, 9755, 9761, 9763, 9767, 9771, 9790, 9795, 9799, 9806, 9814, 9821, 9822, 9825, 9827, 9833, 9837, 9839, 9840, 9847, 9858, 9864, 9867, 9870, 9872, 9873, 9875, 9877
- Sharp promoted rule: `selected_has=13`, `salt_adjacent=False`, `salt_min_mod4=3`

## Stop Conditions

- Stop claiming a lane improvement if exact FFE replay fails on any of the 6
  sharp positive transfers.
- Stop promoting inherited positives if exact bridge certificates cannot be
  recovered for the same row-key, selector, top-k, and selected support.
- Stop broadening the lane if backfill targets grow faster than exact positive
  transfers.
