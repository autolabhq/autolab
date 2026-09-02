# Next Hypothesis

The next pressure point is to remove the two remaining kinds of caveat:

1. Freshness: run the fixed public root-hyperplane order on a new, held-out hit
   stream.  Do not choose a policy per transfer after seeing outcomes.
2. Root recovery cost: the first direct-root companion is now in the selector
   probe.  Stress it on fresh streams and keep reporting conservative scan and
   direct-root costs side by side.
3. Prior selection: the held-out zero-root prior has lower mean cost but two
   over-rho tails.  Train a proper policy selector on older transfers and
   freeze it before testing later transfers.

Suggested next run:

```text
Apply the fixed best public policy and a frozen learned zero-root prior to a fresh 88-95 surface bank, using ffe_first_fall_root_hyperplane_selector_probe.py with both scan and direct-root cost summaries.
```

Success criterion:

- fixed policy selected before testing
- no same-transfer or same-row leakage for learned priors
- public-zero recovered and preserving on all fresh surfaces
- charged direct-root path below rho on all fresh surfaces
- conservative root-scan path reported honestly even if it has tails
