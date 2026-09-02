# Next Hypothesis

The `salt165 / transfer mod 6 = 0` selector may still be a valid full-remainder activation cell, but the current replay path cannot test it because the selected row profile is not recoverable through `witness_specs(...)`.

## Next Concrete Experiment

Build a forced-witness exact-profile materializer that consumes the `row_leaf_keys` entry already present in `low_term_total3_total4_public_bounded_full_selector_*_*.json` and reconstructs the row context without requiring the row to appear in the default witness spec index.

Target the first two failed exposures:

```text
376_383 transfer 378, salt165, leaves 79 and 65,79
416_423 transfer 420, salt165, leaves 90 and 8,90
```

Success criteria:

- materializes at least one Sage surface for transfer 378 or 420
- records `surface_id`, selected leaves, resultant stats, full remainder stats, and Sage factor candidates
- distinguishes "no preserving factor" from "full remainder over rho" from "full remainder below rho"
- keeps Pollard-rho charging explicit via `remainder_ffe_ops_over_rho`

## Why This Is Worth Doing

The coverage audit found 96 frozen-selector profiles across 12 later transfer indices. That is enough future exposure to test the original sparsity rule, if the materializer can build the surfaces.

The control space is also good: 1720 same-target neighbor profiles at the same transfer residue are available, spanning 16 neighboring salts. Once materialization works, controls can be run without changing the selector.

## Claim Boundary

The current evidence does not show a new speedup. It shows:

- a public frozen selector with future exposure
- two failed exact-profile attempts caused by missing witness specs
- a concrete implementation gap between public selector artifacts and Sage factorization

Do not tune the selector on later windows until the forced-witness materializer can replay transfer 378 and 420 exactly.
