# Next Hypothesis

The next useful step is to separate the direct-root win from the scan-only
weakness and then test the public `target_root_hash` signal on a fresh window.

Concrete proposal for 120-127:

- Preregister `target_root_hash` as the primary static public root policy.
- Keep `residue16_then_global_hash_cap1` as the learned/public hybrid baseline.
- Keep `summax_sage_low_constant_target_hash` as the frozen <=87 static
  baseline.
- Continue the same strict signature and pre-factor gate.

Reason:

- `target_root_hash` was post hoc best on 112-119: 3/3 preserving, 0 false
  positives, 3/3 scan-below-rho, and max scan cost 0.912.
- It is a public static policy, so a fresh preregistration can avoid learned
  train-set leakage concerns entirely.
- The cap1 hybrid is promising under direct-root accounting, but still has one
  scan-only miss at 1.016 ops/rho.

Parallel implementation work:

- Add a stricter evaluator summary for "all selected surfaces below rho by
  scan", "all selected surfaces below rho by direct-root", and "all selected
  surfaces below rho by either route" so future handoffs do not rely on manual
  comparison of two counts.
- Keep rejecting full-remainder claims until a measured surface has
  full-remainder FFE below rho.

The algorithmic target is now sharper: public pre-factor gate plus public or
hybrid root-hyperplane ordering is beating rho on measured fresh FFE surfaces;
the remaining proof obligations are scan-uniformity and a fresh hit-stream
generator rather than more posthoc surface selection.
