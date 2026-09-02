# Next Hypothesis

Preregister the mean-cost learned root prior before touching the next unseen
window.

Candidate rule:

- train rows: all preregistered pre-factor-gated surfaces through 87;
- eligible policies: static public policies plus learned zero-root priors;
- selection rule: among policies with all calibration rows preserving and zero
  false positives, choose the minimum mean scan ops/rho;
- current selected policy under that rule: `learned_global_zero_root_prior`;
- apply the selected policy unchanged to 96-111 or to unseen target labels.

Required next run order:

1. Generate fixed row/leaf signatures for the unseen window.
2. Write strict signature candidates.
3. Write the pre-factor gate manifest before Sage factorization.
4. Factor only gate-selected surfaces.
5. Build first-fall/root-hyperplane audit rows.
6. Apply the preregistered `learned_global_zero_root_prior` policy trained only
   on <=87 rows.
7. Report scan, direct-root, and full-remainder costs separately.

Promotion condition:

The branch only becomes a stronger algorithm candidate if the preregistered
learned policy preserves every selected root pair, has zero false positives,
and keeps scan or direct-root accounting below Pollard-rho on the new unseen
surfaces. If it fails, keep the pre-factor gate but treat root ordering as the
remaining blocker.
