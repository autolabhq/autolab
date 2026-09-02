# Hypothesis

The preregistered pre-factor FFE gate is strong enough to support a stricter
root-policy test: choose one root-hyperplane ordering policy using only
pre-88 gated surfaces, then score the same policy on the fresh 88-95 gated
surfaces.

The leakage boundary is:

- training rows may use only 72-79 and 80-87 selector outcomes after the
  pre-factor gate manifest has already selected surfaces;
- holdout rows may use only the 88-95 pre-factor gate manifest for inclusion;
- the selected root policy must be fixed before looking at 88-95 policy
  performance;
- post hoc best policies on 88-95 are diagnostics, not claims.

This handoff also records a sensitivity probe for learned root priors trained
externally from the <=87 gated rows. Those learned-prior results are candidate
next preregistrations, not proof for the already-inspected 88-95 window.
