# Hypothesis

The 144-151 fresh test preregistered the total2-only policy
`prefactor_unique_leaf_hit_root_first_target_hash`.

The policy uses only pre-factor gate-manifest fields to locate a Sage
root-hyperplane factor:

- read `pre_factor_selected_hit_roots` and
  `pre_factor_selected_hit_root_pairs`;
- rank candidate roots in that set first;
- prefer roots coming from selected leaves with fewer pre-factor hit roots;
- fall back to `target_root_hash`;
- stop at the first public-zero root hyperplane.

Promotion condition:

- direct-root positive if all total2 gate-selected surfaces are preserving,
  false-positive-free, and direct-below-rho;
- scan-positive only if all are also scan-below-rho.

The test remains total2-scoped. Total3/4 was excluded because the same
pre-factor policy family already false-positives on the known 120-127
total3/4 negative-control surface.
