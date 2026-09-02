# Hypothesis

The 168-175 fresh test repeats the preregistered total2-only
single-hit-root gate for a third disjoint window.

Preregistered route:

- Channel: low-term total2 only.
- Gate: `single_prefactor_hit_root`.
- Gate rule: keep a public-zero surface only when the pre-factor gate manifest
  exposes exactly one distinct selected hit root before Sage factorization.
- Primary policy: `prefactor_unique_leaf_hit_root_first_target_hash`.
- Promotion target: scan-positive if every retained surface is preserving,
  false-positive-free, and below generic rho under conservative root scan.
- Direct-root positive is a weaker fallback if every retained surface is
  preserving, false-positive-free, and direct below rho.

The algorithmic hypothesis is now sharper: total2 first-fall resultant
factorization exposes public quadratic root hyperplanes `c + r*b + r^2`, and
when the pre-factor selected leaves expose exactly one hit root, that root
selects a preserving hyperplane without post-factor labels.

The claim remains component-level. It is not an end-to-end ECDLP speedup and
does not promote total3/4.
