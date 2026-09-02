# Hypothesis

The 160-167 fresh test repeats the preregistered total2-only
single-hit-root gate after the 152-159 promotion candidate.

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

The claimed component is a public first-fall root-selection route, not a
generic high-degree resultant search. The structural premise is that Sage
factorization exposes resultant factors of the form `c + r*b + r^2`, so the
public pre-factor hit root can select the correct quadratic root hyperplane.

The claim remains component-level. It is not an end-to-end ECDLP speedup and
does not promote total3/4.
