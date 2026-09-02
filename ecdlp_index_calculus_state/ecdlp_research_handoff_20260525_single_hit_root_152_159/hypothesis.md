# Hypothesis

The 152-159 fresh test preregistered the stricter total2-only
single-hit-root gate before seeing the new window.

Preregistered route:

- Channel: low-term total2 only.
- Gate: `single_prefactor_hit_root`.
- Gate rule: keep a public-zero surface only when the pre-factor gate manifest
  shows exactly one distinct selected hit root.
- Primary policy: `prefactor_unique_leaf_hit_root_first_target_hash`.
- Promotion target: scan-positive if every retained surface is preserving,
  false-positive-free, and below generic rho under conservative root scan.
- Direct-root positive is a weaker fallback if every retained surface is
  preserving, false-positive-free, and direct below rho.

The hypothesis is that the first-fall quotient route is not a generic
high-degree resultant search. On these measured surfaces, Sage factorization
breaks the resultant into quadratic root hyperplanes of the form
`c + r*b + r^2`, and the useful public component is a root-selection problem.
The single-hit-root gate asks whether pre-factor public evidence can identify
the only safe root before factorization.

The claim remains component-level. It is not an end-to-end ECDLP speedup and
does not promote total3/4.
