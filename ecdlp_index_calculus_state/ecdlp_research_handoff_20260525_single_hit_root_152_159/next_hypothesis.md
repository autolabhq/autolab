# Next Hypothesis

Preregister the same single-hit-root gate on the next fresh total2 window.

Concrete proposal for 160-167:

- Channel: total2 only.
- Gate: `single_prefactor_hit_root`.
- Primary policy: `prefactor_unique_leaf_hit_root_first_target_hash`.
- Promotion target: scan-positive if all selected surfaces are preserving,
  false-positive-free, and scan-below-rho.
- Direct-root positive remains a weaker fallback.

Controls:

- Broad all-prefactor-gate `prefactor_unique_leaf_hit_root_first_target_hash`.
- Broad all-prefactor-gate `prefactor_selected_hit_root_first_target_hash`.
- Root selector controls: `target_root_hash`, `global_root_hash`,
  `low_root_norm`, and `summax_sage_low_constant_target_hash`.
- Explicit multi-hit-root audit showing whether dropped surfaces have any
  preserving Sage factor.

Parallel research thread:

- Build a separate multi-hit discriminator instead of weakening the strict
  gate.
- For each multi-hit surface, rank roots using only public pre-factor fields:
  repeated-root count across selected leaves, root frequency across nearby row
  salts, transfer-index stability, and target-local root recurrence.
- Treat a multi-hit route as a new hypothesis only after it rejects the known
  bad `22050` transfer-150 and transfer-159 salt174 surfaces.

Open obligations:

- This is still a component route, not an end-to-end ECDLP speedup.
- The relation-harvesting bridge must show how retained first-fall root
  hyperplanes assemble into enough independent relations to beat rho globally.
- Total3/4 stays excluded until a public channel discriminator rejects the
  known non-preserving total3/4 failure mode.
