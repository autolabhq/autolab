# Hypothesis

Declare `residue16_then_global_hash_cap1` before the 112-119 window and score
it only after the strict signature, pre-factor gate, Sage factorization, and
first-fall root-hyperplane audit.

The policy is a capped learned/public hybrid:

- train `learned_target_zero_residue16_prior` only on <=87 gated surfaces;
- evaluate the first residue16-ranked candidate;
- if that candidate is not public-zero, continue with remaining candidates in
  `global_root_hash` order;
- stop at the first public-zero root hyperplane;
- charge every evaluated selector step plus root-scan or direct-root recovery.

Promotion condition:

- The pre-factor gate must use only materialized-surface nonvacuity and
  selected-leaf hit-root proxy fields.
- Sage factorization must be run only on pre-factor gate-selected surfaces.
- The chosen root policy must preserve all original selected root pairs.
- The chosen root policy must have zero false positives.
- Every gate-selected surface must beat Pollard rho under scan or direct-root
  accounting.

Declared controls:

- `summax_sage_low_constant_target_hash`
- `global_root_hash`
- `learned_target_zero_residue16_prior`
