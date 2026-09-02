# Hypothesis

Declare `target_root_hash` before the 120-127 window as the primary public
static root-hyperplane orderer.

The policy was promoted only from prior-window evidence: on 112-119 it was
diagnostic, not a claim, but it scored 3/3 preserving, zero false positives,
and 3/3 scan-below-rho on gate-selected surfaces.

Promotion condition:

- Use the same pre-factor gate as prior windows.
- Factor only gate-selected surfaces.
- Score `target_root_hash` across all gate-selected total2 and total3/4
  surfaces.
- Count as a scan-positive promotion only if every selected surface is
  preserving, false-positive-free, and below Pollard rho by scan accounting.
- Count as a weaker direct-root positive only if every selected surface is
  preserving, false-positive-free, and below rho by direct-root accounting.

Declared controls:

- `residue16_then_global_hash_cap1`
- `summax_sage_low_constant_target_hash`
- `global_root_hash`
- `learned_target_zero_residue16_prior`
