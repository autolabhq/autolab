# Next Hypothesis

Run a two-policy preregistered comparison on 104-111:

- `learned_global_zero_root_prior`, kept as the prior preregistered baseline;
- `learned_target_zero_residue16_prior`, because it is externally trained on
  <=87 rows and was 2/2 below rho on 96-103;
- optionally include static `low_root_norm` as a public baseline, but label it
  separately because its 96-103 success was observed post hoc.

The selection rule for a single promoted policy should not be changed using
104-111 after inspection. If a single policy must be preregistered before the
next run, choose `learned_target_zero_residue16_prior` and keep
`low_root_norm` as the control.

Required run order:

1. Write `ffe_learned_root_prior_preregistration_104_111.json`.
2. Generate fixed-selector total2 and total3/4 stress outputs.
3. Aggregate strict signatures.
4. Apply the pre-factor gate before Sage factorization.
5. Factor only gate-selected surfaces.
6. Audit first-fall root-hyperplane shape.
7. Score the preregistered learned policy with <=87 training rows only.
8. Report scan, direct-root, and full-remainder costs separately.

Promotion condition:

The candidate policy must preserve every selected root pair, have zero false
positives, and keep at least one honest accounting line below Pollard-rho on
every new gate-selected surface. If it fails, the algorithmic bottleneck is
root ordering after the pre-factor gate, not the FFE factorization itself.
