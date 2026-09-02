# Next Hypothesis

The next fresh window should test a two-stage root-ordering policy that keeps
the false-positive discipline of `learned_target_zero_residue16_prior` but adds
a public/static fallback before root rank becomes expensive.

Concrete proposal for 112-119:

- Preregister `summax_sage_low_constant_target_hash` as the frozen static
  baseline again, because it is selected entirely from <=87 static public rows
  and scored 2/2 below rho on 104-111.
- Preregister `global_root_hash` only as a public control, not as the learned
  claim, because it was post hoc best on 104-111.
- Add a new learned/public hybrid candidate:
  `residue16_then_global_hash_cap4`.
- Policy sketch:
  - order roots first by `learned_target_zero_residue16_prior`;
  - if no preserving/public-zero hit occurs in the first four evaluated roots,
    switch to `global_root_hash` order for the remaining candidates;
  - charge all selector and root-scan work.

Why this is the right next cut:

- On 104-111 the residue16 prior chose the correct total2 factor first.
- On 104-111 the residue16 prior also chose the correct total3/4 factor, but
  only at rank 11.
- The global hash static policy found the total3/4 preserving factor at rank 4
  and remained below rho.
- A cap-and-fallback policy is simple enough to preregister and falsify.

Continue to reject any promotion that depends on full-remainder costs; both
104-111 full remainders remained above rho. The useful current mechanism is
pre-factor gating plus first-fall root-hyperplane ordering.
