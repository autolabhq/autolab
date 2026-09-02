# Hypothesis

The 120-127 total2-only root-hyperplane signal failed under generic public
hash ordering because the preserving root could appear late in the factor
order. The next hypothesis was that the pre-factor gate already exposes a
public root locator:

- materialize a total2 FFE surface from strict low-term total2 certificates;
- before Sage factorization, record selected public leaf roots in
  `pre_factor_selected_hit_roots`;
- after factorization, rank root hyperplanes whose root is in that pre-factor
  set before public hash fallbacks;
- require preserving roots, zero false positives, and below-rho costs against
  the same Pollard-rho baseline.

This is still a component hypothesis. It does not claim end-to-end ECDLP
recovery faster than Pollard rho, and it does not promote total3/4 surfaces.

The 136-143 preregistered fresh variant added a tie-break:

`prefactor_unique_leaf_hit_root_first_target_hash`

This chooses roots that come from selected leaves with fewer pre-factor hit
roots before falling back to `target_root_hash`.
