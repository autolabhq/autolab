# Next Hypothesis

The next fresh window should preregister the unique-leaf pre-factor root
locator, not the original selected-hit-root variant.

Concrete proposal for 144-151:

- Channel: total2 only.
- Primary policy: `prefactor_unique_leaf_hit_root_first_target_hash`.
- Promotion target: direct-root positive first; scan-positive only if every
  selected surface also stays below rho under root-scan charging.
- Controls:
  - `prefactor_selected_hit_root_first_target_hash`
  - `target_root_hash`
  - `low_root_norm`
  - `summax_sage_low_constant_target_hash`

Reason:

- On fresh 136-143, the original selected-hit-root policy had one false
  positive caused by a multi-root selected leaf.
- The unique-leaf tie-break used only pre-factor root multiplicity and fixed
  that false positive.
- It produced 6/6 preserving and 6/6 direct-below-rho on fresh 136-143.
- It is 12/12 preserving and 12/12 direct-below-rho across total2 112-143.

Open proof obligations:

- The scan route still has one 136-143 surface at 1.064 ops/rho, so scan-positive
  promotion needs cheaper root-scan accounting or a sharper public subscan.
- Total3/4 remains unsafe: the 120-127 negative-control surface still produces
  a false positive under the same pre-factor family.
- This is still a component route. The next stronger milestone is to connect
  the direct-root component to relation-derived ECDLP recovery rather than only
  preserving FFE root hyperplanes below rho.

Immediate engineering follow-up:

- Consider promoting `prefactor_unique_leaf_hit_root_first_target_hash` into the
  shared root-selector/evaluator policy list only after one more fresh total2
  window.
- Add a total3/4 public discriminator before any all-channel retest.
- Keep full-remainder claims rejected; all full remainders in 128-143 were
  above rho.
