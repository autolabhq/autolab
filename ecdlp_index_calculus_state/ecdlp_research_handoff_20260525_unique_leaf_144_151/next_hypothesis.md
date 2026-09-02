# Next Hypothesis

The next fresh window should preregister the stricter single-hit-root gate.

Concrete proposal for 152-159:

- Channel: total2 only.
- Gate: `single_prefactor_hit_root`.
- Primary policy: `prefactor_unique_leaf_hit_root_first_target_hash`.
- Promotion target: scan-positive if all selected surfaces are preserving,
  false-positive-free, and scan-below-rho.
- Direct-root positive is a weaker fallback if scan has exactly one miss.

Reason:

- The all-prefactor 144-151 policy failed because the salt174 transfer-150
  surface had two pre-factor hit roots and no preserving factor.
- Across total2 112-151, the single-hit-root gate keeps 13 surfaces and is
  13/13 preserving, false-positive-free, scan-below-rho, and direct-below-rho.
- The dropped 136-143 multi-hit preserving surface was already the lone
  scan-over-rho case, so the stricter gate improves precision and restores the
  scan-positive boundary.

Controls:

- all-prefactor-gate `prefactor_unique_leaf_hit_root_first_target_hash`;
- `prefactor_selected_hit_root_first_target_hash`;
- `target_root_hash`;
- `global_root_hash`;
- `low_root_norm`;
- `summax_sage_low_constant_target_hash`.

Open obligations:

- The single-hit-root gate must survive a fresh window before promotion.
- Multi-hit-root surfaces need a separate discriminator instead of being mixed
  into the promoted route.
- Total3/4 remains excluded until a public channel discriminator rejects the
  known non-preserving failure mode.
- This is still a component route, not an end-to-end ECDLP speedup.
