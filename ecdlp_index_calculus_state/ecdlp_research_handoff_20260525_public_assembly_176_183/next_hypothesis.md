# Next Hypothesis

The next public assembly rule should keep companion rows, not only retained
root surfaces.

Recommended preregistered rule for 184-191:

1. Use the same public below-rho total2 candidate signature.
2. Select cases with at least one single-hit-root public-zero anchor below rho.
3. Keep the anchor surface plus its full public leaf companion row set from the
   candidate case.
4. Require public source ops/rho below 1.0 for the complete companion assembly.
5. Rank by anchor root cost, then source ops/rho, then deterministic case key.
6. Replay the complete selected assembly and report same-challenge derivation.

Success criterion:

Promote only if complete companion assemblies derive public keys below rho on
fresh transfer seeds without using public-key verification, relation count,
rank, preserving labels, or false-positive labels for selection.

Failure criterion:

If complete companion assemblies still stall at rank 1, the next branch should
target a public independent-row expander around retained roots rather than more
root-hyperplane scoring.
