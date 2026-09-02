# Next Hypothesis

The next selector should explicitly optimize for independent row forms around
public retained-root anchors.

Recommended preregistered 192-199 test:

1. Keep the public below-rho total2 candidate generation.
2. Keep single-hit-root anchor selection exactly as in 184-191.
3. For each anchor challenge, retain a small public companion pool rather than
   only the companion rows from the selected source case.
4. Rank companion candidates by public row diversity:
   distinct salt, distinct row schedule key, distinct leaf signature, and low
   public source ops/rho.
5. Replay anchor plus top-k public companion rows for k in a fixed grid such as
   1, 2, 3, 4.
6. Report unique relation forms and same-challenge public derivation.

Success criterion:

Promote only if the fixed public independent-row expander raises hard-target
groups, especially `67.a1@9803`, from rank 1 to public-key-verified rank 2
below the rho proxy.

Failure criterion:

If public diversity expansion still yields one unique form on `67.a1@9803`,
the next branch should change the leaf selector or hit-stream source rather
than adding more rows from the same low-term-total2 family.
