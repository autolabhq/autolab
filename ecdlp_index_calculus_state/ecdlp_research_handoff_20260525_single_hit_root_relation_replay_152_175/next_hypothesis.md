# Next Hypothesis

The next test should remove the last verifier-positive replay dependency:
choose row/leaf assemblies from public retained-root evidence before seeing
whether the relation system derives.

Recommended preregistered stress:

1. Freeze the current single-hit-root retention rule:
   `chosen_preserves_selected_root_pairs and not chosen_false_positive and below_rho`.
2. For fresh transfer indices 176-183, generate fixed-selector low-term-total2
   candidates as before.
3. Before public-key verification, select row/leaf assemblies using only:
   retained root-hyperplane metadata, public low-term-total2 leaf scores,
   direct/root ops-over-rho estimates, and surface uniqueness.
4. Replay only those preregistered assemblies through the verifier.
5. Report both retained-root derivation rate and the Pollard-rho-normalized
   public work estimate.

Success criterion:

A candidate family is promoted only if fresh preregistered assemblies produce
public-key-verified same-challenge relation systems below the rho proxy with no
post-hoc leaf choice.
