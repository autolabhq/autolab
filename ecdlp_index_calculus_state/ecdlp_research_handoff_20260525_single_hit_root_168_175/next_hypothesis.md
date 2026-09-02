# Next Hypothesis

Start the cross-window relation-assembly bridge before spending another turn
only extending the same replication sequence.

Concrete bridge for 152-175:

- Inputs: strict single-hit-root policy rows from 152-159, 160-167, and
  168-175.
- Group retained hyperplanes by target, challenge seed, and transfer index.
- Treat each preserving root hyperplane as a candidate relation-generation
  event, not as a completed ECDLP solve.
- Count the verifier equations and ranks already present in the source
  signature cases for retained surfaces.
- Charge the online root-selection path against the same `generic_rho_steps`.
- Report whether any target has enough independent retained events to derive a
  verifier secret without returning a scalar.

Follow-up replication:

- A fourth total2 window, 176-183, is useful only if the relation bridge needs
  more retained hyperplanes or if 152-175 reveals a target-specific weakness.
- Keep the same `single_prefactor_hit_root` gate and controls if this window is
  run.

Open obligations:

- No end-to-end ECDLP speedup is claimed yet.
- Multi-hit-root surfaces need their own discriminator before inclusion.
- Total3/4 stays excluded until a public channel discriminator rejects the
  known non-preserving total3/4 failure mode.
