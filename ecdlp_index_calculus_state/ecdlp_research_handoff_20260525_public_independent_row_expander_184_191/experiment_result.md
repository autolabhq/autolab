# Experiment Result

Primary artifacts:

- Selector: `ecdlp_index_calculus_state/ffe_public_independent_row_expander_selector_184_191.json`
- Replay: `ecdlp_index_calculus_state/ffe_public_independent_row_expander_replay_184_191.json`
- Probe script: `tasks/ecdlp_index_calculus/ffe_public_independent_row_expander_probe.py`

Selector:

- Anchor source: `ecdlp_index_calculus_state/ffe_public_single_hit_root_assembly_selector_184_191.json`
- Candidate source: `ecdlp_index_calculus_state/low_term_total2_candidate_signature_fixed_selector_184_191.json`
- Anchor cases: 10.
- Added public diversity cases: 18.
- Total retained source cases: 28.
- Challenge groups: 5.
- Retained surfaces: 27, made of 7 anchors and 20 expander surfaces.
- `22050.cf1@11731`: 10 cases, 11 retained surfaces.
- `67.a1@9803`: 18 cases, 16 retained surfaces.
- Selected source ops/rho: min 0.416, mean 0.76242336, max 0.952.

Replay:

- Context errors: 0.
- Retained-only public-key verified cases: 1/28.
- Same-challenge verified groups: 1/5.
- Challenge-group relation count sum: 6.
- Retained-only relation count sum: 9.
- Maximum rank: 2.
- Verified group: `22050.cf1@11731`, transfer 189, rank 2,
  derived secret 529.
- `67.a1@9803`: 18 cases across transfers 185, 190, and 191; 0 verified
  cases; retained groups stayed rank 1 or rank 0.

Comparison to companion replay:

- The expander increased selected cases from 10 to 28 and retained surfaces
  from 13 to 27.
- The same single `22050.cf1@11731` transfer-189 group verified.
- No new `67.a1@9803` relation system became rank 2.
- Relation counts did not increase at the challenge-group level, so the added
  same-family rows mainly added zero-event or duplicate-form scans.

Interpretation:

Public independent-row expansion inside the same low-term-total2 family is an
honest negative for the hard target.  It confirms that the rank-1 boundary is
not solved by simply adding more public below-rho cases around the existing
anchors.  The next branch should change the leaf selector or hit-stream source,
then reapply the same FFE single-hit-root gate and replay comparison.
