# Public Factor Portfolio Robustness

The current FFE quotient lead is the guarded public-factor quadratic-root
portfolio on low-term total-2 surfaces.  The live work order says the
slice/quadratic root route is useful only if the factor choice becomes public
and held-out-predictive, rather than an oracle replay of preserving roots.

Hypothesis: after the public guard
`selected_leaf_count_eq1_and_factor_zero_eq1`, the cheapest public-zero
portfolio remains leave-one-window-out robust, and the remaining fragility is
localized to policy-family eligibility rather than verifier labels.

Controls:

1. Baseline leave-one-window-out with all clean policy families, including hash
   policies.
2. Leave-one-window-out with hash policies excluded.
3. Leave-one-window-out with at least two training surfaces required per
   eligible policy.
4. Leave-one-window-out with a strict `max_train_worst_ratio=0.75` eligibility
   cap.

The test uses mounted live AutoLab state from
`/Volumes/Volume/autolab/ecdlp_index_calculus_state` as read-only input and
writes local worktree artifacts.
