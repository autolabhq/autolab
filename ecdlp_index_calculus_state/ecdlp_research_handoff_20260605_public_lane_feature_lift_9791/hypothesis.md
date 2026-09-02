# Hypothesis

The broad low-term total2 carrier `selected_has=13` is not the whole public
mechanism.  Inside that carrier, non-adjacent source salt pairs appear to be
the reusable public sublane that preserves rank-gain comparators while leaving
fresh direct/rank targets to validate.

Operational hypothesis:

- Primary queue carrier: `selected_has=13 AND salt_adjacent=False`.
- High-precision diagnostic sublane:
  `selected_has=13 AND salt_adjacent=False AND salt_min_mod4=3`.
- The diagnostic sublane should be treated as a public row-routing rule, not as
  a speedup proof, until its missing full-family transfers receive direct/rank
  exports.

Why this matters for the index-calculus line: the rule is expressed only in
public support/salt geometry and does not inspect the accepted missing column
label.  If the missing rows reproduce the exported rank-gain behavior, the
relation search has a smaller public queue to feed into the FFE/summation
polynomial stage.
