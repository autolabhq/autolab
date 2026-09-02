# Hypothesis

The 176-183 public assembly test showed a rank-1 failure when replay kept only
one retained root anchor, and rank recovery when the full companion row/leaf
assembly was kept.  The fresh 184-191 test preregisters that complete-companion
rule:

1. Select public below-rho low-term-total2 candidate cases.
2. Gate anchors by the single-hit-root pre-factor proxy.
3. Score roots with the unique-leaf selected-hit-root policy.
4. Select public anchor cases below the rho proxy.
5. Retain every row/leaf surface in those selected cases before verifier
   replay.

Selection still forbids public-key verification, relation count, rank,
preserving labels, false-positive labels, and preserving-conditioned below-rho
labels.
