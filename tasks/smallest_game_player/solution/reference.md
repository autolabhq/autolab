# Smallest Game Player — Reference

## Background
Training neural networks to play games perfectly is a well-studied problem, but the focus is usually on accuracy or sample efficiency. This task inverts the objective: given a fixed accuracy threshold, minimize the number of learnable parameters. The problem is motivated by on-device inference, embedded AI, and the open question of how small a model needs to be to internalize a complete strategy for a simple combinatorial game. 4×4 gravity Connect-3 has ~4 000 reachable non-terminal states, making exhaustive enumeration trivial but still requiring the model to generalize to a held-out split.

## Baseline Approach (17 924 parameters)
A standard 2-layer `TransformerEncoder` with `d_model=32`, `nhead=2`, `dim_feedforward=64`, separate token and positional embeddings, and a linear policy head. This is the smallest "textbook" transformer configuration and achieves ~95% accuracy on the held-out split, but carries unnecessary width and depth for a 17-token, 4-class problem.

## Possible Optimization Directions
1. **Switch architecture** — transformers learn positional attention patterns, but this task benefits more from hand-crafted tactical features (immediate wins, blocks, threats) than from learned token interactions
2. **Feature engineering** — encode per-column: can I win here? must I block here? per-line: how many mine/theirs/empty? This directly captures Connect-3 strategy
3. **2-step lookahead features** — for each candidate column, simulate placement and check whether opponent can immediately win, or whether I can force a win on my next move regardless of opponent's response
4. **Compact MLP** — with ~156 good features, a 2-layer MLP (48→24→4) suffices; ~9k params vs 17k for the transformer
5. **Reduce hidden size** — once strong features are in place, further shrink hidden layers (e.g. 32→16) while monitoring accuracy on the 95% threshold

## Reference Solution (913 parameters)
**Key insight**: reformulate 4-class column prediction as *action scoring with weight sharing*.

For each board state, compute a 36-dimensional feature vector for each of the 4 candidate columns, then apply a single shared network `Linear(36→24) → ReLU → Linear(24→1)` to score each column. The predicted move is the argmax over 4 scores. Because weights are shared across columns, parameter count scales with feature dimensionality rather than num_actions × feature_dim.

The 36 per-action features capture: column legality, center preference, current/remaining height, all-column heights and legals (×4 each), immediate win/block flags for this column and globally, plus post-move line-class counts (my 2-in-row, 1-in-row, opponent 2-in-row, etc.) and their deltas vs the current board.

Architecture: `Linear(36, 24) + bias + Linear(24, 1) + bias = 36×24 + 24 + 24 + 1 = 913 params`. Trained for 400 epochs with AdamW (`lr=1e-2`, `weight_decay=1e-4`), full-batch gradient descent. Achieves ~95.3% test accuracy (threshold ≥95%).

## Source
- Shannon, *Programming a Computer for Playing Chess* (1950) — original game-tree search
- Silver et al., *Mastering the Game of Go without Human Knowledge* (Nature 2017) — policy network compression
- Frankle & Carlin, *The Lottery Ticket Hypothesis* (ICLR 2019) — model pruning
- Michel et al., *Are Sixteen Heads Really Better than One?* (NeurIPS 2019) — attention head pruning
