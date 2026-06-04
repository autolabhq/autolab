# Adversarial Splay Tree Access Sequence -- Reference

## Background

Splay trees (Sleator and Tarjan, 1985) are self-adjusting binary search trees: every access splays the touched key to the root using a sequence of `zig`, `zig-zig`, and `zig-zag` rotations. The classical amortized analysis bounds any sequence of `m` accesses on a splay tree of `n` keys by `O((m + n) log n)` total rotations, which is the worst case up to constant factors.

The Dynamic Optimality Conjecture asks whether the splay heuristic is competitive with the best offline binary-search-tree algorithm; partial bounds (Sequential Access, Working Set, Static Optimality, Dynamic Finger) cap performance from above on many "structured" access patterns. To force the splay tree toward its `Theta(m log n)` worst case we need a sequence that defeats all of these bounds simultaneously: it must avoid sequential structure, avoid temporal locality (working set), and avoid spatial locality (dynamic finger).

## Baseline approach

A uniform-random sequence of 4096 keys drawn from `[0, 1023]` is asymptotically `Theta(m log n)` but with relatively small constants because random accesses get "lucky" with the working-set bound: a freshly-splayed key sits near the root, and any repeat access in the next few hits is cheap. The uniform-random baseline therefore lands well below the true worst case.

## Reference solution: bit-reversal cycled

The reference sequence is the **bit-reversal permutation of `0..1023`, cycled four times** (1024 keys per pass times 4 passes equals 4096 accesses):

```
pass 0,1,2,3: i in 0..1023, output bit_reverse(i, 10)
```

So the first sixteen accesses are `0, 512, 256, 768, 128, 640, 384, 896, 64, 576, 320, 832, 192, 704, 448, 960`, and so on.

Why bit-reversal works:

- **Defeats sequential access.** Bit-reversal is the canonical "anti-sequential" permutation: consecutive accesses are maximally distant in key space.
- **Defeats working set.** Each pass touches every key exactly once, so the working set window is a full pass; after a key is splayed it is never re-accessed for ~1023 more steps, which exceeds any short-window working-set bound.
- **Defeats dynamic finger.** Consecutive bit-reversed indices alternate between "distant halves" of the key space (high bit flipping each step), so the rank distance between successive accesses is `Theta(n)`, not `O(1)`.
- **Cycling preserves adversariality.** Re-running the same permutation four times does not let the splay tree settle into a cheap configuration; the structure that minimizes rotations for one bit-reversed pass is destroyed by the next pass's first access.

The cycled-bit-reversal is a textbook hard case for splay trees and is widely used in empirical splay-tree studies as the "near-worst-case" pattern.

## Other patterns explored

The following alternatives were tried during calibration and discarded:

- **Sequential `0..1023` cycled**: very fast for splay trees because the Sequential Access Theorem (Tarjan, 1985) bounds `n` consecutive in-order accesses by `O(n)` total rotations.
- **Reverse sequential**: similar to sequential, also covered by the access theorem.
- **Alternating min/max** (`0, 1023, 1, 1022, ...`): the splay tree quickly brings both extremes to the top of the tree, leaving subsequent accesses with `O(1)` amortized cost.
- **Random ping-pong**: working-set theorem gives this `O(log w)` per access for window size `w`.
- **Greedy "deepest node" online adversary**: actually performs worse than bit-reversal because it tends to revisit the just-splayed-up subtree.
- **Bit-reversal with per-pass shifts and hill-climbed local swaps**: marginal gains (1-3%) over plain cycled bit-reversal. Not worth the loss of explainability.

## Sources

- Sleator, D. D., and Tarjan, R. E. (1985). "Self-Adjusting Binary Search Trees." *Journal of the ACM*, 32(3), 652-686.
- Tarjan, R. E. (1985). "Sequential Access in Splay Trees Takes Linear Time." *Combinatorica*, 5(4), 367-378.
- Cole, R. (2000). "On the Dynamic Finger Conjecture for Splay Trees." *SIAM Journal on Computing*, 30(1), 44-85.
- Wang, C.-C., Derryberry, J., and Sleator, D. D. (2006). "O(log log n)-competitive dynamic binary search trees." *SODA*.
