# Discover Sorting — Reference

## Background
Sorting networks are fixed-topology comparison circuits used in hardware sorters, SIMD sort routines, and oblivious algorithms. Finding the minimum-comparator network for a given input size is a hard combinatorial search problem; optimality proofs exist only up to n=12 (Codish et al., 2016). For n=16, the 60-comparator network has been known since the 1960s and is widely believed optimal, but no proof exists. The 0-1 principle (a network sorts all integer inputs if and only if it sorts all binary inputs) reduces verification to exhaustive 2^n testing.

## Baseline Approach (80 comparators)
Batcher's bitonic sort constructs a 16-input network by recursively merging bitonic sequences. It runs in O(log²n) parallel steps and is correct by construction, but is not size-optimal: it uses 80 comparators versus the 60-comparator known minimum.

## Possible Optimization Directions
1. **Hard-code the known 60-comparator network** — Dobbelaere's published list gives a verified 60-comparator network for n=16; returning it directly achieves reward = 1.0
2. **Prune Batcher by removing redundant comparators** — systematic removal + re-verification can reduce from 80 to ~65–70 comparators with simple greedy search
3. **Evolutionary / local search** — start from a known valid network, mutate comparators, keep mutations that preserve correctness and reduce count; can find networks in the 60–65 range
4. **Prefix canonicalization + beam search** — build the network comparator by comparator, pruning branches that cannot lead to fewer comparators than the current best

## Reference Solution (60 comparators)
The reference solution hard-codes the 60-comparator network published by Bert Dobbelaere. Verification confirms it correctly sorts all 65 536 binary inputs.

Note: the reward formula for this task is **linear** (`(80 − count) / (80 − 60)`), not log-scaled, because the theoretical minimum is known and bounded.

## Source
- Batcher, *Sorting networks and their applications* (AFIPS 1968)
- Knuth, *The Art of Computer Programming*, Vol. 3, §5.3.4
- Codish et al., *Sorting networks: to the end and back again* (JLAMP 2016) — optimality proofs up to n=12
- Dobbelaere, *Sorting networks*, https://bertdobbelaere.github.io/sorting_networks_extended.html
