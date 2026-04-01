# VLIW Scheduler — Reference Solution

## Optimization Levels

### Level 0 — Baseline (4,080 cycles, reward = 0.0)

One op per bundle. Iterate ops in input order; insert NOP bundles to satisfy RAW hazards:

```c
for (int i = 0; i < n_ops; i++) {
    int earliest = max(ready[rs1], ready[rs2]);
    while (b < earliest) out_bundles[b++] = EMPTY;
    place op in its slot; update ready[rd]; b++;
}
```

**Why slow**: never reorders ops or fills multiple slots per bundle. The 600 independent ALU and MEM ops each consume a full bundle even though their slots are free.

---

### Level 1 — ASAP multi-slot packing (~1,600 cycles, reward ≈ 0.42)

Build a ready queue. Each cycle, greedily pick one ALU, one MUL, one MEM op from the ready queue:

```c
// Build dependence counts
int n_preds[N]; memset(n_preds, 0, sizeof(n_preds));
// For each op i, compute earliest[i] from its source ready times
// Maintain ready lists per slot type
// Each bundle: pick first available op per slot
```

**Key improvement**: fills all three slots simultaneously. The 1,200 ALU + 1,200 MEM ops pack efficiently; MUL chains still stall but no longer block other slots.

**Bottleneck**: without priority ordering, MUL chains (listed last in input) may not be scheduled until after all ALU/MEM ops are drained.

---

### Level 2 — Critical-path list scheduling (~1,300 cycles, reward ≈ 0.50)

Compute `height[i]` = longest latency-weighted path from op `i` to any sink (op with no dependents):

```c
// Reverse topological order (process sinks first)
// height[sink] = 0
// height[i] = latency(i) + max(height[j] for j depending on i)
```

Use `height` as scheduling priority. When multiple ops are ready for the same slot, pick the one with the highest `height`. This forces MUL chains (height = 9 × 3 = 27 for the first op in a 10-op chain) to be scheduled early, hiding their latency behind independent ALU/MEM work.

**Implementation sketch**:

```c
#include <stdlib.h>
#include <string.h>
#include "solve.h"

#define MAX(a,b) ((a)>(b)?(a):(b))

static int latency(OpType t) {
    return t == OP_ALU ? 1 : t == OP_MUL ? 3 : t == OP_MEM ? 4 : 0;
}

int vliw_schedule(const VliwOp *ops, int n_ops, VliwBundle *out_bundles) {
    /* Build successor lists and predecessor counts */
    // ... dep graph from rs1/rs2 → rd ...

    /* Compute heights (reverse topological order) */
    int height[VLIW_MAX_OPS] = {0};
    // ... topo sort, then height[i] = lat(i) + max(height[succ]) ...

    /* ASAP scheduling with height priority */
    // ready_alu[], ready_mul[], ready_mem[] = min-heap by -height
    // each bundle: pop best ALU, MUL, MEM op from respective queues
    // after issuing, update ready times and enqueue newly-ready ops
    int b = 0;
    // ...
    return b;
}
```

**Why 1,300?** The ALU slot is the bottleneck: 1,200 independent ALU ops take 1,200 cycles minimum. The 60 MUL chains (each 10 ops × 3-cycle latency = 30 cycles) are fully hidden inside the ALU-bound schedule with good priority ordering.

---

### Level 3 — Register pressure + fine-grained tie-breaking (~1,050 cycles, reward ≈ 0.59)

Beyond critical-path height, two refinements push further:

1. **Earliest-cycle tie-breaking**: among ops with equal height, prefer the one with the smallest `earliest` cycle (furthest in the past = most urgent to schedule).

2. **Lookahead packing**: if the best-height op for ALU slot becomes available in cycle `c`, and there's an independent op available now, consider whether issuing now vs. waiting for a better bundle is worthwhile.

These micro-optimizations typically shave 200–300 cycles by reducing bubble overhead in the transition between the independent-op phase and the MUL-chain cleanup phase.

---

## Key Algorithmic Insight

The adversarial op ordering (MUL chains last) is the crux of the challenge. A naive scheduler that processes ops in input order will:
1. Fill 1,200 ALU bundles → 1,200 cycles
2. Fill 1,200 MEM bundles → 1,200 cycles
3. Then process 600 MUL ops with stalls → ~1,800 cycles
Total: **~4,200 cycles**

A critical-path scheduler recognizes that the MUL chains have `height = 27` (the 10-op chain's latency-weighted depth) and pulls them to the front of the schedule, overlapping their long latency with the abundant independent ALU/MEM work.

## References

- Fisher, J.A. "Very Long Instruction Word architectures and the ELI-512." ISCA 1983.
- Bernstein, D. & Rodeh, M. "Global instruction scheduling for superscalar machines." PLDI 1991.
- Muchnick, S. "Advanced Compiler Design and Implementation." Chapter 17: Instruction Scheduling. Morgan Kaufmann, 1997.
- Rau, B.R. & Glaeser, C.D. "Some scheduling techniques and an easily schedulable horizontal architecture for high performance scientific computing." MICRO 1981.
