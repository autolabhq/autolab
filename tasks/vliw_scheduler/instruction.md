# VLIW Instruction Scheduler

Implement `vliw_schedule()` in `/app/solve.c` to pack a sequential instruction stream into VLIW bundles, minimizing the total cycle count.

## Problem

A VLIW processor issues three operations simultaneously per cycle (bundle):

| Slot | Operation types | Result latency |
|------|----------------|----------------|
| ALU  | ADD, AND, OR, XOR, SHL | 1 cycle |
| MUL  | MUL | 3 cycles |
| MEM  | LD | 4 cycles |

**Hazard rule**: if op A writes register `r` in bundle `i` with latency `L`, then any op reading `r` must be placed in bundle `j ≥ i + L`. All three slots in a bundle issue simultaneously (read-before-write: sources are sampled at the start of the cycle).

The benchmark generates **3,000 ops** with an adversarial ordering:
- 1,200 independent ALU ops (all read r0=0)
- 1,200 independent MEM ops (all read r0=0)
- 600 MUL ops forming 60 sequential chains of 10 (each op depends on the previous) — listed **last**

The baseline scheduler issues one op per bundle and never reorders, giving ~4,080 cycles. A good scheduler extracts instruction-level parallelism across all three slots and prioritises the long-latency MUL chains.

## Files

| File | Role |
|------|------|
| `solve.c` | **Edit this file only** |
| `solve.h` | Fixed interface — structs and function signature |
| `main.c`  | Program generator + correctness checker (read-only) |
| `Makefile` | Build config (read-only) |

## Interface

```c
// Schedule n_ops operations into out_bundles[].
// Returns the number of bundles emitted (= cycle count).
int vliw_schedule(const VliwOp *ops, int n_ops, VliwBundle *out_bundles);
```

```c
typedef struct {
    OpType  type;   /* OP_NOP / OP_ALU / OP_MUL / OP_MEM */
    int     rd;     /* destination register (-1 = none)   */
    int     rs1;    /* source register 1   (-1 = none)    */
    int     rs2;    /* source register 2   (-1 = none)    */
    int32_t imm;    /* immediate value                    */
    int     id;     /* original index — preserve in output */
} VliwOp;

typedef struct { VliwOp alu; VliwOp mul; VliwOp mem; } VliwBundle;
```

## Rules

- Edit `/app/solve.c` **only**.
- C standard library and GCC builtins (`__builtin_*`, `__attribute__`) are available.
- Single-threaded only. No new external libraries.
- Output must be structurally correct (each op in the right slot type, every op emitted exactly once) and hazard-free — wrong results score 0.

