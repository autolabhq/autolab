#include "solve.h"
#include <string.h>

static int latency(OpType t) {
    switch (t) {
        case OP_ALU: return 1;
        case OP_MUL: return 3;
        case OP_MEM: return 4;
        default:     return 0;
    }
}

static const VliwBundle EMPTY_BUNDLE = {
    { OP_NOP, -1, -1, -1, 0, -1 },
    { OP_NOP, -1, -1, -1, 0, -1 },
    { OP_NOP, -1, -1, -1, 0, -1 },
};

#define MAX(a,b) ((a)>(b)?(a):(b))

int vliw_schedule(const VliwOp *ops, int n_ops, VliwBundle *out_bundles) {
    /* ready[r] = earliest bundle index at which r may be read */
    int ready[VLIW_REGS];
    memset(ready, 0, sizeof(ready));

    int b = 0;

    for (int i = 0; i < n_ops; i++) {
        /* Compute earliest bundle this op can issue */
        int earliest = 0;
        if (ops[i].rs1 >= 0) earliest = MAX(earliest, ready[ops[i].rs1]);
        if (ops[i].rs2 >= 0) earliest = MAX(earliest, ready[ops[i].rs2]);

        /* Emit NOP bundles to reach earliest */
        while (b < earliest) {
            out_bundles[b++] = EMPTY_BUNDLE;
        }

        /* Issue the op in the correct slot */
        VliwBundle bun = EMPTY_BUNDLE;
        switch (ops[i].type) {
            case OP_ALU: bun.alu = ops[i]; break;
            case OP_MUL: bun.mul = ops[i]; break;
            case OP_MEM: bun.mem = ops[i]; break;
            default: break;
        }
        out_bundles[b] = bun;

        /* Update ready time for destination register */
        if (ops[i].rd >= 0) {
            ready[ops[i].rd] = b + latency(ops[i].type);
        }
        b++;
    }

    return b;
}
