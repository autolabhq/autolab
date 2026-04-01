#pragma once
#include <stdint.h>

/* ── Machine parameters ─────────────────────────────────── */
#define VLIW_REGS    32          /* r0=0 (hardwired), r1..r31 general */
#define VLIW_MAX_OPS 4096        /* upper bound on input op count      */
#define VLIW_MAX_BUNDLES (VLIW_MAX_OPS * 6)  /* worst-case output      */

typedef enum {
    OP_NOP = 0,
    OP_ALU = 1,   /* ADD/AND/OR/XOR/SHL  latency 1  → ALU slot */
    OP_MUL = 2,   /* MUL                 latency 3  → MUL slot */
    OP_MEM = 3,   /* LD                  latency 4  → MEM slot */
} OpType;

typedef struct {
    OpType  type;   /* OP_NOP / OP_ALU / OP_MUL / OP_MEM          */
    int     rd;     /* destination register (-1 = none)            */
    int     rs1;    /* source register 1   (-1 = none)             */
    int     rs2;    /* source register 2   (-1 = none, imm used)   */
    int32_t imm;    /* immediate (MEM: effective addr = rs1 + imm) */
    int     id;     /* original index in ops[] – set by main.c     */
} VliwOp;

typedef struct {
    VliwOp alu;    /* OP_ALU or OP_NOP */
    VliwOp mul;    /* OP_MUL or OP_NOP */
    VliwOp mem;    /* OP_MEM or OP_NOP */
} VliwBundle;

/*
 * Schedule n_ops operations into VLIW bundles.
 * Returns the number of bundles emitted (= cycle count).
 * out_bundles must have room for at least VLIW_MAX_BUNDLES entries.
 */
int vliw_schedule(const VliwOp *ops, int n_ops, VliwBundle *out_bundles);
