#include "solve.h"
#include <string.h>

/*
 * Critical-path list scheduler for VLIW.
 *
 * Dependency graph:
 *   RAW: reader depends on last writer (with writer's latency)
 *   WAW: Only final writer per register needs to be after all other writers.
 *        Same-type WAW has latency 1 (can't share slot).
 *        Cross-type WAW where later writer has higher slot: latency 0.
 *   WAR: Only needed when a reader's value could be corrupted by a writer.
 *        Since all independent ops read r0 (hardwired), WAR is rare.
 *
 * Key optimization: for registers that are never read between writes,
 * replace WAW chains with fan-in to the final writer only.
 */

#define MAX_OPS   VLIW_MAX_OPS
#define MAX_SUCCS 64

static int lat(OpType t) {
    switch (t) {
        case OP_ALU: return 1;
        case OP_MUL: return 3;
        case OP_MEM: return 4;
        default:     return 0;
    }
}

static int slot_order(OpType t) {
    switch (t) {
        case OP_ALU: return 0;
        case OP_MUL: return 1;
        case OP_MEM: return 2;
        default:     return -1;
    }
}

typedef struct { int succ; int min_lat; } Edge;

static Edge succs_arr[MAX_OPS][MAX_SUCCS];
static int  n_succs[MAX_OPS];
static int  n_preds[MAX_OPS];
static int  height_arr[MAX_OPS];
static int  earliest[MAX_OPS];

static int ready_alu[MAX_OPS], n_ready_alu;
static int ready_mul[MAX_OPS], n_ready_mul;
static int ready_mem[MAX_OPS], n_ready_mem;

static const VliwOp NOP_OP = { OP_NOP, -1, -1, -1, 0, -1 };

static void add_edge(int from, int to, int min_lat) {
    if (from == to || from < 0 || to < 0) return;
    for (int i = 0; i < n_succs[from]; i++) {
        if (succs_arr[from][i].succ == to) {
            if (min_lat > succs_arr[from][i].min_lat)
                succs_arr[from][i].min_lat = min_lat;
            return;
        }
    }
    if (n_succs[from] < MAX_SUCCS) {
        succs_arr[from][n_succs[from]].succ = to;
        succs_arr[from][n_succs[from]].min_lat = min_lat;
        n_succs[from]++;
        n_preds[to]++;
    }
}

/* Track all writers to each register */
#define MAX_WRITERS_PER_REG 256
static int reg_writers[VLIW_REGS][MAX_WRITERS_PER_REG];
static int n_reg_writers[VLIW_REGS];

int vliw_schedule(const VliwOp *ops, int n_ops, VliwBundle *out_bundles) {
    if (n_ops == 0) return 0;

    memset(n_succs, 0, n_ops * sizeof(int));
    memset(n_preds, 0, n_ops * sizeof(int));
    memset(earliest, 0, n_ops * sizeof(int));

    for (int r = 0; r < VLIW_REGS; r++)
        n_reg_writers[r] = 0;

    /*
     * Pass 1: Identify last writer per register and track all writers.
     */
    int final_writer[VLIW_REGS];
    for (int r = 0; r < VLIW_REGS; r++) final_writer[r] = -1;

    /* Also track whether register r is ever read (by any op, between writes) */
    /* Actually, track for each consecutive pair of writes whether a read occurs between */
    /* Simpler: track if register r is ever used as rs1/rs2 by any op */
    int reg_ever_read[VLIW_REGS];
    memset(reg_ever_read, 0, sizeof(reg_ever_read));

    for (int i = 0; i < n_ops; i++) {
        if (ops[i].rs1 > 0) reg_ever_read[ops[i].rs1] = 1;
        if (ops[i].rs2 > 0) reg_ever_read[ops[i].rs2] = 1;
        if (ops[i].rd > 0) {
            int r = ops[i].rd;
            if (n_reg_writers[r] < MAX_WRITERS_PER_REG)
                reg_writers[r][n_reg_writers[r]++] = i;
            final_writer[r] = i;
        }
    }

    /*
     * Pass 2: Build dependency edges.
     */
    int last_writer[VLIW_REGS];
    int last_reader[VLIW_REGS];
    int had_read[VLIW_REGS];

    for (int r = 0; r < VLIW_REGS; r++) {
        last_writer[r] = -1;
        last_reader[r] = -1;
        had_read[r] = 0;
    }

    for (int i = 0; i < n_ops; i++) {
        int rs1 = ops[i].rs1, rs2 = ops[i].rs2, rd = ops[i].rd;

        /* RAW: this op reads a register written by last writer */
        if (rs1 > 0 && last_writer[rs1] >= 0) {
            add_edge(last_writer[rs1], i, lat(ops[last_writer[rs1]].type));
            had_read[rs1] = 1;
            last_reader[rs1] = i;
        }
        if (rs2 > 0 && last_writer[rs2] >= 0) {
            add_edge(last_writer[rs2], i, lat(ops[last_writer[rs2]].type));
            had_read[rs2] = 1;
            last_reader[rs2] = i;
        }

        /* Track reads for WAR even without prior writer */
        if (rs1 > 0) { had_read[rs1] = 1; last_reader[rs1] = i; }
        if (rs2 > 0) { had_read[rs2] = 1; last_reader[rs2] = i; }

        if (rd > 0) {
            /* WAR: if any op read this register since last write, new writer
             * must be after the reader */
            if (had_read[rd] && last_reader[rd] >= 0) {
                add_edge(last_reader[rd], i, 1);
            }

            if (reg_ever_read[rd]) {
                /* This register IS read somewhere → use conservative WAW chains */
                if (last_writer[rd] >= 0) {
                    int ps = slot_order(ops[last_writer[rd]].type);
                    int cs = slot_order(ops[i].type);
                    int waw_lat = (cs > ps) ? 0 : 1;
                    add_edge(last_writer[rd], i, waw_lat);
                }
            } else {
                /* Register is NEVER read → only final writer matters.
                 * Add edge from this writer to final writer (fan-in). */
                if (i != final_writer[rd] && final_writer[rd] >= 0) {
                    int cs = slot_order(ops[final_writer[rd]].type);
                    int ps = slot_order(ops[i].type);
                    int waw_lat = (cs > ps) ? 0 : 1;
                    add_edge(i, final_writer[rd], waw_lat);
                }
            }

            last_writer[rd] = i;
            had_read[rd] = 0;
            last_reader[rd] = -1;
        }
    }

    /* Compute heights via reverse topological order */
    {
        static Edge preds_list[MAX_OPS][MAX_SUCCS];
        static int n_preds_list[MAX_OPS];
        int out_degree[MAX_OPS];
        memset(n_preds_list, 0, n_ops * sizeof(int));

        for (int i = 0; i < n_ops; i++) {
            height_arr[i] = lat(ops[i].type);
            out_degree[i] = n_succs[i];
        }

        for (int i = 0; i < n_ops; i++)
            for (int e = 0; e < n_succs[i]; e++) {
                int j = succs_arr[i][e].succ;
                if (n_preds_list[j] < MAX_SUCCS) {
                    preds_list[j][n_preds_list[j]].succ = i;
                    preds_list[j][n_preds_list[j]].min_lat = succs_arr[i][e].min_lat;
                    n_preds_list[j]++;
                }
            }

        static int queue[MAX_OPS];
        int qh = 0, qt = 0;
        int rev_remaining[MAX_OPS];
        for (int i = 0; i < n_ops; i++) {
            rev_remaining[i] = n_succs[i];
            if (out_degree[i] == 0) queue[qt++] = i;
        }
        while (qh < qt) {
            int j = queue[qh++];
            for (int e = 0; e < n_preds_list[j]; e++) {
                int i = preds_list[j][e].succ;
                int c = preds_list[j][e].min_lat + height_arr[j];
                if (c > height_arr[i]) height_arr[i] = c;
                if (--rev_remaining[i] == 0) queue[qt++] = i;
            }
        }
    }

    /* List scheduling */
    n_ready_alu = n_ready_mul = n_ready_mem = 0;
    for (int i = 0; i < n_ops; i++) {
        if (n_preds[i] == 0) {
            earliest[i] = 0;
            switch (ops[i].type) {
                case OP_ALU: ready_alu[n_ready_alu++] = i; break;
                case OP_MUL: ready_mul[n_ready_mul++] = i; break;
                case OP_MEM: ready_mem[n_ready_mem++] = i; break;
                default: break;
            }
        }
    }

    int n_scheduled = 0, cycle = 0;

    #define SCHED(rlist, nready, field) do { \
        int _best=-1,_bh=-1,_bi=-1; \
        for (int _i=0;_i<nready;_i++) { \
            int _op=rlist[_i]; \
            if(earliest[_op]<=cycle && height_arr[_op]>_bh) \
                {_best=_op;_bh=height_arr[_op];_bi=_i;} \
        } \
        if (_best>=0) { \
            bun.field=ops[_best]; n_scheduled++; \
            rlist[_bi]=rlist[--nready]; \
            for(int _e=0;_e<n_succs[_best];_e++){ \
                int _s=succs_arr[_best][_e].succ; \
                int _rc=cycle+succs_arr[_best][_e].min_lat; \
                if(_rc>earliest[_s])earliest[_s]=_rc; \
                if(--n_preds[_s]==0){ \
                    switch(ops[_s].type){ \
                        case OP_ALU:ready_alu[n_ready_alu++]=_s;break; \
                        case OP_MUL:ready_mul[n_ready_mul++]=_s;break; \
                        case OP_MEM:ready_mem[n_ready_mem++]=_s;break; \
                        default:break; \
                    } \
                } \
            } \
        } \
    } while(0)

    while (n_scheduled < n_ops) {
        VliwBundle bun;
        bun.alu = NOP_OP;
        bun.mul = NOP_OP;
        bun.mem = NOP_OP;

        SCHED(ready_alu, n_ready_alu, alu);
        SCHED(ready_mul, n_ready_mul, mul);
        SCHED(ready_mem, n_ready_mem, mem);

        out_bundles[cycle] = bun;
        cycle++;
        if (cycle > VLIW_MAX_BUNDLES - 1) break;
    }

    return cycle;
}
