/*
 * main.c — VLIW Scheduler Benchmark Harness
 *
 * Generates a 3000-op program, calls vliw_schedule(), verifies correctness,
 * and reports cycle count.
 *
 * Output: n_ops=3000 cycles=<N> result=ok|WRONG
 *
 * DO NOT MODIFY THIS FILE.
 */

#include "solve.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

/* ── LCG ──────────────────────────────────────────────── */
static uint32_t lcg_state;
static void lcg_seed(uint32_t s) { lcg_state = s; }
static uint32_t lcg_next(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}

/* ── Program generation ───────────────────────────────── */
/*
 * N = 3000 ops:
 *   - 1200 independent ALU ops  (r1..r20 used as destinations, rs from r0)
 *   - 1200 independent MEM ops  (r1..r20 used as destinations, rs from r0)
 *   -  600 MUL chain ops: 60 chains × 10 ops, where each op reads the
 *          previous op's rd.  All 600 are appended LAST (adversarial order).
 *
 * Register allocation:
 *   - Independent ALU/MEM: destinations cycle through r1..r20, src = r0
 *   - MUL chains: chain k uses registers starting at r(21 + k%10); each
 *     chain writes a sequence of fresh registers to avoid WAW hazards across
 *     chains.  We use r21..r31 (11 regs) — chains share registers only
 *     across non-overlapping chains.
 *
 * To avoid WAW/WAR hazards in the independent block we assign destinations
 * with a stride so neighbouring ops don't alias.
 */

#define N_OPS        3000
#define N_MUL_CHAINS  60
#define CHAIN_LEN     10
#define N_MUL_OPS    (N_MUL_CHAINS * CHAIN_LEN)   /* 600 */
#define N_ALU_OPS    1200
#define N_MEM_OPS    1200

/* ALU sub-types (all latency 1) */
typedef enum { A_ADD=0, A_AND, A_OR, A_XOR, A_SHL, A_COUNT } AluSub;

static VliwOp  g_ops[N_OPS];
static VliwBundle g_bundles[VLIW_MAX_BUNDLES];

/*
 * Simulate register file execution of a sequential program.
 * r0 is hardwired to 0. Returns final register state in regs[].
 */
static void sim_sequential(const VliwOp *ops, int n, uint32_t regs[VLIW_REGS]) {
    memset(regs, 0, VLIW_REGS * sizeof(uint32_t));
    for (int i = 0; i < n; i++) {
        uint32_t v1 = (ops[i].rs1 >= 0) ? regs[ops[i].rs1] : 0;
        uint32_t v2 = (ops[i].rs2 >= 0) ? regs[ops[i].rs2] : (uint32_t)ops[i].imm;
        uint32_t res = 0;
        switch (ops[i].type) {
            case OP_ALU:
                /* encode sub-type in imm upper bits: bits 31..28 */
                switch ((ops[i].imm >> 28) & 0xF) {
                    case A_ADD: res = v1 + v2; break;
                    case A_AND: res = v1 & v2; break;
                    case A_OR:  res = v1 | v2; break;
                    case A_XOR: res = v1 ^ v2; break;
                    case A_SHL: res = v1 << (v2 & 31); break;
                    default:    res = v1 + v2; break;
                }
                break;
            case OP_MUL: res = v1 * v2; break;
            case OP_MEM: res = v1 + (uint32_t)ops[i].imm; break;  /* simulated: addr as value */
            default: continue;
        }
        if (ops[i].rd > 0) regs[ops[i].rd] = res;
    }
}

/*
 * Simulate register file execution of VLIW bundles.
 * Simultaneous-issue: snapshot sources BEFORE applying writes in each bundle.
 */
static void sim_vliw(const VliwBundle *bundles, int n_bundles,
                     uint32_t regs[VLIW_REGS]) {
    memset(regs, 0, VLIW_REGS * sizeof(uint32_t));

    for (int b = 0; b < n_bundles; b++) {
        const VliwOp *slots[3] = { &bundles[b].alu, &bundles[b].mul, &bundles[b].mem };
        /* Snapshot sources */
        uint32_t src1[3], src2[3];
        for (int s = 0; s < 3; s++) {
            src1[s] = (slots[s]->rs1 >= 0) ? regs[slots[s]->rs1] : 0;
            src2[s] = (slots[s]->rs2 >= 0) ? regs[slots[s]->rs2] : (uint32_t)slots[s]->imm;
        }
        /* Apply writes */
        for (int s = 0; s < 3; s++) {
            if (slots[s]->type == OP_NOP) continue;
            uint32_t res = 0;
            switch (slots[s]->type) {
                case OP_ALU:
                    switch ((slots[s]->imm >> 28) & 0xF) {
                        case A_ADD: res = src1[s] + src2[s]; break;
                        case A_AND: res = src1[s] & src2[s]; break;
                        case A_OR:  res = src1[s] | src2[s]; break;
                        case A_XOR: res = src1[s] ^ src2[s]; break;
                        case A_SHL: res = src1[s] << (src2[s] & 31); break;
                        default:    res = src1[s] + src2[s]; break;
                    }
                    break;
                case OP_MUL: res = src1[s] * src2[s]; break;
                case OP_MEM: res = src1[s] + (uint32_t)slots[s]->imm; break;
                default: break;
            }
            if (slots[s]->rd > 0) regs[slots[s]->rd] = res;
        }
    }
}

static void generate_program(void) {
    lcg_seed(0xDEADC0DEUL);
    int idx = 0;

    /* 1200 independent ALU ops: dst in r1..r20, src = r0 */
    for (int i = 0; i < N_ALU_OPS; i++) {
        AluSub sub = (AluSub)(lcg_next() % A_COUNT);
        int rd  = (int)(lcg_next() % 20) + 1;   /* r1..r20 */
        uint32_t imm_val = lcg_next() & 0x0FFFFFFF;  /* low 28 bits = immediate value */
        g_ops[idx].type = OP_ALU;
        g_ops[idx].rd   = rd;
        g_ops[idx].rs1  = 0;   /* r0 = 0 */
        g_ops[idx].rs2  = -1;  /* use imm */
        g_ops[idx].imm  = (int32_t)(((uint32_t)sub << 28) | imm_val);
        g_ops[idx].id   = idx;
        idx++;
    }

    /* 1200 independent MEM ops: dst in r1..r20, src = r0 */
    for (int i = 0; i < N_MEM_OPS; i++) {
        int rd  = (int)(lcg_next() % 20) + 1;
        int32_t imm = (int32_t)(lcg_next() & 0xFFFF);
        g_ops[idx].type = OP_MEM;
        g_ops[idx].rd   = rd;
        g_ops[idx].rs1  = 0;
        g_ops[idx].rs2  = -1;
        g_ops[idx].imm  = imm;
        g_ops[idx].id   = idx;
        idx++;
    }

    /* 60 MUL chains × 10 ops — appended LAST (adversarial ordering).
     * Chain k: uses registers starting at base_reg.
     * We give each chain its own dedicated register band to avoid
     * cross-chain hazards in the baseline scheduler.
     * r21..r31 → 11 regs; chains cycle: chain k writes r(21 + k%11).
     * Within a chain: op j writes r(21 + k%11) alternating with scratch.
     * Simpler: each chain gets 1 output reg; intermediate values reuse
     * the same reg (chain k: reg = 21 + k%11).  Works because chains
     * are sequential internally.
     */
    for (int k = 0; k < N_MUL_CHAINS; k++) {
        int reg = 21 + (k % 11);   /* r21..r31 */
        /* First op: multiply r0 (=0) by a constant seeded from LCG to make
         * the chain non-trivial.  Actually let's just use r0 * r0 = 0; that's
         * fine for correctness checking. */
        /* op 0: rd=reg, rs1=r0, rs2=r0  → reg = 0*0 = 0 */
        g_ops[idx].type = OP_MUL;
        g_ops[idx].rd   = reg;
        g_ops[idx].rs1  = 0;
        g_ops[idx].rs2  = 0;
        g_ops[idx].imm  = 0;
        g_ops[idx].id   = idx;
        idx++;
        /* ops 1..9: rd=reg, rs1=reg, rs2=reg  (chain: reg = reg*reg) */
        for (int j = 1; j < CHAIN_LEN; j++) {
            g_ops[idx].type = OP_MUL;
            g_ops[idx].rd   = reg;
            g_ops[idx].rs1  = reg;
            g_ops[idx].rs2  = reg;
            g_ops[idx].imm  = 0;
            g_ops[idx].id   = idx;
            idx++;
        }
    }

    /* sanity */
    if (idx != N_OPS) {
        fprintf(stderr, "BUG: generated %d ops, expected %d\n", idx, N_OPS);
        exit(1);
    }
}

/* ── Structural verification ──────────────────────────── */
static int verify_structure(const VliwBundle *bundles, int n_bundles, int n_ops) {
    /* Check each slot has correct type; count total ops emitted */
    int seen[N_OPS];
    memset(seen, 0, sizeof(seen));
    int count = 0;

    for (int b = 0; b < n_bundles; b++) {
        /* ALU slot */
        const VliwOp *a = &bundles[b].alu;
        if (a->type != OP_NOP && a->type != OP_ALU) return 0;
        if (a->type == OP_ALU) {
            if (a->id < 0 || a->id >= n_ops) return 0;
            if (seen[a->id]) return 0;
            seen[a->id] = 1; count++;
        }
        /* MUL slot */
        const VliwOp *m = &bundles[b].mul;
        if (m->type != OP_NOP && m->type != OP_MUL) return 0;
        if (m->type == OP_MUL) {
            if (m->id < 0 || m->id >= n_ops) return 0;
            if (seen[m->id]) return 0;
            seen[m->id] = 1; count++;
        }
        /* MEM slot */
        const VliwOp *e = &bundles[b].mem;
        if (e->type != OP_NOP && e->type != OP_MEM) return 0;
        if (e->type == OP_MEM) {
            if (e->id < 0 || e->id >= n_ops) return 0;
            if (seen[e->id]) return 0;
            seen[e->id] = 1; count++;
        }
    }
    return (count == n_ops);
}

/* ── Hazard verification ──────────────────────────────── */
static int latency_of(OpType t) {
    switch (t) { case OP_ALU: return 1; case OP_MUL: return 3; case OP_MEM: return 4; default: return 0; }
}

static int verify_hazards(const VliwBundle *bundles, int n_bundles) {
    /* written_at[r] = bundle index where r was last written; lat[r] = latency */
    int written_at[VLIW_REGS];
    int wlat[VLIW_REGS];
    for (int r = 0; r < VLIW_REGS; r++) { written_at[r] = -100; wlat[r] = 0; }

    for (int b = 0; b < n_bundles; b++) {
        const VliwOp *slots[3] = { &bundles[b].alu, &bundles[b].mul, &bundles[b].mem };
        /* Check reads */
        for (int s = 0; s < 3; s++) {
            if (slots[s]->type == OP_NOP) continue;
            int r1 = slots[s]->rs1, r2 = slots[s]->rs2;
            if (r1 > 0 && b < written_at[r1] + wlat[r1]) return 0;
            if (r2 > 0 && b < written_at[r2] + wlat[r2]) return 0;
        }
        /* Record writes */
        for (int s = 0; s < 3; s++) {
            if (slots[s]->type == OP_NOP) continue;
            int rd = slots[s]->rd;
            if (rd > 0) { written_at[rd] = b; wlat[rd] = latency_of(slots[s]->type); }
        }
    }
    return 1;
}

int main(void) {
    generate_program();

    /* Run scheduler */
    int n_bundles = vliw_schedule(g_ops, N_OPS, g_bundles);

    /* Structural check */
    if (!verify_structure(g_bundles, n_bundles, N_OPS)) {
        printf("n_ops=%d cycles=%d result=WRONG (structural)\n", N_OPS, n_bundles);
        return 0;
    }

    /* Hazard check */
    if (!verify_hazards(g_bundles, n_bundles)) {
        printf("n_ops=%d cycles=%d result=WRONG (hazard)\n", N_OPS, n_bundles);
        return 0;
    }

    /* Register-file simulation comparison */
    uint32_t ref_regs[VLIW_REGS], vliw_regs[VLIW_REGS];
    sim_sequential(g_ops, N_OPS, ref_regs);
    sim_vliw(g_bundles, n_bundles, vliw_regs);

    for (int r = 1; r < VLIW_REGS; r++) {
        if (ref_regs[r] != vliw_regs[r]) {
            printf("n_ops=%d cycles=%d result=WRONG (r%d: expected %u got %u)\n",
                   N_OPS, n_bundles, r, ref_regs[r], vliw_regs[r]);
            return 0;
        }
    }

    printf("n_ops=%d cycles=%d result=ok\n", N_OPS, n_bundles);
    return 0;
}
