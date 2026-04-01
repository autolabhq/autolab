/*
 * main.c — PINC (Performance-Instructive Numeric Chip) assembler + simulator + harness
 *
 * This file is READ-ONLY for agents. Edit program.s to optimize the dot-product kernel.
 *
 * Usage: ./solve
 * Output: "cycles=<N> verify=ok|FAIL"
 *
 * DO NOT MODIFY THIS FILE.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ── ISA constants ────────────────────────────────────────────────────────── */
#define NUM_REGS   16
#define MEM_SIZE   4096
#define MAX_PROG   8192
#define MAX_LABELS 256
#define MAX_LINE   256
#define CYCLE_GUARD 1000000

/* Opcodes */
enum Op {
    OP_ADD, OP_SUB, OP_MUL, OP_MAC,
    OP_ADDI,
    OP_LD, OP_ST,
    OP_NOP, OP_HALT,
    OP_BEQ, OP_BNE, OP_BLT, OP_BGE
};

/* Latencies */
static const int LATENCY[] = {
    [OP_ADD]  = 1,
    [OP_SUB]  = 1,
    [OP_MUL]  = 5,
    [OP_MAC]  = 6,
    [OP_ADDI] = 1,
    [OP_LD]   = 5,
    [OP_ST]   = 1,
    [OP_NOP]  = 1,
    [OP_HALT] = 1,
    [OP_BEQ]  = 1,
    [OP_BNE]  = 1,
    [OP_BLT]  = 1,
    [OP_BGE]  = 1,
};

typedef struct {
    int op;
    int rd, rs1, rs2;   /* -1 = not used */
    long long imm;
    int target_pc;      /* resolved branch/jump target (-1 if not branch) */
} Instruction;

/* ── CPU state ─────────────────────────────────────────────────────────────── */
typedef struct {
    long long regs[NUM_REGS];
    long long mem[MEM_SIZE];
    long long ready_at[NUM_REGS];  /* cycle when register becomes available */
    long long cycle;
} CPU;

/* ── Label table ────────────────────────────────────────────────────────────── */
static char  label_names[MAX_LABELS][64];
static int   label_pcs[MAX_LABELS];
static int   num_labels = 0;

static void add_label(const char *name, int pc) {
    if (num_labels >= MAX_LABELS) { fprintf(stderr, "too many labels\n"); exit(1); }
    strncpy(label_names[num_labels], name, 63);
    label_names[num_labels][63] = '\0';
    label_pcs[num_labels] = pc;
    num_labels++;
}

static int find_label(const char *name) {
    for (int i = 0; i < num_labels; i++)
        if (strcmp(label_names[i], name) == 0) return label_pcs[i];
    fprintf(stderr, "undefined label: %s\n", name);
    exit(1);
}

/* ── Helpers ────────────────────────────────────────────────────────────────── */
static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s) - 1;
    while (e >= s && isspace((unsigned char)*e)) *e-- = '\0';
    return s;
}

/* Strip inline comment (semicolon) */
static void strip_comment(char *s) {
    /* Watch out for semicolons that are actually part of nothing – just cut at ; */
    char *p = s;
    int in_str = 0;
    while (*p) {
        if (*p == ';' && !in_str) { *p = '\0'; break; }
        p++;
    }
}

/* Returns 1 if line is blank or comment-only after trimming */
static int is_blank(const char *s) {
    while (isspace((unsigned char)*s)) s++;
    return (*s == '\0');
}

/* Parse register name "rN" → N, or -1 on failure */
static int parse_reg(const char *tok) {
    if (!tok) return -1;
    /* skip optional leading whitespace */
    while (isspace((unsigned char)*tok)) tok++;
    if (tok[0] != 'r' && tok[0] != 'R') return -1;
    char *end;
    long v = strtol(tok + 1, &end, 10);
    if (end == tok + 1) return -1;
    if (v < 0 || v >= NUM_REGS) { fprintf(stderr, "invalid register: %s\n", tok); exit(1); }
    return (int)v;
}

/* Parse immediate: optional '#' prefix, then integer */
static long long parse_imm(const char *tok) {
    while (isspace((unsigned char)*tok)) tok++;
    if (*tok == '#') tok++;
    char *end;
    long long v = strtoll(tok, &end, 10);
    if (end == tok) { fprintf(stderr, "invalid immediate: %s\n", tok); exit(1); }
    return v;
}

/* Parse "imm(rs)" or "#imm(rs)" memory operand → fills *imm and *rs */
static void parse_mem(const char *tok, long long *imm, int *rs) {
    while (isspace((unsigned char)*tok)) tok++;
    if (*tok == '#') tok++;
    char *paren = strchr(tok, '(');
    if (!paren) { fprintf(stderr, "invalid memory operand: %s\n", tok); exit(1); }
    char numbuf[32];
    int len = (int)(paren - tok);
    if (len <= 0) {
        *imm = 0;
    } else {
        strncpy(numbuf, tok, len < 31 ? len : 31);
        numbuf[len < 31 ? len : 31] = '\0';
        char *end;
        *imm = strtoll(numbuf, &end, 10);
    }
    char regbuf[16];
    char *close = strchr(paren, ')');
    if (!close) { fprintf(stderr, "missing ')' in memory operand: %s\n", tok); exit(1); }
    int rlen = (int)(close - paren - 1);
    strncpy(regbuf, paren + 1, rlen < 15 ? rlen : 15);
    regbuf[rlen < 15 ? rlen : 15] = '\0';
    *rs = parse_reg(regbuf);
    if (*rs < 0) { fprintf(stderr, "invalid base register in memory operand: %s\n", tok); exit(1); }
}

/* ── Assembler: pass 1 — collect labels ─────────────────────────────────────── */
static int pass1(FILE *f) {
    char line[MAX_LINE];
    int pc = 0;
    while (fgets(line, sizeof(line), f)) {
        strip_comment(line);
        char *s = trim(line);
        if (is_blank(s)) continue;

        /* Check if line starts with a label (ends with ':') */
        char *colon = strchr(s, ':');
        if (colon) {
            /* label definition */
            *colon = '\0';
            char *lname = trim(s);
            add_label(lname, pc);
            /* remainder after colon might have an instruction */
            char *rest = trim(colon + 1);
            if (!is_blank(rest)) pc++;
        } else {
            pc++;
        }
    }
    return pc;
}

/* ── Assembler: pass 2 — parse instructions ──────────────────────────────────── */
static int pass2(FILE *f, Instruction *prog) {
    char line[MAX_LINE];
    int pc = 0;

    while (fgets(line, sizeof(line), f)) {
        strip_comment(line);
        char *s = trim(line);
        if (is_blank(s)) continue;

        /* Strip label prefix if present */
        char *colon = strchr(s, ':');
        if (colon) {
            s = trim(colon + 1);
            if (is_blank(s)) continue;
        }

        /* Tokenize mnemonic */
        char *saveptr = NULL;
        char buf[MAX_LINE];
        strncpy(buf, s, MAX_LINE - 1);
        buf[MAX_LINE - 1] = '\0';

        char *mnem = strtok_r(buf, " \t,", &saveptr);
        if (!mnem) continue;

        /* Convert to lowercase for comparison */
        for (char *p = mnem; *p; p++) *p = (char)tolower((unsigned char)*p);

        Instruction ins;
        memset(&ins, 0, sizeof(ins));
        ins.rd = ins.rs1 = ins.rs2 = -1;
        ins.target_pc = -1;

#define TOK() strtok_r(NULL, " \t,", &saveptr)

        if (strcmp(mnem, "nop") == 0) {
            ins.op = OP_NOP;

        } else if (strcmp(mnem, "halt") == 0) {
            ins.op = OP_HALT;

        } else if (strcmp(mnem, "add") == 0) {
            ins.op  = OP_ADD;
            ins.rd  = parse_reg(TOK());
            ins.rs1 = parse_reg(TOK());
            ins.rs2 = parse_reg(TOK());

        } else if (strcmp(mnem, "sub") == 0) {
            ins.op  = OP_SUB;
            ins.rd  = parse_reg(TOK());
            ins.rs1 = parse_reg(TOK());
            ins.rs2 = parse_reg(TOK());

        } else if (strcmp(mnem, "mul") == 0) {
            ins.op  = OP_MUL;
            ins.rd  = parse_reg(TOK());
            ins.rs1 = parse_reg(TOK());
            ins.rs2 = parse_reg(TOK());

        } else if (strcmp(mnem, "mac") == 0) {
            /* mac rd, rs1, rs2  →  rd += rs1 * rs2 (rd is both src and dst) */
            ins.op  = OP_MAC;
            ins.rd  = parse_reg(TOK());
            ins.rs1 = parse_reg(TOK());
            ins.rs2 = parse_reg(TOK());

        } else if (strcmp(mnem, "addi") == 0) {
            ins.op  = OP_ADDI;
            ins.rd  = parse_reg(TOK());
            ins.rs1 = parse_reg(TOK());
            char *itok = TOK();
            ins.imm = parse_imm(itok);

        } else if (strcmp(mnem, "ld") == 0) {
            /* ld rd, imm(rs1) */
            ins.op  = OP_LD;
            ins.rd  = parse_reg(TOK());
            char *mtok = TOK();
            parse_mem(mtok, &ins.imm, &ins.rs1);

        } else if (strcmp(mnem, "st") == 0) {
            /* st rs2, imm(rs1)  — value reg first, then base */
            ins.op  = OP_ST;
            ins.rs2 = parse_reg(TOK());   /* value being stored */
            char *mtok = TOK();
            parse_mem(mtok, &ins.imm, &ins.rs1);

        } else if (strcmp(mnem, "beq") == 0 || strcmp(mnem, "bne") == 0 ||
                   strcmp(mnem, "blt") == 0 || strcmp(mnem, "bge") == 0) {
            if      (strcmp(mnem, "beq") == 0) ins.op = OP_BEQ;
            else if (strcmp(mnem, "bne") == 0) ins.op = OP_BNE;
            else if (strcmp(mnem, "blt") == 0) ins.op = OP_BLT;
            else                               ins.op = OP_BGE;
            ins.rs1 = parse_reg(TOK());
            ins.rs2 = parse_reg(TOK());
            char *ltok = TOK();
            if (!ltok) { fprintf(stderr, "missing branch target\n"); exit(1); }
            while (isspace((unsigned char)*ltok)) ltok++;
            ins.target_pc = find_label(ltok);

        } else {
            fprintf(stderr, "unknown mnemonic: %s\n", mnem);
            exit(1);
        }

#undef TOK

        prog[pc++] = ins;
    }
    return pc;
}

/* ── Simulator ──────────────────────────────────────────────────────────────── */
static long long simulate(CPU *cpu, Instruction *prog, int prog_len) {
    int pc = 0;
    cpu->cycle = 0;
    for (int r = 0; r < NUM_REGS; r++) cpu->ready_at[r] = 0;

    while (pc < prog_len) {
        Instruction *ins = &prog[pc];

        if (ins->op == OP_HALT) {
            cpu->cycle++;
            break;
        }

        /* Compute the earliest cycle this instruction can issue */
        long long issue = cpu->cycle;

        /* Stall for source register availability */
        if (ins->rs1 >= 0 && cpu->ready_at[ins->rs1] > issue)
            issue = cpu->ready_at[ins->rs1];
        if (ins->rs2 >= 0 && cpu->ready_at[ins->rs2] > issue)
            issue = cpu->ready_at[ins->rs2];
        /* MAC also reads rd (accumulate into it) */
        if (ins->op == OP_MAC && ins->rd >= 0 && cpu->ready_at[ins->rd] > issue)
            issue = cpu->ready_at[ins->rd];

        /* Advance cycle to issue cycle */
        cpu->cycle = issue;

        /* Execute */
        long long rs1_val = (ins->rs1 >= 0) ? cpu->regs[ins->rs1] : 0LL;
        long long rs2_val = (ins->rs2 >= 0) ? cpu->regs[ins->rs2] : 0LL;
        long long result  = 0;
        int branch_taken  = 0;

        switch (ins->op) {
            case OP_ADD:  result = rs1_val + rs2_val;             break;
            case OP_SUB:  result = rs1_val - rs2_val;             break;
            case OP_MUL:  result = rs1_val * rs2_val;             break;
            case OP_MAC:  result = cpu->regs[ins->rd] + rs1_val * rs2_val; break;
            case OP_ADDI: result = rs1_val + ins->imm;            break;
            case OP_LD: {
                long long addr = rs1_val + ins->imm;
                if (addr < 0 || addr >= MEM_SIZE) {
                    fprintf(stderr, "memory out of bounds: %lld\n", addr);
                    exit(1);
                }
                result = cpu->mem[addr];
                break;
            }
            case OP_ST: {
                long long addr = rs1_val + ins->imm;
                if (addr < 0 || addr >= MEM_SIZE) {
                    fprintf(stderr, "memory out of bounds: %lld\n", addr);
                    exit(1);
                }
                cpu->mem[addr] = cpu->regs[ins->rs2];
                break;
            }
            case OP_BEQ: branch_taken = (rs1_val == rs2_val); break;
            case OP_BNE: branch_taken = (rs1_val != rs2_val); break;
            case OP_BLT: branch_taken = (rs1_val <  rs2_val); break;
            case OP_BGE: branch_taken = (rs1_val >= rs2_val); break;
            case OP_NOP: break;
            default: break;
        }

        /* Write-back (skip r0, skip ST/branch/nop) */
        if (ins->rd >= 1 &&
            ins->op != OP_ST && ins->op != OP_BEQ && ins->op != OP_BNE &&
            ins->op != OP_BLT && ins->op != OP_BGE && ins->op != OP_NOP) {
            cpu->regs[ins->rd] = result;
            cpu->ready_at[ins->rd] = cpu->cycle + LATENCY[ins->op];
        }

        /* Advance PC */
        if (branch_taken) {
            pc = ins->target_pc;
            cpu->cycle += 3;   /* 1 for this insn + 2 flush penalty */
        } else {
            pc++;
            cpu->cycle++;
        }

        /* Infinite-loop guard */
        if (cpu->cycle > CYCLE_GUARD) {
            puts("verify=FAIL cycles=999999");
            exit(0);
        }
    }

    return cpu->cycle;
}

/* ── Main ───────────────────────────────────────────────────────────────────── */
int main(void) {
    /* ── Load and assemble program.s ── */
    FILE *f = fopen("program.s", "r");
    if (!f) { perror("program.s"); return 1; }

    int prog_len = pass1(f);
    if (prog_len > MAX_PROG) { fprintf(stderr, "program too large\n"); return 1; }

    Instruction *prog = calloc(prog_len + 1, sizeof(Instruction));
    if (!prog) { perror("calloc"); return 1; }

    rewind(f);
    num_labels = 0;   /* reset for pass2 (pass1 already populated, but pass2 re-resolves) */
    /* Re-run pass1 to repopulate labels (pass2 needs them for branch resolution) */
    rewind(f);
    pass1(f);
    rewind(f);
    int actual_len = pass2(f, prog);
    fclose(f);

    if (actual_len != prog_len) {
        fprintf(stderr, "pass mismatch: pass1=%d pass2=%d\n", prog_len, actual_len);
        /* Use actual_len, which is the ground truth */
        prog_len = actual_len;
    }

    /* ── Initialize CPU and memory ── */
    CPU cpu;
    memset(&cpu, 0, sizeof(cpu));

    /* Deterministic data: A[0..511] at mem[0..511], B[0..511] at mem[512..1023] */
    for (int i = 0; i < 512; i++) {
        cpu.mem[i]       = (long long)((i * 12345LL + 6789) % 997);   /* A[i] */
        cpu.mem[512 + i] = (long long)((i * 54321LL + 9876) % 997);   /* B[i] */
    }

    /* ── Compute reference dot product in C ── */
    long long ref_sum = 0;
    for (int i = 0; i < 512; i++)
        ref_sum += cpu.mem[i] * cpu.mem[512 + i];

    /* ── Run simulator ── */
    long long cycles = simulate(&cpu, prog, prog_len);

    free(prog);

    /* ── Verify result ── */
    long long got = cpu.regs[1];   /* result must be in r1 */
    if (got != ref_sum) {
        printf("cycles=%lld verify=FAIL (got=%lld expected=%lld)\n",
               cycles, got, ref_sum);
        return 0;
    }

    printf("cycles=%lld verify=ok\n", cycles);
    return 0;
}
