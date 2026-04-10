#define _POSIX_C_SOURCE 200809L
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MEM_SIZE 2048
#define STACK_SIZE 256
#define MAX_PROG 2048
#define MAX_LABELS 256
#define MAX_LINE 256
#define EXEC_GUARD 1000000LL

enum Op {
    OP_PUSH, OP_DUP, OP_SWAP, OP_OVER, OP_DROP,
    OP_LD, OP_ST, OP_LDX,
    OP_ADD, OP_SUB, OP_MUL, OP_INC, OP_DEC, OP_EQZ,
    OP_JMP, OP_JZ, OP_HALT
};

typedef struct {
    int op;
    long long imm;
    int target;
} Inst;

static char label_names[MAX_LABELS][64];
static int label_pcs[MAX_LABELS];
static int num_labels = 0;

static void strip_comment(char *s) {
    char *p = strchr(s, ';');
    if (p) *p = '\0';
}

static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) --e;
    *e = '\0';
    return s;
}

static int is_blank(const char *s) { return *s == '\0'; }

static void add_label(const char *name, int pc) {
    if (num_labels >= MAX_LABELS) { fprintf(stderr, "too many labels\n"); exit(1); }
    strncpy(label_names[num_labels], name, 63);
    label_names[num_labels][63] = '\0';
    label_pcs[num_labels] = pc;
    num_labels++;
}

static int find_label(const char *name) {
    for (int i = 0; i < num_labels; i++) {
        if (strcmp(label_names[i], name) == 0) return label_pcs[i];
    }
    fprintf(stderr, "undefined label: %s\n", name);
    exit(1);
}

static int pass1(FILE *f) {
    char line[MAX_LINE];
    int pc = 0;
    while (fgets(line, sizeof(line), f)) {
        strip_comment(line);
        char *s = trim(line);
        if (is_blank(s)) continue;
        char *colon = strchr(s, ':');
        if (colon) {
            *colon = '\0';
            add_label(trim(s), pc);
            s = trim(colon + 1);
            if (!is_blank(s)) pc++;
        } else {
            pc++;
        }
    }
    return pc;
}

static int pass2(FILE *f, Inst *prog) {
    char line[MAX_LINE];
    int pc = 0;
    while (fgets(line, sizeof(line), f)) {
        strip_comment(line);
        char *s = trim(line);
        if (is_blank(s)) continue;
        char *colon = strchr(s, ':');
        if (colon) {
            s = trim(colon + 1);
            if (is_blank(s)) continue;
        }

        char buf[MAX_LINE];
        strncpy(buf, s, sizeof(buf)-1);
        buf[sizeof(buf)-1] = '\0';
        char *save = NULL;
        char *op = strtok_r(buf, " \t,", &save);
        if (!op) continue;
        for (char *p = op; *p; ++p) *p = (char)tolower((unsigned char)*p);

        Inst in = {0};
        in.imm = 0;
        in.target = -1;

        if (strcmp(op, "push") == 0) {
            in.op = OP_PUSH;
            char *tok = strtok_r(NULL, " \t,", &save);
            if (!tok) { fprintf(stderr, "push missing imm\n"); exit(1); }
            in.imm = strtoll(tok, NULL, 10);
        } else if (strcmp(op, "dup") == 0) in.op = OP_DUP;
        else if (strcmp(op, "swap") == 0) in.op = OP_SWAP;
        else if (strcmp(op, "over") == 0) in.op = OP_OVER;
        else if (strcmp(op, "drop") == 0) in.op = OP_DROP;
        else if (strcmp(op, "ld") == 0) {
            in.op = OP_LD;
            char *tok = strtok_r(NULL, " \t,", &save);
            in.imm = strtoll(tok, NULL, 10);
        } else if (strcmp(op, "st") == 0) {
            in.op = OP_ST;
            char *tok = strtok_r(NULL, " \t,", &save);
            in.imm = strtoll(tok, NULL, 10);
        } else if (strcmp(op, "ldx") == 0) in.op = OP_LDX;
        else if (strcmp(op, "add") == 0) in.op = OP_ADD;
        else if (strcmp(op, "sub") == 0) in.op = OP_SUB;
        else if (strcmp(op, "mul") == 0) in.op = OP_MUL;
        else if (strcmp(op, "inc") == 0) in.op = OP_INC;
        else if (strcmp(op, "dec") == 0) in.op = OP_DEC;
        else if (strcmp(op, "eqz") == 0) in.op = OP_EQZ;
        else if (strcmp(op, "jmp") == 0) {
            in.op = OP_JMP;
            char *tok = strtok_r(NULL, " \t,", &save);
            in.target = find_label(tok);
        } else if (strcmp(op, "jz") == 0) {
            in.op = OP_JZ;
            char *tok = strtok_r(NULL, " \t,", &save);
            in.target = find_label(tok);
        } else if (strcmp(op, "halt") == 0) in.op = OP_HALT;
        else {
            fprintf(stderr, "unknown op: %s\n", op);
            exit(1);
        }

        prog[pc++] = in;
    }
    return pc;
}

static unsigned int g_seed = 0;

static void generate_data(long long *mem) {
    for (int i = 0; i < 256; i++) {
        mem[i]       = ((long long)i * 12345 + 6789 + g_seed) % 997;
        mem[256 + i] = ((long long)i * 54321 + 9876 + g_seed * 3) % 997;
    }
}

static long long reference_dot(long long *mem) {
    long long acc = 0;
    for (int i = 0; i < 256; i++)
        acc += mem[i] * mem[256 + i];
    return acc;
}

static long long pop_value(long long *stack, int *sp) {
    if (*sp <= 0) {
        fprintf(stderr, "stack underflow\n");
        exit(1);
    }
    return stack[--(*sp)];
}

static void push_value(long long *stack, int *sp, long long x) {
    if (*sp >= STACK_SIZE) {
        fprintf(stderr, "stack overflow\n");
        exit(1);
    }
    stack[(*sp)++] = x;
}

int main(int argc, char **argv) {
    if (argc > 1) g_seed = (unsigned int)atoi(argv[1]);

    FILE *f = fopen("program.stk", "r");
    if (!f) { perror("program.stk"); return 1; }
    Inst prog[MAX_PROG];
    int prog_len = pass1(f);
    rewind(f);
    prog_len = pass2(f, prog);
    fclose(f);

    long long mem[MEM_SIZE] = {0};
    generate_data(mem);

    long long stack[STACK_SIZE];
    int sp = 0;
    int pc = 0;
    long long executed = 0;

    while (pc >= 0 && pc < prog_len) {
        if (++executed > EXEC_GUARD) {
            printf("instructions=%lld verify=FAIL reason=guard\n", executed);
            return 0;
        }
        Inst in = prog[pc];
        switch (in.op) {
            case OP_PUSH: push_value(stack, &sp, in.imm); pc++; break;
            case OP_DUP: { long long x = pop_value(stack, &sp); push_value(stack, &sp, x); push_value(stack, &sp, x); pc++; break; }
            case OP_SWAP: { long long b = pop_value(stack, &sp), a = pop_value(stack, &sp); push_value(stack, &sp, b); push_value(stack, &sp, a); pc++; break; }
            case OP_OVER: {
                if (sp < 2) { fprintf(stderr, "stack underflow\n"); return 1; }
                push_value(stack, &sp, stack[sp - 2]);
                pc++;
                break;
            }
            case OP_DROP: { (void)pop_value(stack, &sp); pc++; break; }
            case OP_LD: {
                if (in.imm < 0 || in.imm >= MEM_SIZE) { fprintf(stderr, "bad ld\n"); return 1; }
                push_value(stack, &sp, mem[in.imm]); pc++; break;
            }
            case OP_ST: {
                long long v = pop_value(stack, &sp);
                if (in.imm < 0 || in.imm >= MEM_SIZE) { fprintf(stderr, "bad st\n"); return 1; }
                mem[in.imm] = v; pc++; break;
            }
            case OP_LDX: {
                long long addr = pop_value(stack, &sp);
                if (addr < 0 || addr >= MEM_SIZE) { fprintf(stderr, "bad ldx\n"); return 1; }
                push_value(stack, &sp, mem[addr]); pc++; break;
            }
            case OP_ADD: { long long b = pop_value(stack, &sp), a = pop_value(stack, &sp); push_value(stack, &sp, a + b); pc++; break; }
            case OP_SUB: { long long b = pop_value(stack, &sp), a = pop_value(stack, &sp); push_value(stack, &sp, a - b); pc++; break; }
            case OP_MUL: { long long b = pop_value(stack, &sp), a = pop_value(stack, &sp); push_value(stack, &sp, a * b); pc++; break; }
            case OP_INC: { long long a = pop_value(stack, &sp); push_value(stack, &sp, a + 1); pc++; break; }
            case OP_DEC: { long long a = pop_value(stack, &sp); push_value(stack, &sp, a - 1); pc++; break; }
            case OP_EQZ: { long long a = pop_value(stack, &sp); push_value(stack, &sp, a == 0 ? 1 : 0); pc++; break; }
            case OP_JMP: pc = in.target; break;
            case OP_JZ: { long long a = pop_value(stack, &sp); pc = (a == 0) ? in.target : pc + 1; break; }
            case OP_HALT: goto done;
            default: fprintf(stderr, "bad op\n"); return 1;
        }
    }

done:
    {
        long long got = (sp > 0) ? stack[sp - 1] : 0;
        long long ref = reference_dot(mem);
        printf("instructions=%lld verify=%s result=%lld expected=%lld\n",
               executed, (got == ref ? "ok" : "FAIL"), got, ref);
    }
    return 0;
}
