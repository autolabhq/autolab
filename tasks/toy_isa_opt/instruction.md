# Toy ISA Optimization (PINC)

Rewrite `program.s` to minimize the **simulated cycle count** for computing a 512-element integer dot product on the PINC scoreboard pipeline.

## Problem

Compute `sum = A[0]*B[0] + ... + A[511]*B[511]` and store the result in `r1` at `halt`.

Memory layout (word-addressed): `A[0..511]` at addresses 0–511, `B[0..511]` at 512–1023.

Data: `A[i] = (i*12345+6789)%997`, `B[i] = (i*54321+9876)%997`.

## ISA

| Instruction | Operation | Latency |
|-------------|-----------|---------|
| `add rd, rs1, rs2` | `rd = rs1 + rs2` | 1 |
| `mul rd, rs1, rs2` | `rd = rs1 * rs2` | 5 |
| `mac rd, rs1, rs2` | `rd += rs1 * rs2` | 6 |
| `addi rd, rs1, #imm` | `rd = rs1 + imm` | 1 |
| `ld rd, #imm(rs1)` | `rd = mem[rs1+imm]` | 5 |
| `st rs2, #imm(rs1)` | `mem[rs1+imm] = rs2` | 1 |
| `beq/bne/blt/bge` | branch | 1 (+2 if taken) |

The pipeline issues one instruction per cycle and stalls until all source registers are ready. `r0` is always zero.

Baseline: **~9,220 cycles**.

## Files

| Path | Permission |
|------|-----------|
| `/app/program.s` | **Edit** |
| `/app/main.c` | Read-only |
| `/app/Makefile` | Read-only |

## Rules

- Do not modify `Makefile`. Build configuration is fixed.
- Edit `/app/program.s` only.
- `r1` must equal the reference dot product at `halt` — wrong answer scores 0.
