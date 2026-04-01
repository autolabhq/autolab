# Stack Machine Golf

Rewrite `program.stk` to minimize the executed instruction count on a pure stack machine while computing a 256-element integer dot product.

## Problem

Compute:

`sum = A[0]*B[0] + A[1]*B[1] + ... + A[255]*B[255]`

Memory layout:

- `A[0..255]` at addresses `0..255`
- `B[0..255]` at addresses `256..511`

Data is deterministic:

- `A[i] = (i*12345 + 6789) % 997`
- `B[i] = (i*54321 + 9876) % 997`

At `halt`, the top of the stack must equal the reference dot product.

## VM

This is a **pure stack machine** with no registers.

### Stack / Memory Ops

| Instruction | Meaning |
|-------------|---------|
| `push N` | push immediate integer `N` |
| `dup` | duplicate top of stack |
| `swap` | swap top two |
| `over` | duplicate second-from-top |
| `drop` | discard top |
| `ld A` | push `mem[A]` |
| `st A` | pop value and store to `mem[A]` |
| `ldx` | pop address, push `mem[address]` |

### Arithmetic

| Instruction | Meaning |
|-------------|---------|
| `add` | pop `b`, `a`; push `a+b` |
| `sub` | pop `b`, `a`; push `a-b` |
| `mul` | pop `b`, `a`; push `a*b` |
| `inc` | increment top |
| `dec` | decrement top |
| `eqz` | replace top with `1` if zero else `0` |

### Control Flow

| Instruction | Meaning |
|-------------|---------|
| `jmp label` | unconditional jump |
| `jz label` | pop top; jump if it is zero |
| `halt` | stop execution |

Labels end with `:`.

## Files

| File | Role |
|------|------|
| `program.stk` | **Edit this file only** |
| `main.c` | Read-only VM simulator |
| `Makefile` | Read-only build file |

## Rules

- Edit `/app/program.stk` only.
- You may use scratch memory at addresses `1024..1039`.
- Wrong final result scores 0.
- Infinite loops or stack underflow score 0.
