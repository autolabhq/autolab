# Toy ISA Dot-Product Optimization — Reference

## Background
Instruction scheduling and pipeline hazard elimination are fundamental to extracting performance from in-order processors. The PINC (Performance-Instructive Numeric Chip) ISA is a teaching vehicle for the same scoreboard techniques used in classic RISC designs (MIPS R-series, early SPARC). Dot products are the inner kernel of BLAS SDOT/DDOT routines, neural network inference, and signal processing — making this a representative microbenchmark for instruction-level parallelism (ILP).

## Baseline Approach
The naive loop issues `ld A[i]` → `ld B[i]` → `mul` → `add` in strict sequential order. Because the scoreboard stalls the next instruction until all sources are ready, every iteration suffers the full 5-cycle load latency twice plus the 5-cycle multiply latency before the accumulator `add` can issue. The always-taken branch adds 2 flush cycles per iteration. With a single accumulator there is no opportunity to overlap iterations.

Baseline cost: ~9,220 cycles (~18 cycles/iteration × 512 + overhead).

## Possible Optimization Directions
1. **Loop unrolling** — processing 4+ elements per iteration reduces branch overhead (2-cycle flush × 512 → 2-cycle flush × 128) and exposes independent operations to the scheduler
2. **Multiple independent accumulators** — 4 separate accumulator registers break the single add→add→… dependency chain, letting the scoreboard issue accumulates for different elements without waiting on each other
3. **Instruction scheduling / latency hiding** — interleave loads for element N+1 between the multiply and accumulate of element N, filling stall slots with useful work rather than bubbles
4. **Use MAC instruction** — `mac rd, rs1, rs2` (rd += rs1*rs2, latency 6) can replace a `mul`+`add` pair, though careful scheduling is needed since `rd` is also a source operand
5. **Software pipelining** — overlap the load phase of iteration N+1 with the multiply/accumulate phase of iteration N to achieve near-full pipeline utilization

## Reference Solution
4× unrolled loop with 4 independent accumulators (`r1`, `r9`, `r10`, `r11`) and carefully interleaved loads and multiplies. Each iteration processes elements `i`, `i+1`, `i+2`, `i+3`:
- All 6 loads for elements 0–2 are issued upfront, filling the scoreboard while earlier loads complete.
- `mul` for element 0 issues once loads 0 are ready; loads for element 3 are then issued during that multiply's 5-cycle latency.
- Multiplies for elements 1 and 2 follow, with pointer/counter updates scheduled into remaining stall slots.
- Accumulates for all 4 elements use independent registers, avoiding any add→add RAW hazard.
- A 3-instruction final reduction merges the 4 accumulators after the loop.

## Source
- Tomasulo, R. M., *An Efficient Algorithm for Exploiting Multiple Arithmetic Units* (1967) — scoreboard / out-of-order issue foundations
- Hennessy, J. & Patterson, D., *Computer Architecture: A Quantitative Approach* — instruction scheduling, loop unrolling, software pipelining (all editions)
