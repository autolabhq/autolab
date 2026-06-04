# Canonical Huffman Decode -- Reference

## Background

Canonical Huffman decoding is bit-serial within a single stream: the
length of the current code determines where the next code starts, so
the decoder cannot trivially split work along a single bitstream. The
GPU mitigations are (a) decoding many independent streams concurrently
and (b) replacing the per-bit walk with a wide lookup table so most
symbols cost a single shared-memory read.

## Baseline Approach

`environment/solve.cu` runs one CUDA thread per stream. Each thread
reconstructs the canonical decode arrays in registers and walks the bit
stream one bit at a time, checking after every appended bit whether a
code of the current length has just completed. There is no LUT, no
shared memory, no warp cooperation, and adjacent warp lanes touch
unrelated streams so per-bit loads are uncoalesced. Cost is dominated
by the bit-by-bit accumulate loop and scattered global-memory reads.

## Possible Optimization Directions

1. **Wide primary LUT.** A 10-bit LUT collapses any code of length
   <= 10 bits into a single shared-memory read returning
   `(symbol, length)`. On natural-byte payloads the vast majority of
   codes are <= 10 bits, so almost every symbol costs one load.
2. **Secondary LUT for long codes.** A 16-bit fallback table covers
   codes of length 11..16 with a single load when the primary LUT
   misses. Only built / consulted when the canonical table actually
   contains long codes.
3. **Shared-memory canonical-table reconstruction.** Build
   `bl_count`, `first_code`, `first_idx` and the LUTs once in a setup
   kernel using a single block of cooperating threads, then cache the
   result across calls.
4. **Warp-per-stream layout with rolling 64-bit bit register.** Lane 0
   of each warp owns the symbol loop; the bit register is refilled
   32 bits at a time from coalesced uint32 loads.
5. **Unrolled multi-symbol inner loop with packed stores.** Decode a
   small batch of symbols per iteration and pack the output bytes into
   uint32 stores so the output path is a small number of 4-byte writes
   rather than per-byte writes.

## Reference Solution

`solution/solve_optimized.cu` is the reference oracle adapted from an
agent run. It combines all five directions above:

- A single-block setup kernel cooperatively rebuilds the canonical
  table in shared memory, fills a 1024-entry 10-bit primary LUT
  (`sym`, `len` per entry), and -- only when the table actually has
  codes longer than 10 bits -- also fills a 64Ki-entry 16-bit fallback
  LUT.
- The decode launches one warp per stream, 16 warps per block, with
  the 10-bit LUT staged into block shared memory so the hot path is
  one SMEM read per symbol.
- Lane 0 of each warp maintains a 64-bit rolling buffer refilled 32
  bits at a time, and the inner loop decodes 12 symbols per iteration,
  emitting them as three uint32 writes.
- A fast kernel is used when `max_code_length <= 10`; otherwise the
  fallback kernel handles long codes through the 16-bit LUT.
- The LUT build is cached and only re-run when the canonical table's
  device pointer changes across calls, so repeated benchmark
  iterations pay the build cost once.
