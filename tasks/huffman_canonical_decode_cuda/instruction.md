# Canonical Huffman Decode -- CUDA

Optimize `huffman_decode()` in `/app/solve.cu` to decode `K` independent canonical-Huffman bitstreams back to their byte payloads as fast as possible on a single H100.

```cpp
void huffman_decode(
    const uint32_t* d_bits,        // K * bits_stride uint32 packed bits
    const uint32_t* d_bit_lens,    // K entries, encoded length in bits
    uint8_t*        d_out,         // K * out_stride bytes
    const uint32_t* d_out_lens,    // K entries, decoded length in bytes
    uint32_t        out_stride,    // byte stride between successive streams
    uint32_t        bits_stride,   // uint32 stride between successive streams
    const uint8_t*  d_code_len,    // 256 entries
    const uint16_t* d_symbols,     // 256 entries (canonical order)
    uint32_t        K,
    cudaStream_t    stream);
```

All pointers are device memory. Stream `k`'s encoded bits start at `d_bits + k * bits_stride` (length `d_bit_lens[k]` bits) and its decoded bytes go to `d_out + k * out_stride` (length `d_out_lens[k]` bytes). The Huffman table is shared across streams: `d_code_len[s]` is the canonical code length for symbol `s` (length `0` means the symbol is absent and must not appear in any stream), and `d_symbols[i]` is the symbol at canonical-order rank `i` (i.e. the `i`-th symbol when sorted by `(code_length, symbol_id)`). Maximum code length is `HUFF_MAX_CODE_LEN = 16` bits.

Codes are packed **MSB-first** into the buffer at the lowest free bit position: writing a 3-bit code `0b110` first into a fresh stream lands its MSB at bit 0, next bit at bit 1, last bit at bit 2 (where bit `i` is at word `i / 32`, bit position `i % 32`, with bit `0` being the LSB of its word).

## Setup

| Item | Path / Value |
|------|--------------|
| Editable | `/app/solve.cu` |
| Function signature | `/app/solve.h` (read-only) |
| Harness | `/app/main.cu` (read-only) |
| Build | `/app/Makefile` (read-only) |
| GPU | Single H100 |
| Local build / run | `make -C /app && /app/<binary>` |

## Your Goal

Minimize the median wall-clock runtime of `huffman_decode()` on the timed workload. Decoded bytes must match the encoder's original payload exactly for every stream. Any mismatch scores 0.

## Evaluation

The evaluation measures median wall-clock time of `huffman_decode()` over several trials via CUDA events after a warmup. Lower is better.

## Rules

- Edit `/app/solve.cu` only. You may create scratch files while working, but when verification runs, `/app` must contain only the original files plus your edits to `solve.cu`. Any extra file under `/app` causes the run to score 0.
- CUDA and the C++ standard library only. No external libraries (Thrust, CUB, cuBLAS, cuSPARSE, CUTLASS, nvCOMP, zlib, zstd, cuDNN, etc.).
- Single GPU, no host-side decode, no internet, no subprocesses.
- Do NOT branch on `K`, on the seed, on payload size, or on any other property of the workload that is hidden from your kernel. The verifier rotates the random seed on every scored run, so memorising input/output is not a viable strategy.
- Time budget: 2 hours.
