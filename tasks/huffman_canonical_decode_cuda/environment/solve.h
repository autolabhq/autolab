#pragma once
//
// solve.h -- Public interface and shared constants for the canonical Huffman
//             decode CUDA task.  DO NOT MODIFY.
//
// All numeric scoring constants live in task.toml; this header only carries
// the workload shape so main.cu and solve.cu agree on buffer layouts.
//

#include <cstdint>
#include <cuda_runtime.h>

// ---- workload shape ----------------------------------------------------
// Number of independent encoded bitstreams in the full benchmark.
#define HUFF_NUM_STREAMS_BENCH    2048

// Decoded payload bytes per stream in the full benchmark.
#define HUFF_BYTES_PER_STREAM     65536

// Number of streams in the small public --verify check.
#define HUFF_NUM_STREAMS_VERIFY   8

// Decoded payload bytes per stream in the small public --verify check.
#define HUFF_BYTES_PER_STREAM_VERIFY  256

// Alphabet is 0..255 (one byte per symbol).  Code-length array is
// uint8[256]; symbol order (canonical sorted-by-(length,symbol)) is
// uint16[256].
#define HUFF_ALPHABET_SIZE        256

// Max canonical code length in bits.  All code lengths in the table are
// in [1, HUFF_MAX_CODE_LEN].
#define HUFF_MAX_CODE_LEN         16

// Worst-case packed-bits buffer per stream, in 32-bit words.  The encoder
// guarantees encoded_bits <= HUFF_BYTES_PER_STREAM * HUFF_MAX_CODE_LEN,
// so this is enough head-room.  We round up to a multiple of 32 bits.
#define HUFF_MAX_WORDS_PER_STREAM \
    ((HUFF_BYTES_PER_STREAM * HUFF_MAX_CODE_LEN + 31) / 32)

// ---- decode entry point ------------------------------------------------
// Decode K independent canonical-Huffman bitstreams in parallel back to
// byte streams.
//
// Parameters (all device-resident unless noted):
//   d_bits            packed bitstreams, K * bits_stride uint32 words.
//                     Stream k starts at d_bits + k * bits_stride and
//                     uses d_bit_lens[k] bits there.  Bit ordering: bit i
//                     of stream k lives at word_index = i / 32 and
//                     bit_pos_in_word = i % 32 where bit 0 is the LSB
//                     within the word.  Canonical Huffman codes were
//                     written MSB-first into the stream (the first bit
//                     emitted is the MSB of the code; it ends up at the
//                     lowest bit-index).
//   d_bit_lens        per-stream encoded length in bits, K entries.
//   d_out             output buffer, K * out_stride bytes.  Stream k
//                     writes its decoded payload to d_out + k * out_stride
//                     and produces exactly d_out_lens[k] bytes there.
//   d_out_lens        per-stream decoded length in bytes, K entries.
//   out_stride        byte stride between successive streams in d_out.
//   bits_stride       uint32-word stride between successive streams in
//                     d_bits.
//   d_code_len        uint8[256], canonical code length for each symbol;
//                     0 means "symbol not in alphabet" (must not occur in
//                     the encoded streams).
//   d_symbols         uint16[256], symbols sorted in canonical order by
//                     (code_length, symbol_id) -- this is the canonical
//                     decoding order.  Index i is the i-th symbol in that
//                     ordering.
//   K                 number of streams.
//   stream            CUDA stream to launch on (0 = default stream).
//
// On entry d_out is uninitialised.  On exit d_out[k * out_stride + j] for
// j in [0, d_out_lens[k]) holds the decoded byte; bytes beyond
// d_out_lens[k] need not be touched.
void huffman_decode(
    const uint32_t* d_bits,
    const uint32_t* d_bit_lens,
    uint8_t*        d_out,
    const uint32_t* d_out_lens,
    uint32_t        out_stride,
    uint32_t        bits_stride,
    const uint8_t*  d_code_len,
    const uint16_t* d_symbols,
    uint32_t        K,
    cudaStream_t    stream);
