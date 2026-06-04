/*
 * solve.cu -- canonical Huffman decode on CUDA.
 *
 * One thread per stream: reconstructs the canonical decode arrays
 * (bl_count, first_code, first_idx) from per-symbol code lengths,
 * then walks the bitstream and emits one decoded byte per match.
 *
 * Edit this file to optimize.
 */

#include "solve.h"

#define BLOCK_SIZE 64

__global__ void huffman_decode_baseline_kernel(
    const uint32_t* __restrict__ bits_buf,
    const uint32_t* __restrict__ bit_lens,
    uint8_t*        __restrict__ out_buf,
    const uint32_t* __restrict__ out_lens,
    uint32_t        out_stride,
    uint32_t        bits_stride,
    const uint8_t*  __restrict__ code_len_g,
    const uint16_t* __restrict__ symbols_g,
    uint32_t        K)
{
    const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    // Reconstruct bl_count, first_code, first_idx in local memory.
    int bl_count[HUFF_MAX_CODE_LEN + 2];
    #pragma unroll
    for (int i = 0; i <= HUFF_MAX_CODE_LEN + 1; ++i) bl_count[i] = 0;
    for (int s = 0; s < HUFF_ALPHABET_SIZE; ++s) {
        bl_count[code_len_g[s]]++;
    }
    bl_count[0] = 0;

    uint32_t first_code[HUFF_MAX_CODE_LEN + 2];
    int      first_idx [HUFF_MAX_CODE_LEN + 2];
    #pragma unroll
    for (int l = 0; l <= HUFF_MAX_CODE_LEN + 1; ++l) {
        first_code[l] = 0;
        first_idx[l]  = 0;
    }
    {
        uint32_t code = 0;
        int      idx  = 0;
        for (int l = 1; l <= HUFF_MAX_CODE_LEN; ++l) {
            code = (code + (uint32_t)bl_count[l - 1]) << 1;
            first_code[l] = code;
            first_idx[l]  = idx;
            idx += bl_count[l];
        }
    }

    // Per-stream pointers.
    const uint32_t* my_bits = bits_buf + (size_t)k * bits_stride;
    uint32_t        bit_len = bit_lens[k];
    uint8_t*        my_out  = out_buf  + (size_t)k * out_stride;
    uint32_t        out_len = out_lens[k];

    uint32_t pos = 0;
    for (uint32_t i = 0; i < out_len; ++i) {
        uint32_t cur = 0;
        int      l   = 0;
        bool     done = false;
        while (l < HUFF_MAX_CODE_LEN) {
            if (pos >= bit_len) break;
            uint32_t w = my_bits[pos >> 5];
            uint32_t b = (w >> (pos & 31u)) & 1u;
            pos++;
            cur = (cur << 1) | b;
            l++;
            if (bl_count[l] > 0 &&
                cur < first_code[l] + (uint32_t)bl_count[l]) {
                int o = first_idx[l] + (int)(cur - first_code[l]);
                my_out[i] = (uint8_t)symbols_g[o];
                done = true;
                break;
            }
        }
        if (!done) {
            my_out[i] = 0xFF;
            return;
        }
        if (pos > bit_len) return;
    }
}

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
    cudaStream_t    stream)
{
    if (K == 0) return;
    dim3 block(BLOCK_SIZE);
    dim3 grid((K + BLOCK_SIZE - 1) / BLOCK_SIZE);
    huffman_decode_baseline_kernel<<<grid, block, 0, stream>>>(
        d_bits, d_bit_lens, d_out, d_out_lens, out_stride, bits_stride,
        d_code_len, d_symbols, K);
}
