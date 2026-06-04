#include "solve.h"

#define LUT_BITS 10
#define LUT_SIZE (1 << LUT_BITS)
#define LUT_MASK ((1u << LUT_BITS) - 1u)

__global__ void build_luts(
    const uint8_t*  __restrict__ d_code_len,
    const uint16_t* __restrict__ d_symbols,
    uint8_t*        __restrict__ d_lut_sym,
    uint8_t*        __restrict__ d_lut_len,
    uint32_t*       __restrict__ d_lut_full,
    int*            __restrict__ d_max_len)
{
    __shared__ int bl_count[HUFF_MAX_CODE_LEN + 2];
    __shared__ uint32_t first_code[HUFF_MAX_CODE_LEN + 2];
    __shared__ int first_idx[HUFF_MAX_CODE_LEN + 2];
    __shared__ uint8_t code_len_s[256];
    __shared__ uint16_t symbols_s[256];
    __shared__ int s_max_len;
    int tid = threadIdx.x;
    if (tid == 0) s_max_len = 0;
    if (tid < 256) { code_len_s[tid] = d_code_len[tid]; symbols_s[tid] = d_symbols[tid]; }
    if (tid <= HUFF_MAX_CODE_LEN + 1) bl_count[tid] = 0;
    __syncthreads();
    if (tid < 256) { int cl = code_len_s[tid]; if (cl > 0) { atomicAdd(&bl_count[cl], 1); atomicMax(&s_max_len, cl); } }
    __syncthreads();
    if (tid == 0) {
        *d_max_len = s_max_len; bl_count[0] = 0;
        uint32_t code = 0; int idx = 0;
        for (int l = 1; l <= HUFF_MAX_CODE_LEN; ++l) {
            code = (code + (uint32_t)bl_count[l-1]) << 1;
            first_code[l] = code; first_idx[l] = idx; idx += bl_count[l];
        }
    }
    __syncthreads();
    for (int i = tid; i < LUT_SIZE; i += blockDim.x) {
        uint8_t sym = 0, len = 0;
        uint32_t raw = (uint32_t)i, cur = 0;
        for (int l = 1; l <= LUT_BITS && l <= HUFF_MAX_CODE_LEN; ++l) {
            cur = (cur << 1) | ((raw >> (l-1)) & 1u);
            if (bl_count[l] > 0 && cur >= first_code[l] && cur < first_code[l] + (uint32_t)bl_count[l]) {
                int o = first_idx[l] + (int)(cur - first_code[l]);
                sym = (uint8_t)symbols_s[o]; len = (uint8_t)l; break;
            }
        }
        d_lut_sym[i] = sym; d_lut_len[i] = len;
    }
    if (s_max_len > LUT_BITS) {
        for (int i = tid; i < (1<<16); i += blockDim.x) {
            uint32_t result = 0, raw = (uint32_t)i, cur = 0;
            for (int l = 1; l <= HUFF_MAX_CODE_LEN; ++l) {
                cur = (cur << 1) | ((raw >> (l-1)) & 1u);
                if (bl_count[l] > 0 && cur >= first_code[l] && cur < first_code[l] + (uint32_t)bl_count[l]) {
                    int o = first_idx[l] + (int)(cur - first_code[l]);
                    result = (uint32_t)((uint8_t)symbols_s[o]) | ((uint32_t)l << 8); break;
                }
            }
            d_lut_full[i] = result;
        }
    }
}

#define WARPS_PER_BLK 16
#define THREADS_PER_BLK (WARPS_PER_BLK * 32)

__global__ __launch_bounds__(THREADS_PER_BLK)
void huffman_decode_fast(
    const uint32_t* __restrict__ bits_buf,
    const uint32_t* __restrict__ bit_lens,
    uint8_t*        __restrict__ out_buf,
    const uint32_t* __restrict__ out_lens,
    uint32_t        out_stride,
    uint32_t        bits_stride,
    const uint8_t*  __restrict__ d_lut_sym,
    const uint8_t*  __restrict__ d_lut_len,
    uint32_t        K)
{
    __shared__ uint8_t s_sym[LUT_SIZE];
    __shared__ uint8_t s_len[LUT_SIZE];
    int tid = threadIdx.x;
    for (int i = tid; i < LUT_SIZE; i += THREADS_PER_BLK) { s_sym[i] = d_lut_sym[i]; s_len[i] = d_lut_len[i]; }
    __syncthreads();
    if ((tid & 31) != 0) return;
    uint32_t k = blockIdx.x * WARPS_PER_BLK + (tid >> 5);
    if (k >= K) return;
    const uint32_t* my_bits = bits_buf + (size_t)k * bits_stride;
    uint8_t* my_out = out_buf + (size_t)k * out_stride;
    uint32_t out_len = out_lens[k];
    uint32_t max_words = (bit_lens[k] + 31) >> 5;
    uint64_t buf = 0; int buf_bits = 0; uint32_t word_idx = 0;
    if (word_idx < max_words) { buf = (uint64_t)my_bits[word_idx++]; buf_bits = 32; }
    if (word_idx < max_words) { buf |= ((uint64_t)my_bits[word_idx++]) << 32; buf_bits = 64; }
    #define D(sv) { uint32_t idx = (uint32_t)(buf & LUT_MASK); sv = s_sym[idx]; int l = s_len[idx]; buf >>= l; buf_bits -= l; }
    #define R() if (buf_bits <= 32 && word_idx < max_words) { buf |= ((uint64_t)my_bits[word_idx++]) << buf_bits; buf_bits += 32; }
    
    // 12 symbols per iteration, D D D R pattern, writes interleaved
    uint32_t out_len12 = out_len - (out_len % 12);
    uint32_t i = 0;
    for (; i < out_len12; i += 12) {
        uint8_t s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb;
        D(s0); D(s1); D(s2); R();
        D(s3); D(s4); D(s5); R();
        *((uint32_t*)(my_out + i)) = (uint32_t)s0 | ((uint32_t)s1 << 8) | ((uint32_t)s2 << 16) | ((uint32_t)s3 << 24);
        D(s6); D(s7); D(s8); R();
        *((uint32_t*)(my_out + i + 4)) = (uint32_t)s4 | ((uint32_t)s5 << 8) | ((uint32_t)s6 << 16) | ((uint32_t)s7 << 24);
        D(s9); D(sa); D(sb); R();
        *((uint32_t*)(my_out + i + 8)) = (uint32_t)s8 | ((uint32_t)s9 << 8) | ((uint32_t)sa << 16) | ((uint32_t)sb << 24);
    }
    for (; i < out_len; ++i) { uint8_t s; D(s); my_out[i] = s; R(); }
    #undef D
    #undef R
}

__global__ __launch_bounds__(THREADS_PER_BLK)
void huffman_decode_fallback(
    const uint32_t* __restrict__ bits_buf,
    const uint32_t* __restrict__ bit_lens,
    uint8_t*        __restrict__ out_buf,
    const uint32_t* __restrict__ out_lens,
    uint32_t        out_stride,
    uint32_t        bits_stride,
    const uint8_t*  __restrict__ d_lut_sym,
    const uint8_t*  __restrict__ d_lut_len,
    const uint32_t* __restrict__ d_full_lut,
    uint32_t        K)
{
    __shared__ uint8_t s_sym[LUT_SIZE];
    __shared__ uint8_t s_len[LUT_SIZE];
    int tid = threadIdx.x;
    for (int i = tid; i < LUT_SIZE; i += THREADS_PER_BLK) { s_sym[i] = d_lut_sym[i]; s_len[i] = d_lut_len[i]; }
    __syncthreads();
    if ((tid & 31) != 0) return;
    uint32_t k = blockIdx.x * WARPS_PER_BLK + (tid >> 5);
    if (k >= K) return;
    const uint32_t* my_bits = bits_buf + (size_t)k * bits_stride;
    uint8_t* my_out = out_buf + (size_t)k * out_stride;
    uint32_t out_len = out_lens[k];
    uint32_t max_words = (bit_lens[k] + 31) >> 5;
    uint64_t buf = 0; int buf_bits = 0; uint32_t word_idx = 0;
    if (word_idx < max_words) { buf = (uint64_t)my_bits[word_idx++]; buf_bits = 32; }
    if (word_idx < max_words) { buf |= ((uint64_t)my_bits[word_idx++]) << 32; buf_bits = 64; }
    for (uint32_t i = 0; i < out_len; ++i) {
        uint32_t idx = (uint32_t)(buf & LUT_MASK);
        int len = s_len[idx];
        if (len > 0) { my_out[i] = s_sym[idx]; buf >>= len; buf_bits -= len; }
        else { uint32_t e = __ldg(&d_full_lut[(uint32_t)(buf & 0xFFFFu)]); my_out[i] = (uint8_t)e; len = (int)(e >> 8); buf >>= len; buf_bits -= len; }
        if (buf_bits <= 32 && word_idx < max_words) { buf |= ((uint64_t)my_bits[word_idx++]) << buf_bits; buf_bits += 32; }
    }
}

static uint8_t* d_lut_sym_g = nullptr;
static uint8_t* d_lut_len_g = nullptr;
static uint32_t* d_full_lut_global = nullptr;
static int* d_max_len_global = nullptr;
static int cached_max_len = -1;
static const uint8_t* last_code_len = nullptr;

void huffman_decode(
    const uint32_t* d_bits, const uint32_t* d_bit_lens,
    uint8_t* d_out, const uint32_t* d_out_lens,
    uint32_t out_stride, uint32_t bits_stride,
    const uint8_t* d_code_len, const uint16_t* d_symbols,
    uint32_t K, cudaStream_t stream)
{
    if (K == 0) return;
    if (!d_lut_sym_g) {
        cudaMalloc(&d_lut_sym_g, LUT_SIZE);
        cudaMalloc(&d_lut_len_g, LUT_SIZE);
        cudaMalloc(&d_full_lut_global, sizeof(uint32_t) * (1 << 16));
        cudaMalloc(&d_max_len_global, sizeof(int));
    }
    if (d_code_len != last_code_len) {
        build_luts<<<1, 1024, 0, stream>>>(d_code_len, d_symbols, d_lut_sym_g, d_lut_len_g, d_full_lut_global, d_max_len_global);
        cudaMemcpyAsync(&cached_max_len, d_max_len_global, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        last_code_len = d_code_len;
    }
    int n_blocks = (K + WARPS_PER_BLK - 1) / WARPS_PER_BLK;
    if (cached_max_len <= LUT_BITS) {
        huffman_decode_fast<<<n_blocks, THREADS_PER_BLK, 0, stream>>>(
            d_bits, d_bit_lens, d_out, d_out_lens, out_stride, bits_stride, d_lut_sym_g, d_lut_len_g, K);
    } else {
        huffman_decode_fallback<<<n_blocks, THREADS_PER_BLK, 0, stream>>>(
            d_bits, d_bit_lens, d_out, d_out_lens, out_stride, bits_stride, d_lut_sym_g, d_lut_len_g, d_full_lut_global, K);
    }
}