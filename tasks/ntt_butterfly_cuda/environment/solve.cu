#include "solve.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint64_t P = NTT_PRIME;  // 2^64 - 2^32 + 1

// Modular addition: returns (a + b) mod P. Inputs must be in [0, P).
__device__ __forceinline__ uint64_t mod_add_dev(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a || s >= P) {
        s -= P;
    }
    return s;
}

__device__ __forceinline__ uint64_t mod_sub_dev(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + (P - b));
}

// 128-bit product of two 64-bit values, then reduced mod P using the integer modulo operator on
// the upper word followed by another integer mod on the recombined value.
__device__ __forceinline__ uint64_t mod_mul_dev(uint64_t a, uint64_t b) {
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);
    // (hi * 2^64 + lo) mod P. Reduce the high half first, then add the low half and reduce again.
    unsigned long long hi_red = hi % P;
    // 2^64 mod P. Compute it once per call -- inexpensive and keeps the code obvious.
    unsigned long long two64_mod = ((unsigned long long)0xFFFFFFFFFFFFFFFFULL % P + 1ULL) % P;
    unsigned long long high_part = (unsigned long long)(((__uint128_t)hi_red * (__uint128_t)two64_mod) % (__uint128_t)P);
    unsigned long long lo_red = lo % P;
    unsigned long long s = high_part + lo_red;
    if (s < high_part || s >= P) s -= P;
    return s;
}

__device__ __forceinline__ uint64_t mod_pow_dev(uint64_t base, uint64_t exp) {
    uint64_t r = 1;
    uint64_t b = base % P;
    while (exp > 0) {
        if (exp & 1ULL) r = mod_mul_dev(r, b);
        b = mod_mul_dev(b, b);
        exp >>= 1;
    }
    return r;
}

// Primitive 2^log_n-th root of unity in F_P.
__device__ __forceinline__ uint64_t omega_for_stage(int log_n) {
    // omega_(2^32) = 7^((p-1)/2^32) = 7^(2^32 - 1) mod P.
    uint64_t omega32 = mod_pow_dev(7ULL, ((uint64_t)1 << 32) - 1ULL);
    // Square it (32 - log_n) times to get the primitive 2^log_n-th root of unity.
    int shift = 32 - log_n;
    for (int i = 0; i < shift; ++i) {
        omega32 = mod_mul_dev(omega32, omega32);
    }
    return omega32;
}

__device__ __forceinline__ uint32_t bit_reverse_dev(uint32_t x, int bits) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = (x >> 16) | (x << 16);
    return x >> (32 - bits);
}

// Bit-reverse permutation kernel: one thread per element, swaps with its bit-reversed partner.
__global__ void bitrev_kernel(uint64_t* data, int batch, int n, int log_n) {
    int row = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || i >= n) return;
    uint32_t j = bit_reverse_dev((uint32_t)i, log_n);
    if ((int)j > i) {
        uint64_t* row_data = data + (size_t)row * (size_t)n;
        uint64_t a = row_data[i];
        uint64_t b = row_data[j];
        row_data[i] = b;
        row_data[j] = a;
    }
}

// One stage of the butterfly. Each thread handles one (k+j) pair within one block of size m.
// log_n is the total transform length log; s is the current stage in [1, log_n].
__global__ void butterfly_stage_kernel(uint64_t* data, int batch, int n, int s) {
    int row = blockIdx.y;
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int half_total = n >> 1;
    if (row >= batch || pair >= half_total) return;

    int m = 1 << s;
    int half = m >> 1;
    int block_id = pair / half;          // which butterfly block within the row
    int j = pair - block_id * half;      // index within the half
    int k = block_id * m;

    uint64_t omega_m = omega_for_stage(s);
    uint64_t w = mod_pow_dev(omega_m, (uint64_t)j);

    uint64_t* row_data = data + (size_t)row * (size_t)n;
    uint64_t u = row_data[k + j];
    uint64_t v = mod_mul_dev(row_data[k + j + half], w);
    row_data[k + j] = mod_add_dev(u, v);
    row_data[k + j + half] = mod_sub_dev(u, v);
}

}  // namespace

void ntt_forward_cuda(uint64_t* data,
                      void* /*workspace*/,
                      size_t /*workspace_bytes*/,
                      int batch,
                      int n,
                      cudaStream_t stream) {
    if (!data || batch <= 0 || n <= 0) return;

    int log_n = 0;
    while ((1 << log_n) < n) ++log_n;

    int block = 256;
    int grid_x = (n + block - 1) / block;
    dim3 grid(grid_x, batch);
    bitrev_kernel<<<grid, block, 0, stream>>>(data, batch, n, log_n);

    int half_total = n >> 1;
    int grid_x2 = (half_total + block - 1) / block;
    dim3 grid2(grid_x2, batch);
    for (int s = 1; s <= log_n; ++s) {
        butterfly_stage_kernel<<<grid2, block, 0, stream>>>(data, batch, n, s);
    }
}
