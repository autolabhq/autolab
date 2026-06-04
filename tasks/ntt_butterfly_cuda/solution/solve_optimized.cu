#include "solve.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint64_t P = NTT_PRIME;          // 2^64 - 2^32 + 1
constexpr uint64_t EPSILON = (1ULL << 32) - 1ULL;  // 2^32 - 1 == 2^64 mod P

// ---------- Goldilocks-prime fast modular arithmetic ----------
//
// p = 2^64 - 2^32 + 1, so 2^64 = 2^32 - 1 (mod p). This lets us reduce
// a 128-bit product without a 64-bit divide.

__device__ __forceinline__ uint64_t add_modp(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a || s >= P) s -= P;
    return s;
}

__device__ __forceinline__ uint64_t sub_modp(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + (P - b));
}

// Reduce a 128-bit value (hi:lo) to a 64-bit result in [0, P), using the Goldilocks identity
// 2^64 = 2^32 - 1 mod p. Splits hi into hi_hi (top 32 bits) and hi_lo (lower 32 bits):
//   x = lo + 2^64 * hi
//     = lo + (2^32 - 1) * hi
//     = lo + (hi_lo << 32) - hi_lo + hi_hi * (2^32 - 1)^2 ...
// Following Plonky2's reduce128: combined high contribution = (hi_lo << 32) - hi_lo + hi_hi.
// The result of (hi_lo << 32) - hi_lo - hi_hi (with corrections) lands in [0, p) after one
// subtraction.
__device__ __forceinline__ uint64_t reduce128(uint64_t lo, uint64_t hi) {
    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo = hi & 0xFFFFFFFFULL;

    // r0 = lo - hi_hi (with borrow). If borrow, add EPSILON (== 2^64 - p) to compensate.
    uint64_t r0;
    unsigned borrow = (lo < hi_hi) ? 1u : 0u;
    r0 = lo - hi_hi;
    if (borrow) r0 -= EPSILON;

    // r1 = r0 + (hi_lo << 32) - hi_lo, computed as r0 + product where product = hi_lo * EPSILON.
    // (hi_lo * EPSILON) = (hi_lo << 32) - hi_lo, fits in 64 bits since hi_lo < 2^32.
    uint64_t product = (hi_lo << 32) - hi_lo;
    uint64_t r1 = r0 + product;
    if (r1 < r0) {
        // overflow -> subtract EPSILON
        r1 += EPSILON;
    }
    if (r1 >= P) r1 -= P;
    return r1;
}

__device__ __forceinline__ uint64_t mul_modp(uint64_t a, uint64_t b) {
    unsigned long long lo = a * b;
    unsigned long long hi = __umul64hi(a, b);
    return reduce128((uint64_t)lo, (uint64_t)hi);
}

__device__ __forceinline__ uint64_t pow_modp(uint64_t base, uint64_t exp) {
    uint64_t r = 1;
    uint64_t b = base;
    if (b >= P) b -= P;
    while (exp > 0) {
        if (exp & 1ULL) r = mul_modp(r, b);
        b = mul_modp(b, b);
        exp >>= 1;
    }
    return r;
}

// ---------- twiddle precompute ----------
//
// We precompute, into the workspace, omega_n^j for j in [0, n/2) -- enough twiddles for every
// stage, since omega_(2^s)^j = omega_n^(j * (n / 2^s)). One small kernel does it once per
// invocation; later butterfly stages read from this single shared table.

__device__ __forceinline__ uint64_t omega_n(int n) {
    // omega_(2^32) = 7^(2^32 - 1)
    uint64_t w = pow_modp(7ULL, ((uint64_t)1 << 32) - 1ULL);
    int log_n = 0;
    while ((1 << log_n) < n) ++log_n;
    int shift = 32 - log_n;
    for (int i = 0; i < shift; ++i) {
        w = mul_modp(w, w);
    }
    return w;
}

__global__ void compute_twiddles_kernel(uint64_t* twiddles, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n >> 1;
    if (t >= half) return;
    uint64_t w = omega_n(n);
    twiddles[t] = pow_modp(w, (uint64_t)t);
}

__device__ __forceinline__ uint32_t bitrev(uint32_t x, int bits) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = (x >> 16) | (x << 16);
    return x >> (32 - bits);
}

// ---------- shared-memory butterfly for the inner CHUNK stages ----------
//
// Each block loads CHUNK=1024 elements of a row into shared memory after the global bit-reverse,
// runs all log2(CHUNK)=10 butterfly stages in shared memory, and writes back. Twiddle indices
// for stage s within a CHUNK starting at chunk_off use w_n^(j * (n / 2^s)) where j is the
// position within the m-block.

constexpr int CHUNK_LOG = 10;          // 1024 elements per block
constexpr int CHUNK = 1 << CHUNK_LOG;  // 1024

__global__ void __launch_bounds__(512, 2)
intra_chunk_kernel(uint64_t* __restrict__ data,
                   const uint64_t* __restrict__ twiddles,
                   int batch, int n) {
    int row = blockIdx.y;
    int chunk = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= batch) return;

    __shared__ uint64_t s_data[CHUNK];

    int chunk_off = chunk * CHUNK;
    uint64_t* row_data = data + (size_t)row * (size_t)n + (size_t)chunk_off;

    // Cooperative load: 512 threads, 2 elements each.
    s_data[tid] = row_data[tid];
    s_data[tid + (CHUNK >> 1)] = row_data[tid + (CHUNK >> 1)];
    __syncthreads();

    // Run intra-chunk butterfly stages s = 1 .. CHUNK_LOG.
    // For stage s of the global transform of length n: omega_(2^s)^j = twiddles[j * (n / 2^s)].
    int half_total = CHUNK >> 1;
    for (int s = 1; s <= CHUNK_LOG; ++s) {
        int m = 1 << s;
        int half = m >> 1;
        int twiddle_stride = n >> s;

        // Each thread handles one (k+j) pair within the chunk.
        int pair = tid;
        // 512 threads, half_total = 512 pairs -- exact.
        int block_id = pair / half;
        int j = pair - block_id * half;
        int k = block_id * m;

        uint64_t w = twiddles[(size_t)j * (size_t)twiddle_stride];
        uint64_t u = s_data[k + j];
        uint64_t v = mul_modp(s_data[k + j + half], w);
        uint64_t up = add_modp(u, v);
        uint64_t dn = sub_modp(u, v);
        __syncthreads();
        s_data[k + j] = up;
        s_data[k + j + half] = dn;
        __syncthreads();
        (void)half_total;
    }

    row_data[tid] = s_data[tid];
    row_data[tid + (CHUNK >> 1)] = s_data[tid + (CHUNK >> 1)];
}

// ---------- inter-chunk butterfly (one stage per launch, global memory) ----------
//
// For stages s > CHUNK_LOG, butterflies span across chunks; we must write to global memory.

__global__ void inter_stage_kernel(uint64_t* __restrict__ data,
                                   const uint64_t* __restrict__ twiddles,
                                   int batch, int n, int s) {
    int row = blockIdx.y;
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int half_total = n >> 1;
    if (row >= batch || pair >= half_total) return;

    int m = 1 << s;
    int half = m >> 1;
    int twiddle_stride = n >> s;

    int block_id = pair / half;
    int j = pair - block_id * half;
    int k = block_id * m;

    uint64_t* row_data = data + (size_t)row * (size_t)n;
    uint64_t w = twiddles[(size_t)j * (size_t)twiddle_stride];
    uint64_t u = row_data[k + j];
    uint64_t v = mul_modp(row_data[k + j + half], w);
    row_data[k + j] = add_modp(u, v);
    row_data[k + j + half] = sub_modp(u, v);
}

// ---------- bit-reverse permutation kernel ----------

__global__ void bitrev_kernel(uint64_t* data, int batch, int n, int log_n) {
    int row = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || i >= n) return;
    uint32_t j = bitrev((uint32_t)i, log_n);
    if ((int)j > i) {
        uint64_t* row_data = data + (size_t)row * (size_t)n;
        uint64_t a = row_data[i];
        uint64_t b = row_data[j];
        row_data[i] = b;
        row_data[j] = a;
    }
}

}  // namespace

void ntt_forward_cuda(uint64_t* data,
                      void* workspace,
                      size_t workspace_bytes,
                      int batch,
                      int n,
                      cudaStream_t stream) {
    if (!data || !workspace || batch <= 0 || n <= 0) return;

    int log_n = 0;
    while ((1 << log_n) < n) ++log_n;

    int half_total = n >> 1;
    size_t twiddle_bytes = (size_t)half_total * sizeof(uint64_t);
    if (twiddle_bytes > workspace_bytes) return;

    uint64_t* twiddles = static_cast<uint64_t*>(workspace);

    // Precompute twiddles for the whole transform.
    {
        int block = 256;
        int grid = (half_total + block - 1) / block;
        compute_twiddles_kernel<<<grid, block, 0, stream>>>(twiddles, n);
    }

    // Bit-reverse the input so the iterative butterfly sees natural-order indexing.
    {
        int block = 256;
        int grid_x = (n + block - 1) / block;
        dim3 grid(grid_x, batch);
        bitrev_kernel<<<grid, block, 0, stream>>>(data, batch, n, log_n);
    }

    // For small n where chunk == n, just do the intra-chunk pass with chunk=n. Otherwise tile.
    if (n <= CHUNK) {
        // Single chunk per row -- the intra_chunk kernel assumes CHUNK exactly. Use the global
        // butterfly path for safety on small n.
        int block = 256;
        int grid_x = (half_total + block - 1) / block;
        dim3 grid(grid_x, batch);
        for (int s = 1; s <= log_n; ++s) {
            inter_stage_kernel<<<grid, block, 0, stream>>>(data, twiddles, batch, n, s);
        }
        return;
    }

    // Run intra-chunk stages (1..CHUNK_LOG) per chunk in shared memory.
    {
        int chunks = n / CHUNK;
        dim3 grid(chunks, batch);
        intra_chunk_kernel<<<grid, CHUNK / 2, 0, stream>>>(data, twiddles, batch, n);
    }

    // Run inter-chunk stages (CHUNK_LOG+1 .. log_n) in global memory.
    {
        int block = 256;
        int grid_x = (half_total + block - 1) / block;
        dim3 grid(grid_x, batch);
        for (int s = CHUNK_LOG + 1; s <= log_n; ++s) {
            inter_stage_kernel<<<grid, block, 0, stream>>>(data, twiddles, batch, n, s);
        }
    }
}
