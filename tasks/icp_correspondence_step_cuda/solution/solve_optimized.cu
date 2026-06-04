/*
 * solve_optimized.cu -- Optimized ICP correspondence step on GPU.
 *
 * Optimizations vs. baseline:
 *
 *  1. KD-tree traversal of the precomputed tree (kd_nodes):
 *       - O(N log M) instead of O(N M).
 *       - Iterative DFS with a small register stack (depth ~ log2(M)+8).
 *       - Squared-distance-to-AABB pruning culls subtrees whose closest
 *         possible point is already farther than the current best.
 *  2. Two-pass with on-GPU centroid reduction (no host sync):
 *       - Pass 1: each thread finds its NN, atomically adds (p,q,1)
 *         into a 7-int/double accumulator using one block-shared add
 *         followed by a single warp / block reduction.
 *       - Centroid divide is done in a tiny single-block kernel.
 *       - Pass 2: each thread computes (p-cP)(q-cQ)^T contribution
 *         and the warp / block performs a 9-element reduction with
 *         __shfl_down_sync, then one global atomicAdd per block.
 *  3. fp64 accumulators throughout (numerical stability), fp32 outputs.
 *
 * Pre-built KD-tree layout (see /app/kdtree.h):
 *   struct KDNode {
 *       float bbox_min[3];
 *       float bbox_max[3];
 *       int   point_idx;
 *       int   left;
 *       int   right;
 *       int   axis;       // 0..2 = split axis, 3 = leaf
 *   };
 */

#include "solve.h"
#include "kdtree.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

#define KD_STACK_DEPTH 64
#define BLOCK_NN       128
#define BLOCK_H        128

/* -------- file-scope scratch (allocated lazily, never freed) ---------- */
static int*    g_match_idx     = nullptr;
static double* g_centroid_acc  = nullptr;     /* 6 doubles */
static double* g_H_acc         = nullptr;     /* 9 doubles */
static double* g_err_acc       = nullptr;     /* 1 double */
static double* g_centroid_div  = nullptr;     /* 6 doubles, divided */
static int     g_match_cap     = 0;

/* ── KD-tree NN search (iterative, register stack) ─────────────────────── */

__device__ static inline float aabb_dist2(float px, float py, float pz,
                                          const float* __restrict__ bmin,
                                          const float* __restrict__ bmax) {
    float dx = fmaxf(0.0f, fmaxf(bmin[0] - px, px - bmax[0]));
    float dy = fmaxf(0.0f, fmaxf(bmin[1] - py, py - bmax[1]));
    float dz = fmaxf(0.0f, fmaxf(bmin[2] - pz, pz - bmax[2]));
    return dx * dx + dy * dy + dz * dz;
}

__device__ static int kd_find_nn(const KDNode* __restrict__ nodes,
                                 const float*  __restrict__ tgt,
                                 float px, float py, float pz,
                                 float d_max_sq,
                                 float* best_d2_out) {
    int stack[KD_STACK_DEPTH];
    int top = 0;
    stack[top++] = 0;

    float best = d_max_sq;        /* never accept beyond d_max */
    int   best_j = -1;

    while (top > 0) {
        int idx = stack[--top];
        if (idx < 0) continue;

        const KDNode n = nodes[idx];

        /* Prune: minimum possible squared distance to bbox of this subtree. */
        float d2_box = aabb_dist2(px, py, pz, n.bbox_min, n.bbox_max);
        if (d2_box >= best) continue;

        /* Test the pivot point itself. */
        int pi = n.point_idx;
        float qx = tgt[3 * pi + 0];
        float qy = tgt[3 * pi + 1];
        float qz = tgt[3 * pi + 2];
        float dx = px - qx, dy = py - qy, dz = pz - qz;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < best) { best = d2; best_j = pi; }

        if (n.axis == 3) continue;          /* leaf */

        /* Visit nearer child first. */
        int ax = n.axis;
        float p_ax = (ax == 0) ? px : (ax == 1) ? py : pz;
        float pivot = (ax == 0) ? qx : (ax == 1) ? qy : qz;
        int near_c, far_c;
        if (p_ax < pivot) { near_c = n.left;  far_c = n.right; }
        else              { near_c = n.right; far_c = n.left;  }

        /* Push far first, near on top so it's popped next. */
        if (far_c  >= 0 && top < KD_STACK_DEPTH) stack[top++] = far_c;
        if (near_c >= 0 && top < KD_STACK_DEPTH) stack[top++] = near_c;
    }

    *best_d2_out = best;
    return best_j;
}

/* ── Pass 1: find NN, accumulate centroids and count ───────────────────── */

__global__ void nn_pass_kernel(const float*  __restrict__ src,
                               const float*  __restrict__ tgt,
                               const KDNode* __restrict__ kd,
                               int N, float d_max_sq,
                               int* __restrict__ match_idx,
                               double* __restrict__ centroid_acc,
                               int* __restrict__ count_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Per-thread contribution. */
    double cp[3] = {0.0, 0.0, 0.0};
    double cq[3] = {0.0, 0.0, 0.0};
    int   has = 0;
    int   match_j = -1;

    if (i < N) {
        float px = src[3 * i + 0];
        float py = src[3 * i + 1];
        float pz = src[3 * i + 2];
        float best_d2 = 0.0f;
        int j = kd_find_nn(kd, tgt, px, py, pz, d_max_sq, &best_d2);
        if (j >= 0) {
            float qx = tgt[3 * j + 0];
            float qy = tgt[3 * j + 1];
            float qz = tgt[3 * j + 2];
            cp[0] = (double)px; cp[1] = (double)py; cp[2] = (double)pz;
            cq[0] = (double)qx; cq[1] = (double)qy; cq[2] = (double)qz;
            has = 1;
            match_j = j;
        }
        match_idx[i] = match_j;
    }

    /* Block-level reduction in shared memory. */
    __shared__ double s_cp[3];
    __shared__ double s_cq[3];
    __shared__ int    s_has;
    if (threadIdx.x == 0) {
        s_cp[0] = s_cp[1] = s_cp[2] = 0.0;
        s_cq[0] = s_cq[1] = s_cq[2] = 0.0;
        s_has = 0;
    }
    __syncthreads();

    /* Warp-level shuffle reduction first (cheap). */
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        cp[0] += __shfl_down_sync(mask, cp[0], off);
        cp[1] += __shfl_down_sync(mask, cp[1], off);
        cp[2] += __shfl_down_sync(mask, cp[2], off);
        cq[0] += __shfl_down_sync(mask, cq[0], off);
        cq[1] += __shfl_down_sync(mask, cq[1], off);
        cq[2] += __shfl_down_sync(mask, cq[2], off);
        has   += __shfl_down_sync(mask, has, off);
    }

    /* Lane-0 of each warp adds into shared. */
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&s_cp[0], cp[0]);
        atomicAdd(&s_cp[1], cp[1]);
        atomicAdd(&s_cp[2], cp[2]);
        atomicAdd(&s_cq[0], cq[0]);
        atomicAdd(&s_cq[1], cq[1]);
        atomicAdd(&s_cq[2], cq[2]);
        atomicAdd(&s_has, has);
    }
    __syncthreads();

    /* One thread per block flushes to global. */
    if (threadIdx.x == 0) {
        atomicAdd(&centroid_acc[0], s_cp[0]);
        atomicAdd(&centroid_acc[1], s_cp[1]);
        atomicAdd(&centroid_acc[2], s_cp[2]);
        atomicAdd(&centroid_acc[3], s_cq[0]);
        atomicAdd(&centroid_acc[4], s_cq[1]);
        atomicAdd(&centroid_acc[5], s_cq[2]);
        atomicAdd(count_out, s_has);
    }
}

/* ── Single-block kernel: divide centroids by count -------------------- */

__global__ void centroid_div_kernel(const double* __restrict__ centroid_acc,
                                    const int*    __restrict__ count_out,
                                    double*       __restrict__ centroid_div) {
    int t = threadIdx.x;
    if (t < 6) {
        int c = *count_out;
        double inv = (c > 0) ? (1.0 / (double)c) : 0.0;
        centroid_div[t] = centroid_acc[t] * inv;
    }
}

/* ── Pass 2: accumulate H and err using centroids on GPU ─────────────── */

__global__ void H_pass_kernel(const float*  __restrict__ src,
                              const float*  __restrict__ tgt,
                              const int*    __restrict__ match_idx,
                              const double* __restrict__ centroid_div,
                              int N,
                              double* __restrict__ H_acc,
                              double* __restrict__ err_acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double H[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) H[k] = 0.0;
    double err = 0.0;

    if (i < N) {
        int j = match_idx[i];
        if (j >= 0) {
            double cx_p = centroid_div[0];
            double cy_p = centroid_div[1];
            double cz_p = centroid_div[2];
            double cx_q = centroid_div[3];
            double cy_q = centroid_div[4];
            double cz_q = centroid_div[5];

            double sx = (double)src[3 * i + 0];
            double sy = (double)src[3 * i + 1];
            double sz = (double)src[3 * i + 2];
            double tx = (double)tgt[3 * j + 0];
            double ty = (double)tgt[3 * j + 1];
            double tz = (double)tgt[3 * j + 2];

            double dpx = sx - cx_p, dpy = sy - cy_p, dpz = sz - cz_p;
            double dqx = tx - cx_q, dqy = ty - cy_q, dqz = tz - cz_q;

            H[0] = dpx * dqx; H[1] = dpx * dqy; H[2] = dpx * dqz;
            H[3] = dpy * dqx; H[4] = dpy * dqy; H[5] = dpy * dqz;
            H[6] = dpz * dqx; H[7] = dpz * dqy; H[8] = dpz * dqz;

            double ex = sx - tx, ey = sy - ty, ez = sz - tz;
            err = ex * ex + ey * ey + ez * ez;
        }
    }

    /* Warp reduction. */
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int k = 0; k < 9; ++k) H[k] += __shfl_down_sync(mask, H[k], off);
        err += __shfl_down_sync(mask, err, off);
    }

    __shared__ double s_H[9];
    __shared__ double s_err;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < 9; ++k) s_H[k] = 0.0;
        s_err = 0.0;
    }
    __syncthreads();
    if ((threadIdx.x & 31) == 0) {
        #pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(&s_H[k], H[k]);
        atomicAdd(&s_err, err);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int k = 0; k < 9; ++k) atomicAdd(&H_acc[k], s_H[k]);
        atomicAdd(err_acc, s_err);
    }
}

/* ── Finalize fp64 -> fp32 ----------------------------------------------- */

__global__ void finalize_kernel(const double* __restrict__ H_acc,
                                const double* __restrict__ err_acc,
                                float* __restrict__ H_out,
                                float* __restrict__ err_out) {
    int t = threadIdx.x;
    if (t < 9) H_out[t] = (float)H_acc[t];
    if (t == 9) *err_out = (float)*err_acc;
}

/* ── Entry point ──────────────────────────────────────────────────────── */

static void ensure_scratch(int N, cudaStream_t stream) {
    if (N > g_match_cap) {
        if (g_match_idx) cudaFree(g_match_idx);
        cudaMalloc(&g_match_idx, (size_t)N * sizeof(int));
        g_match_cap = N;
    }
    if (g_centroid_acc == nullptr) cudaMalloc(&g_centroid_acc, 6 * sizeof(double));
    if (g_centroid_div == nullptr) cudaMalloc(&g_centroid_div, 6 * sizeof(double));
    if (g_H_acc        == nullptr) cudaMalloc(&g_H_acc,        9 * sizeof(double));
    if (g_err_acc      == nullptr) cudaMalloc(&g_err_acc,          sizeof(double));
    cudaMemsetAsync(g_centroid_acc, 0, 6 * sizeof(double), stream);
    cudaMemsetAsync(g_centroid_div, 0, 6 * sizeof(double), stream);
    cudaMemsetAsync(g_H_acc,        0, 9 * sizeof(double), stream);
    cudaMemsetAsync(g_err_acc,      0,     sizeof(double), stream);
}

void icp_correspondence(const float* src,
                        const float* tgt,
                        int N, int M,
                        float d_max,
                        const void* kd_nodes,
                        int n_nodes,
                        float* H_out,
                        float* err_out,
                        int* count_out,
                        cudaStream_t stream) {
    (void)M;
    (void)n_nodes;

    ensure_scratch(N, stream);
    cudaMemsetAsync(count_out, 0, sizeof(int), stream);

    float d_max_sq = d_max * d_max;
    const KDNode* kd = (const KDNode*)kd_nodes;

    int grid_nn = (N + BLOCK_NN - 1) / BLOCK_NN;
    nn_pass_kernel<<<grid_nn, BLOCK_NN, 0, stream>>>(src, tgt, kd, N, d_max_sq,
                                                     g_match_idx,
                                                     g_centroid_acc,
                                                     count_out);

    centroid_div_kernel<<<1, 32, 0, stream>>>(g_centroid_acc, count_out,
                                              g_centroid_div);

    int grid_h = (N + BLOCK_H - 1) / BLOCK_H;
    H_pass_kernel<<<grid_h, BLOCK_H, 0, stream>>>(src, tgt, g_match_idx,
                                                  g_centroid_div, N,
                                                  g_H_acc, g_err_acc);

    finalize_kernel<<<1, 32, 0, stream>>>(g_H_acc, g_err_acc, H_out, err_out);
}
