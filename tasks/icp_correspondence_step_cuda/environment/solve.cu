/*
 * solve.cu -- ICP correspondence step on GPU.
 *
 * Computes nearest-neighbor matches between source and target point
 * clouds, then accumulates the centered cross-covariance matrix H and
 * the squared-distance error over valid pairs.
 *
 * Edit this file to optimize.
 */

#include "solve.h"
#include "kdtree.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

/* -------- file-scope scratch (allocated lazily, never freed) ---------- */
static int*    g_match_idx    = nullptr;       /* per-source NN target idx, -1 if invalid */
static double* g_centroid_acc = nullptr;       /* 6 doubles: src_sum xyz, tgt_sum xyz */
static double* g_H_acc        = nullptr;       /* 9 doubles */
static double* g_err_acc      = nullptr;       /* 1 double */
static int     g_match_cap    = 0;

/* ── Pass 1: brute-force nearest neighbor per source point ────────────── */

__global__ void brute_nn_kernel(const float* __restrict__ src,
                                const float* __restrict__ tgt,
                                int N, int M, float d_max_sq,
                                int* __restrict__ match_idx,
                                double* __restrict__ centroid_acc,
                                int* __restrict__ count_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px = src[3 * i + 0];
    float py = src[3 * i + 1];
    float pz = src[3 * i + 2];

    float best = FLT_MAX;
    int best_j = -1;
    for (int j = 0; j < M; ++j) {
        float dx = px - tgt[3 * j + 0];
        float dy = py - tgt[3 * j + 1];
        float dz = pz - tgt[3 * j + 2];
        float d = dx * dx + dy * dy + dz * dz;
        if (d < best) {
            best = d;
            best_j = j;
        }
    }

    if (best_j < 0 || best > d_max_sq) {
        match_idx[i] = -1;
        return;
    }
    match_idx[i] = best_j;

    /* Accumulate into centroid sums via global atomics. */
    float qx = tgt[3 * best_j + 0];
    float qy = tgt[3 * best_j + 1];
    float qz = tgt[3 * best_j + 2];
    atomicAdd(&centroid_acc[0], (double)px);
    atomicAdd(&centroid_acc[1], (double)py);
    atomicAdd(&centroid_acc[2], (double)pz);
    atomicAdd(&centroid_acc[3], (double)qx);
    atomicAdd(&centroid_acc[4], (double)qy);
    atomicAdd(&centroid_acc[5], (double)qz);
    atomicAdd(count_out, 1);
}

/* ── Pass 2: accumulate H and err using the centroids from pass 1 ─────── */

__global__ void accumulate_H_kernel(const float* __restrict__ src,
                                    const float* __restrict__ tgt,
                                    const int* __restrict__ match_idx,
                                    int N,
                                    double cx_p, double cy_p, double cz_p,
                                    double cx_q, double cy_q, double cz_q,
                                    double* __restrict__ H_acc,
                                    double* __restrict__ err_acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = match_idx[i];
    if (j < 0) return;

    double px = (double)src[3 * i + 0] - cx_p;
    double py = (double)src[3 * i + 1] - cy_p;
    double pz = (double)src[3 * i + 2] - cz_p;
    double qx = (double)tgt[3 * j + 0] - cx_q;
    double qy = (double)tgt[3 * j + 1] - cy_q;
    double qz = (double)tgt[3 * j + 2] - cz_q;

    /* H += p q^T  (3x3 row-major) */
    atomicAdd(&H_acc[0], px * qx);
    atomicAdd(&H_acc[1], px * qy);
    atomicAdd(&H_acc[2], px * qz);
    atomicAdd(&H_acc[3], py * qx);
    atomicAdd(&H_acc[4], py * qy);
    atomicAdd(&H_acc[5], py * qz);
    atomicAdd(&H_acc[6], pz * qx);
    atomicAdd(&H_acc[7], pz * qy);
    atomicAdd(&H_acc[8], pz * qz);

    /* err += squared distance (uncentered). */
    double dx = (double)src[3 * i + 0] - (double)tgt[3 * j + 0];
    double dy = (double)src[3 * i + 1] - (double)tgt[3 * j + 1];
    double dz = (double)src[3 * i + 2] - (double)tgt[3 * j + 2];
    atomicAdd(err_acc, dx * dx + dy * dy + dz * dz);
}

/* ── Tiny finalize kernel: copy fp64 accumulators into fp32 outputs. ──── */

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
        if (g_match_idx)    cudaFree(g_match_idx);
        cudaMalloc(&g_match_idx, (size_t)N * sizeof(int));
        g_match_cap = N;
    }
    if (g_centroid_acc == nullptr) {
        cudaMalloc(&g_centroid_acc, 6 * sizeof(double));
    }
    if (g_H_acc == nullptr) {
        cudaMalloc(&g_H_acc, 9 * sizeof(double));
    }
    if (g_err_acc == nullptr) {
        cudaMalloc(&g_err_acc, sizeof(double));
    }
    cudaMemsetAsync(g_centroid_acc, 0, 6 * sizeof(double), stream);
    cudaMemsetAsync(g_H_acc, 0, 9 * sizeof(double), stream);
    cudaMemsetAsync(g_err_acc, 0, sizeof(double), stream);
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
    (void)kd_nodes;
    (void)n_nodes;

    ensure_scratch(N, stream);
    cudaMemsetAsync(count_out, 0, sizeof(int), stream);

    float d_max_sq = d_max * d_max;

    int block = 128;
    int grid  = (N + block - 1) / block;

    brute_nn_kernel<<<grid, block, 0, stream>>>(src, tgt, N, M, d_max_sq,
                                                g_match_idx, g_centroid_acc,
                                                count_out);

    double centroid_h[6];
    int    count_h = 0;
    cudaMemcpyAsync(centroid_h, g_centroid_acc, 6 * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&count_h, count_out, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    double cx_p = 0, cy_p = 0, cz_p = 0;
    double cx_q = 0, cy_q = 0, cz_q = 0;
    if (count_h > 0) {
        double inv = 1.0 / (double)count_h;
        cx_p = centroid_h[0] * inv;
        cy_p = centroid_h[1] * inv;
        cz_p = centroid_h[2] * inv;
        cx_q = centroid_h[3] * inv;
        cy_q = centroid_h[4] * inv;
        cz_q = centroid_h[5] * inv;
    }

    accumulate_H_kernel<<<grid, block, 0, stream>>>(src, tgt, g_match_idx, N,
                                                    cx_p, cy_p, cz_p,
                                                    cx_q, cy_q, cz_q,
                                                    g_H_acc, g_err_acc);

    finalize_kernel<<<1, 32, 0, stream>>>(g_H_acc, g_err_acc, H_out, err_out);
}
