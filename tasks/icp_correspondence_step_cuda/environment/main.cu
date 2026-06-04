/*
 * main.cu -- harness for the ICP correspondence step.
 *
 * Modes:
 *   ./icp_corr --verify              public correctness check (small,
 *                                    fixed seeds, CPU brute-force ref)
 *   ./icp_corr --benchmark           public timed run (no scoring)
 *   ./icp_corr --benchmark-verify    hidden-seeded scored run; prints
 *                                    __VERIFIER_CORRECTNESS__,
 *                                    __VERIFIER_BENCHMARK__,
 *                                    __VERIFIER_SCORE__ on stderr.
 *
 * The KD-tree over the target cloud is built ONCE on the host before
 * any timing begins, uploaded to the GPU, and passed to
 * icp_correspondence() via the kd_nodes / n_nodes arguments. The agent
 * MUST NOT rebuild it.
 *
 * DO NOT MODIFY THIS FILE.
 */

#include "solve.h"
#include "kdtree.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK(expr)                                                    \
    do {                                                                    \
        cudaError_t err__ = (expr);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                \
                         cudaGetErrorString(err__), __FILE__, __LINE__);    \
            return false;                                                   \
        }                                                                   \
    } while (0)

/* -- Configuration ----------------------------------------------------- */

static constexpr int   BENCH_N        = 200000;
static constexpr int   BENCH_M        = 500000;
static constexpr float BENCH_DMAX     = 0.05f;

static constexpr int   VERIFY_N       = 2000;
static constexpr int   VERIFY_M       = 5000;
static constexpr float VERIFY_DMAX    = 0.05f;

static constexpr int   N_WARMUP       = 2;
static constexpr int   N_TIMED        = 7;

/* -- SplitMix64 PRNG --------------------------------------------------- */

struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t s) : state(s) {}
    uint64_t next() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    float next_unit() {
        return (float)((next() >> 11) * (1.0 / 9007199254740992.0));
    }
    float next_signed() { return next_unit() * 2.0f - 1.0f; }
    int   uniform_int(int limit) { return (int)(next() % (uint64_t)limit); }
    float gauss() {
        /* Box-Muller. */
        float u1 = next_unit();
        if (u1 < 1e-10f) u1 = 1e-10f;
        float u2 = next_unit();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
    }
};

/*
 * Procedural "noisy bunny-like" point cloud:
 *   - sample positions on a unit-radius blob deformed by anisotropic
 *     scaling and three sinusoidal lobes (a synthetic stand-in for a
 *     real scanned mesh).
 *   - perturb each point by Gaussian noise with sigma 0.005.
 * Both source and target draw from the same generator with different
 * seeds so the clouds overlap heavily but are not identical, which is
 * what ICP sees in practice.
 */
static void generate_cloud(uint64_t seed, int N, std::vector<float>& out) {
    SplitMix64 rng(seed);
    out.resize((size_t)N * 3);
    for (int i = 0; i < N; ++i) {
        /* Sample on unit sphere, deform anisotropically. */
        float u = rng.next_unit() * 2.0f - 1.0f;             /* z */
        float phi = rng.next_unit() * 6.2831853f;
        float r = std::sqrt(std::max(0.0f, 1.0f - u * u));
        float x = r * std::cos(phi);
        float y = r * std::sin(phi);
        float z = u;

        /* Anisotropic scale. */
        x *= 0.45f;
        y *= 0.30f;
        z *= 0.55f;

        /* Sinusoidal lobes (bunny-like bumps). */
        float bump = 0.06f * std::sin(3.0f * x) * std::cos(2.5f * y) +
                     0.04f * std::cos(4.0f * z);
        x += bump * 0.5f;
        y += bump * 0.3f;
        z += bump * 0.7f;

        /* Gaussian noise. */
        x += 0.005f * rng.gauss();
        y += 0.005f * rng.gauss();
        z += 0.005f * rng.gauss();

        out[(size_t)3 * i + 0] = x;
        out[(size_t)3 * i + 1] = y;
        out[(size_t)3 * i + 2] = z;
    }
}

/* -- KD-tree construction (CPU, balanced median split) ----------------- */

static void compute_bbox(const std::vector<float>& pts,
                         const std::vector<int>& idx,
                         int lo, int hi,
                         float bmin[3], float bmax[3]) {
    bmin[0] = bmin[1] = bmin[2] =  FLT_MAX;
    bmax[0] = bmax[1] = bmax[2] = -FLT_MAX;
    for (int k = lo; k < hi; ++k) {
        int p = idx[k];
        for (int d = 0; d < 3; ++d) {
            float v = pts[(size_t)3 * p + d];
            if (v < bmin[d]) bmin[d] = v;
            if (v > bmax[d]) bmax[d] = v;
        }
    }
}

static int build_kd_recursive(const std::vector<float>& pts,
                              std::vector<int>& idx,
                              int lo, int hi,
                              std::vector<KDNode>& nodes) {
    if (lo >= hi) return -1;

    int node_idx = (int)nodes.size();
    nodes.emplace_back();
    KDNode n{};
    compute_bbox(pts, idx, lo, hi, n.bbox_min, n.bbox_max);

    int count = hi - lo;
    if (count == 1) {
        n.point_idx = idx[lo];
        n.axis      = 3;     /* leaf */
        n.left      = -1;
        n.right     = -1;
        nodes[node_idx] = n;
        return node_idx;
    }

    /* Choose axis with largest extent. */
    int axis = 0;
    float best_ext = n.bbox_max[0] - n.bbox_min[0];
    for (int d = 1; d < 3; ++d) {
        float e = n.bbox_max[d] - n.bbox_min[d];
        if (e > best_ext) { best_ext = e; axis = d; }
    }

    /* nth_element to find median by axis. */
    int mid = lo + count / 2;
    std::nth_element(idx.begin() + lo, idx.begin() + mid, idx.begin() + hi,
                     [&](int a, int b) {
                         return pts[(size_t)3 * a + axis] <
                                pts[(size_t)3 * b + axis];
                     });
    int pivot_idx = idx[mid];

    n.point_idx = pivot_idx;
    n.axis      = axis;
    nodes[node_idx] = n;

    int left_root  = build_kd_recursive(pts, idx, lo, mid, nodes);
    /* Right side excludes the pivot itself: place pivot only at this node,
     * and split children as [lo, mid) and [mid+1, hi). */
    int right_root = build_kd_recursive(pts, idx, mid + 1, hi, nodes);

    nodes[node_idx].left  = left_root;
    nodes[node_idx].right = right_root;
    return node_idx;
}

static void build_kdtree(const std::vector<float>& pts,
                         std::vector<KDNode>& nodes) {
    int N = (int)(pts.size() / 3);
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    nodes.clear();
    nodes.reserve((size_t)2 * N);
    build_kd_recursive(pts, idx, 0, N, nodes);
}

/* -- CPU KD-tree NN search (recursive, host-only) ---------------------- */

static void cpu_kd_nn(const std::vector<KDNode>& nodes,
                      const std::vector<float>& tgt,
                      int node_idx,
                      float px, float py, float pz,
                      float& best_d2, int& best_j) {
    if (node_idx < 0) return;
    const KDNode& n = nodes[node_idx];

    /* Squared distance from query to node's bbox. */
    float dx = std::fmax(0.0f, std::fmax(n.bbox_min[0] - px, px - n.bbox_max[0]));
    float dy = std::fmax(0.0f, std::fmax(n.bbox_min[1] - py, py - n.bbox_max[1]));
    float dz = std::fmax(0.0f, std::fmax(n.bbox_min[2] - pz, pz - n.bbox_max[2]));
    float bbox_d2 = dx * dx + dy * dy + dz * dz;
    if (bbox_d2 >= best_d2) return;

    int pi = n.point_idx;
    float qx = tgt[3 * pi + 0];
    float qy = tgt[3 * pi + 1];
    float qz = tgt[3 * pi + 2];
    float ddx = px - qx, ddy = py - qy, ddz = pz - qz;
    float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
    if (d2 < best_d2) { best_d2 = d2; best_j = pi; }

    if (n.axis == 3) return;

    int ax = n.axis;
    float p_ax = (ax == 0) ? px : (ax == 1) ? py : pz;
    float pivot = (ax == 0) ? qx : (ax == 1) ? qy : qz;
    int near_c, far_c;
    if (p_ax < pivot) { near_c = n.left;  far_c = n.right; }
    else              { near_c = n.right; far_c = n.left;  }
    cpu_kd_nn(nodes, tgt, near_c, px, py, pz, best_d2, best_j);
    cpu_kd_nn(nodes, tgt, far_c,  px, py, pz, best_d2, best_j);
}

/* -- CPU reference: ICP correspondence step (uses KD-tree for speed) --- */

struct RefResult {
    double H[9];
    double err;
    int    count;
};

static void cpu_reference(const std::vector<float>& src,
                          const std::vector<float>& tgt,
                          const std::vector<KDNode>& kd_nodes,
                          float d_max,
                          RefResult& out) {
    int N = (int)(src.size() / 3);
    float d_max_sq = d_max * d_max;

    std::vector<int> match(N, -1);
    int count = 0;
    double cx_p = 0, cy_p = 0, cz_p = 0;
    double cx_q = 0, cy_q = 0, cz_q = 0;
    double err = 0.0;

    for (int i = 0; i < N; ++i) {
        float px = src[3 * i + 0];
        float py = src[3 * i + 1];
        float pz = src[3 * i + 2];
        float best = d_max_sq;     /* never accept beyond d_max */
        int best_j = -1;
        cpu_kd_nn(kd_nodes, tgt, 0, px, py, pz, best, best_j);
        if (best_j < 0) continue;
        match[i] = best_j;
        ++count;
        cx_p += px; cy_p += py; cz_p += pz;
        cx_q += tgt[3 * best_j + 0];
        cy_q += tgt[3 * best_j + 1];
        cz_q += tgt[3 * best_j + 2];
        err += (double)best;
    }

    if (count > 0) {
        double inv = 1.0 / (double)count;
        cx_p *= inv; cy_p *= inv; cz_p *= inv;
        cx_q *= inv; cy_q *= inv; cz_q *= inv;
    }

    for (int k = 0; k < 9; ++k) out.H[k] = 0.0;
    for (int i = 0; i < N; ++i) {
        int j = match[i];
        if (j < 0) continue;
        double dpx = (double)src[3 * i + 0] - cx_p;
        double dpy = (double)src[3 * i + 1] - cy_p;
        double dpz = (double)src[3 * i + 2] - cz_p;
        double dqx = (double)tgt[3 * j + 0] - cx_q;
        double dqy = (double)tgt[3 * j + 1] - cy_q;
        double dqz = (double)tgt[3 * j + 2] - cz_q;
        out.H[0] += dpx * dqx;
        out.H[1] += dpx * dqy;
        out.H[2] += dpx * dqz;
        out.H[3] += dpy * dqx;
        out.H[4] += dpy * dqy;
        out.H[5] += dpy * dqz;
        out.H[6] += dpz * dqx;
        out.H[7] += dpz * dqy;
        out.H[8] += dpz * dqz;
    }
    out.err   = err;
    out.count = count;
}

/* -- GPU buffer management --------------------------------------------- */

struct DeviceBuffers {
    float*  d_src       = nullptr;
    float*  d_tgt       = nullptr;
    KDNode* d_nodes     = nullptr;
    float*  d_H         = nullptr;
    float*  d_err       = nullptr;
    int*    d_count     = nullptr;
    cudaStream_t stream = nullptr;
};

static void release(DeviceBuffers& dev) {
    if (dev.d_src)    cudaFree(dev.d_src);
    if (dev.d_tgt)    cudaFree(dev.d_tgt);
    if (dev.d_nodes)  cudaFree(dev.d_nodes);
    if (dev.d_H)      cudaFree(dev.d_H);
    if (dev.d_err)    cudaFree(dev.d_err);
    if (dev.d_count)  cudaFree(dev.d_count);
    if (dev.stream)   cudaStreamDestroy(dev.stream);
    dev = DeviceBuffers{};
}

static bool upload(const std::vector<float>& src,
                   const std::vector<float>& tgt,
                   const std::vector<KDNode>& nodes,
                   DeviceBuffers& dev) {
    int N = (int)(src.size() / 3);
    int M = (int)(tgt.size() / 3);
    int n_nodes = (int)nodes.size();

    CUDA_CHECK(cudaStreamCreate(&dev.stream));
    CUDA_CHECK(cudaMalloc(&dev.d_src,   (size_t)N * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.d_tgt,   (size_t)M * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.d_nodes, (size_t)n_nodes * sizeof(KDNode)));
    CUDA_CHECK(cudaMalloc(&dev.d_H,     9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.d_err,   sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev.d_count, sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(dev.d_src, src.data(),
                               (size_t)N * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(dev.d_tgt, tgt.data(),
                               (size_t)M * 3 * sizeof(float),
                               cudaMemcpyHostToDevice, dev.stream));
    CUDA_CHECK(cudaMemcpyAsync(dev.d_nodes, nodes.data(),
                               (size_t)n_nodes * sizeof(KDNode),
                               cudaMemcpyHostToDevice, dev.stream));
    /* Pre-fill outputs with NaN so any unwritten cell is detectable. */
    CUDA_CHECK(cudaMemsetAsync(dev.d_H,   0xff, 9 * sizeof(float), dev.stream));
    CUDA_CHECK(cudaMemsetAsync(dev.d_err, 0xff, sizeof(float),     dev.stream));
    CUDA_CHECK(cudaMemsetAsync(dev.d_count, 0, sizeof(int),        dev.stream));
    CUDA_CHECK(cudaStreamSynchronize(dev.stream));
    return true;
}

/* -- Comparison helpers ------------------------------------------------- */

static bool compare_results(const RefResult& ref,
                            const float (&H_got)[9],
                            float err_got, int count_got,
                            double rel_tol) {
    int mismatches = 0;
    /* count must match exactly. */
    if (count_got != ref.count) {
        std::fprintf(stderr, "count mismatch: got=%d ref=%d\n",
                     count_got, ref.count);
        return false;
    }

    /* err: relative tolerance with floor. */
    double err_diff = std::fabs((double)err_got - ref.err);
    double err_tol  = std::max(rel_tol * std::fabs(ref.err), rel_tol);
    if (err_diff > err_tol || std::isnan((double)err_got)) {
        std::fprintf(stderr, "err mismatch: got=%.6e ref=%.6e diff=%.4e tol=%.4e\n",
                     (double)err_got, ref.err, err_diff, err_tol);
        ++mismatches;
    }

    /* H: per-cell relative tolerance. */
    double H_norm = 0.0;
    for (int k = 0; k < 9; ++k) H_norm += ref.H[k] * ref.H[k];
    H_norm = std::sqrt(H_norm) + 1e-6;
    for (int k = 0; k < 9; ++k) {
        double g = (double)H_got[k];
        double r = ref.H[k];
        double diff = std::fabs(g - r);
        double tol  = std::max(rel_tol * H_norm, rel_tol * std::fabs(r));
        if (diff > tol || std::isnan(g)) {
            if (mismatches < 8) {
                std::fprintf(stderr, "H[%d] mismatch: got=%.6e ref=%.6e diff=%.4e tol=%.4e\n",
                             k, g, r, diff, tol);
            }
            ++mismatches;
        }
    }
    return mismatches == 0;
}

/* -- One forward pass --------------------------------------------------- */

static bool forward(const std::vector<float>& src,
                    const std::vector<float>& tgt,
                    const std::vector<KDNode>& nodes,
                    float d_max,
                    float (&H_out)[9],
                    float& err_out, int& count_out) {
    int N = (int)(src.size() / 3);
    int M = (int)(tgt.size() / 3);
    int n_nodes = (int)nodes.size();

    DeviceBuffers dev;
    if (!upload(src, tgt, nodes, dev)) {
        release(dev);
        return false;
    }

    icp_correspondence(dev.d_src, dev.d_tgt, N, M, d_max,
                       (const void*)dev.d_nodes, n_nodes,
                       dev.d_H, dev.d_err, dev.d_count, dev.stream);
    cudaError_t e = cudaStreamSynchronize(dev.stream);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "stream sync error: %s\n", cudaGetErrorString(e));
        release(dev);
        return false;
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(e));
        release(dev);
        return false;
    }

    cudaMemcpy(H_out, dev.d_H, 9 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&err_out, dev.d_err, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count_out, dev.d_count, sizeof(int), cudaMemcpyDeviceToHost);

    release(dev);
    return true;
}

/* -- Timed run ---------------------------------------------------------- */

static bool run_timed(const std::vector<float>& src,
                      const std::vector<float>& tgt,
                      const std::vector<KDNode>& nodes,
                      float d_max,
                      const RefResult& ref,
                      double rel_tol,
                      double* median_ms) {
    int N = (int)(src.size() / 3);
    int M = (int)(tgt.size() / 3);
    int n_nodes = (int)nodes.size();

    DeviceBuffers dev;
    if (!upload(src, tgt, nodes, dev)) {
        release(dev);
        return false;
    }

    /* Warmup. */
    for (int i = 0; i < N_WARMUP; ++i) {
        icp_correspondence(dev.d_src, dev.d_tgt, N, M, d_max,
                           (const void*)dev.d_nodes, n_nodes,
                           dev.d_H, dev.d_err, dev.d_count, dev.stream);
        CUDA_CHECK(cudaStreamSynchronize(dev.stream));
        CUDA_CHECK(cudaGetLastError());
    }

    /* Correctness check on the (large) bench shape. */
    {
        float H_got[9];
        float err_got = 0.0f;
        int   count_got = 0;
        cudaMemcpy(H_got, dev.d_H, 9 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&err_got, dev.d_err, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&count_got, dev.d_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (!compare_results(ref, H_got, err_got, count_got, rel_tol)) {
            release(dev);
            return false;
        }
    }

    /* Timed runs. */
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> times;
    times.reserve((size_t)N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        /* Reset outputs to NaN/zero before each run. */
        CUDA_CHECK(cudaMemsetAsync(dev.d_H,   0xff, 9 * sizeof(float), dev.stream));
        CUDA_CHECK(cudaMemsetAsync(dev.d_err, 0xff, sizeof(float),     dev.stream));
        CUDA_CHECK(cudaMemsetAsync(dev.d_count, 0, sizeof(int),        dev.stream));
        CUDA_CHECK(cudaStreamSynchronize(dev.stream));

        CUDA_CHECK(cudaEventRecord(start, dev.stream));
        icp_correspondence(dev.d_src, dev.d_tgt, N, M, d_max,
                           (const void*)dev.d_nodes, n_nodes,
                           dev.d_H, dev.d_err, dev.d_count, dev.stream);
        CUDA_CHECK(cudaEventRecord(stop, dev.stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        times.push_back((double)elapsed_ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* Final correctness check after all timed runs (catches kernels that
     * write incorrect results some fraction of the time). */
    {
        float H_got[9];
        float err_got = 0.0f;
        int   count_got = 0;
        cudaMemcpy(H_got, dev.d_H, 9 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&err_got, dev.d_err, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&count_got, dev.d_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (!compare_results(ref, H_got, err_got, count_got, rel_tol)) {
            release(dev);
            return false;
        }
    }

    release(dev);

    std::sort(times.begin(), times.end());
    *median_ms = times[times.size() / 2];
    return true;
}

/* -- Hidden seed mixer for --benchmark-verify -------------------------- */

static uint64_t hidden_seed() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t pid = (uint64_t)getpid();
    SplitMix64 mix((uint64_t)ts.tv_nsec ^
                   ((uint64_t)ts.tv_sec << 21) ^
                   (pid * 0x9e3779b97f4a7c15ULL) ^
                   0x49435045524FA427ULL);
    return mix.next();
}

/* -- Main --------------------------------------------------------------- */

int main(int argc, char** argv) {
    const char* mode = (argc >= 2) ? argv[1] : "--benchmark";

    if (std::strcmp(mode, "--verify") == 0) {
        /* Small, fixed seeds. */
        std::vector<float> src, tgt;
        generate_cloud(0xA15CECEDA7A2026ULL, VERIFY_N, src);
        generate_cloud(0xB0BB1ECEDA7A2026ULL, VERIFY_M, tgt);
        std::vector<KDNode> nodes;
        build_kdtree(tgt, nodes);

        RefResult ref{};
        cpu_reference(src, tgt, nodes, VERIFY_DMAX, ref);

        float H_got[9] = {0};
        float err_got = 0.0f;
        int   count_got = 0;
        if (!forward(src, tgt, nodes, VERIFY_DMAX, H_got, err_got, count_got)) {
            std::printf("result=fail\n");
            return 1;
        }
        if (!compare_results(ref, H_got, err_got, count_got, 1e-4)) {
            std::printf("result=fail\n");
            return 1;
        }
        std::printf("result=ok N=%d M=%d count=%d err=%.6e\n",
                    VERIFY_N, VERIFY_M, count_got, (double)err_got);
        std::fprintf(stderr, "__VERIFIER_CORRECTNESS__=PASS\n");
        return 0;
    }

    if (std::strcmp(mode, "--benchmark") == 0) {
        std::vector<float> src, tgt;
        generate_cloud(0x5BACEDA7A2024ULL ^ 0xA1ULL, BENCH_N, src);
        generate_cloud(0xDE45E2024ULL    ^ 0xB0ULL, BENCH_M, tgt);
        std::vector<KDNode> nodes;
        build_kdtree(tgt, nodes);

        RefResult ref{};
        cpu_reference(src, tgt, nodes, BENCH_DMAX, ref);

        double median_ms = 0.0;
        if (!run_timed(src, tgt, nodes, BENCH_DMAX, ref, 1e-4, &median_ms)) {
            std::printf("result=fail\n");
            return 1;
        }
        std::printf("result=ok N=%d M=%d nodes=%d count=%d time_ms=%.6f\n",
                    BENCH_N, BENCH_M, (int)nodes.size(), ref.count, median_ms);
        return 0;
    }

    if (std::strcmp(mode, "--benchmark-verify") == 0) {
        uint64_t hs = hidden_seed();
        std::vector<float> src, tgt;
        generate_cloud(hs ^ 0xA1ULL, BENCH_N, src);
        generate_cloud(hs ^ 0xB0ULL, BENCH_M, tgt);
        std::vector<KDNode> nodes;
        build_kdtree(tgt, nodes);

        std::fprintf(stderr,
                     "Computing CPU reference for benchmark-verify (N=%d M=%d)...\n",
                     BENCH_N, BENCH_M);
        RefResult ref{};
        cpu_reference(src, tgt, nodes, BENCH_DMAX, ref);
        std::fprintf(stderr,
                     "CPU reference: count=%d err=%.6e\n",
                     ref.count, ref.err);

        double median_ms = 0.0;
        if (!run_timed(src, tgt, nodes, BENCH_DMAX, ref, 1e-4, &median_ms)) {
            std::printf("result=fail\n");
            return 1;
        }
        std::printf("result=ok N=%d M=%d nodes=%d count=%d time_ms=%.6f\n",
                    BENCH_N, BENCH_M, (int)nodes.size(), ref.count, median_ms);
        std::fprintf(stderr, "__VERIFIER_CORRECTNESS__=PASS\n");
        std::fprintf(stderr, "__VERIFIER_BENCHMARK__=PASS\n");
        std::fprintf(stderr, "__VERIFIER_SCORE__=%.6f\n", median_ms);
        return 0;
    }

    std::fprintf(stderr,
                 "usage: %s [--verify|--benchmark|--benchmark-verify]\n",
                 argv[0]);
    return 2;
}
