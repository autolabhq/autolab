/*
 * main.cu -- harness for the batched forward NTT (Goldilocks prime) CUDA task. DO NOT MODIFY.
 *
 * Modes:
 *   ./ntt_butterfly --verify              public small correctness check (runtime-random seed)
 *   ./ntt_butterfly --benchmark           unscored sanity benchmark (public seed)
 *   ./ntt_butterfly --benchmark-verify    scored run; uses the env var NTT_BENCH_SEED that
 *                                         test.sh sets just before launch. Checks correctness
 *                                         on the timed workload (full + late-slice probe at a
 *                                         build-time-baked offset) and prints
 *                                             __VERIFIER_CORRECTNESS__=PASS
 *                                             __VERIFIER_BENCHMARK__=PASS
 *                                             __VERIFIER_SCORE__=<median_milliseconds>
 *                                         only on success.
 *
 * Reference NTT is computed in-binary using __int128 modular arithmetic, with all helpers and
 * tables file-local (anonymous namespace + static) so they cannot be linked or dlopen'd from a
 * helper translation unit.
 */

#include "solve.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <vector>

#define CUDA_CHECK_BOOL(expr)                                                  \
    do {                                                                       \
        cudaError_t err__ = (expr);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error: %s (%s:%d)\n",                   \
                         cudaGetErrorString(err__), __FILE__, __LINE__);       \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace {

// ---------- Goldilocks modular arithmetic (host-only reference, file-local) ----------

constexpr uint64_t P = NTT_PRIME;  // 2^64 - 2^32 + 1

static inline uint64_t mod_add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    if (s < a || s >= P) {
        s -= P;
    }
    return s;
}

static inline uint64_t mod_sub(uint64_t a, uint64_t b) {
    return (a >= b) ? (a - b) : (a + (P - b));
}

static inline uint64_t mod_mul(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * (__uint128_t)b;
    return (uint64_t)(prod % (__uint128_t)P);
}

static inline uint64_t mod_pow(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    uint64_t b = base % P;
    while (exp > 0) {
        if (exp & 1ULL) {
            result = mod_mul(result, b);
        }
        b = mod_mul(b, b);
        exp >>= 1;
    }
    return result;
}

// Primitive 2^32-th root of unity in F_p (p = Goldilocks). Plonky2 uses g = 7 as the multiplicative
// generator; g^((p-1)/2^32) is then a primitive 2^32-th root of unity.
static inline uint64_t primitive_root_of_unity(int log_n) {
    // (p - 1) / 2^32 = (2^64 - 2^32) / 2^32 = 2^32 - 1.
    uint64_t omega32 = mod_pow(7ULL, (uint64_t)((__uint128_t)1 << 32) - 1ULL);
    // omega_n = omega32^(2^(32 - log_n))
    int shift = 32 - log_n;
    for (int i = 0; i < shift; ++i) {
        omega32 = mod_mul(omega32, omega32);
    }
    return omega32;
}

// Iterative natural-order forward Cooley-Tukey NTT, in place.
// Pattern: bit-reverse the input, then do log_n butterfly stages.
static inline uint32_t bit_reverse(uint32_t x, int bits) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = (x >> 16) | (x << 16);
    return x >> (32 - bits);
}

static void reference_ntt(uint64_t* a, int n) {
    int log_n = 0;
    while ((1 << log_n) < n) ++log_n;

    for (int i = 0; i < n; ++i) {
        uint32_t j = bit_reverse((uint32_t)i, log_n);
        if ((int)j > i) {
            uint64_t t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    for (int s = 1; s <= log_n; ++s) {
        int m = 1 << s;
        int half = m >> 1;
        uint64_t omega_m = primitive_root_of_unity(s);
        for (int k = 0; k < n; k += m) {
            uint64_t w = 1;
            for (int j = 0; j < half; ++j) {
                uint64_t u = a[k + j];
                uint64_t v = mod_mul(a[k + j + half], w);
                a[k + j] = mod_add(u, v);
                a[k + j + half] = mod_sub(u, v);
                w = mod_mul(w, omega_m);
            }
        }
    }
}

// ---------- splitmix64 PRNG (deterministic per seed) ----------

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

// Fill a row's input deterministically from (seed, row_index). Values reduced mod P.
static void fill_row(uint64_t* row, int n, uint64_t seed, int row_idx) {
    SplitMix64 rng(seed ^ ((uint64_t)(row_idx + 1) * 0x9e3779b97f4a7c15ULL));
    for (int i = 0; i < n; ++i) {
        uint64_t v = rng.next();
        if (v >= P) v -= P;  // single-step reduction is enough since P > 2^63
        row[i] = v;
    }
}

// ---------- harness data ----------

struct Shape {
    const char* name;
    int batch;
    int n;
};

struct CaseData {
    Shape shape;
    std::vector<uint64_t> input;     // batch * n
    std::vector<uint64_t> expected;  // batch * n
};

struct DeviceBuffers {
    uint64_t* d_data = nullptr;
    void* d_workspace = nullptr;
    size_t workspace_bytes = 0;
    cudaStream_t stream = nullptr;
};

static size_t workspace_bytes_for(const Shape& s) {
    // Generous pre-allocation: 64 MiB plus 64 bytes per element for any scratch / twiddle tables.
    size_t per_element = 64;
    size_t base = (size_t)64 * 1024 * 1024;
    return base + (size_t)s.batch * (size_t)s.n * per_element;
}

static bool generate_case(CaseData& data, const Shape& shape, uint64_t seed) {
    data.shape = shape;
    size_t total = (size_t)shape.batch * (size_t)shape.n;
    data.input.assign(total, 0);
    data.expected.assign(total, 0);
    for (int row = 0; row < shape.batch; ++row) {
        uint64_t* row_in = data.input.data() + (size_t)row * (size_t)shape.n;
        uint64_t* row_out = data.expected.data() + (size_t)row * (size_t)shape.n;
        fill_row(row_in, shape.n, seed, row);
        std::memcpy(row_out, row_in, (size_t)shape.n * sizeof(uint64_t));
        reference_ntt(row_out, shape.n);
    }
    return true;
}

static void release_device(DeviceBuffers& dev) {
    if (dev.d_data) cudaFree(dev.d_data);
    if (dev.d_workspace) cudaFree(dev.d_workspace);
    if (dev.stream) cudaStreamDestroy(dev.stream);
    dev = DeviceBuffers{};
}

static bool upload_case(const CaseData& data, DeviceBuffers& dev) {
    const Shape& s = data.shape;
    dev.workspace_bytes = workspace_bytes_for(s);
    CUDA_CHECK_BOOL(cudaStreamCreate(&dev.stream));
    CUDA_CHECK_BOOL(cudaMalloc(&dev.d_data, data.input.size() * sizeof(uint64_t)));
    CUDA_CHECK_BOOL(cudaMalloc(&dev.d_workspace, dev.workspace_bytes));
    return true;
}

static bool reset_input(const CaseData& data, DeviceBuffers& dev) {
    CUDA_CHECK_BOOL(cudaMemcpyAsync(dev.d_data, data.input.data(),
                                    data.input.size() * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, dev.stream));
    CUDA_CHECK_BOOL(cudaStreamSynchronize(dev.stream));
    return true;
}

static void call_solution(const CaseData& data, DeviceBuffers& dev) {
    ntt_forward_cuda(dev.d_data,
                     dev.d_workspace,
                     dev.workspace_bytes,
                     data.shape.batch,
                     data.shape.n,
                     dev.stream);
}

static bool fetch_output(const CaseData& data, DeviceBuffers& dev,
                         std::vector<uint64_t>& got) {
    got.assign(data.input.size(), 0);
    CUDA_CHECK_BOOL(cudaMemcpy(got.data(), dev.d_data,
                               got.size() * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost));
    return true;
}

static bool full_compare(const CaseData& data, const std::vector<uint64_t>& got,
                         const char* phase) {
    int mismatches = 0;
    for (size_t i = 0; i < data.expected.size(); ++i) {
        if (got[i] != data.expected[i]) {
            if (mismatches < 8) {
                int row = (int)(i / (size_t)data.shape.n);
                int idx = (int)(i - (size_t)row * (size_t)data.shape.n);
                std::fprintf(stderr,
                             "%s mismatch row=%d idx=%d got=%llu expected=%llu\n",
                             phase, row, idx,
                             (unsigned long long)got[i],
                             (unsigned long long)data.expected[i]);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "%s mismatches=%d total=%zu\n",
                     phase, mismatches, data.expected.size());
        return false;
    }
    return true;
}

// Late-slice probe: pick a hidden subset of (row, idx) coordinates from the timed batch and verify
// bit-exact equality against the in-binary reference. Coordinates are derived from the seed and
// a build-time-baked salt so the agent cannot predict them. This catches "fake-after-N" or
// row-skip attacks where the agent passes the prefix check but corrupts the deep workload.
static bool late_slice_probe(const CaseData& data, const std::vector<uint64_t>& got,
                             uint64_t seed) {
    // Derive a hidden offset within the batch and a hidden stride.
    SplitMix64 rng(seed ^ 0x6e74745f70726f62ULL ^ 0xC0DECAFEDEADBEEFULL);
    int batch = data.shape.batch;
    int n = data.shape.n;
    int n_probes = 256;  // small enough to be cheap, large enough to catch sloppy outputs
    if (n_probes > batch * n) n_probes = batch * n;

    int mismatches = 0;
    for (int p = 0; p < n_probes; ++p) {
        uint64_t r = rng.next();
        // Bias probes toward the back half / late slice of the batch.
        int row = (int)((r >> 32) % (uint64_t)batch);
        int idx = (int)((uint32_t)r % (uint32_t)n);
        // Force at least half the probes into the second half of rows.
        if ((p & 1) == 0) {
            row = (batch / 2) + (row % (batch - batch / 2));
        }
        size_t flat = (size_t)row * (size_t)n + (size_t)idx;
        if (got[flat] != data.expected[flat]) {
            if (mismatches < 8) {
                std::fprintf(stderr,
                             "late-slice mismatch row=%d idx=%d got=%llu expected=%llu\n",
                             row, idx,
                             (unsigned long long)got[flat],
                             (unsigned long long)data.expected[flat]);
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::fprintf(stderr, "late-slice mismatches=%d probes=%d\n",
                     mismatches, n_probes);
        return false;
    }
    return true;
}

// ---------- run modes ----------

static constexpr int N_WARMUP = 2;
static constexpr int N_TIMED = 7;

static bool run_correctness_case(const Shape& shape, uint64_t seed) {
    CaseData data;
    if (!generate_case(data, shape, seed)) return false;

    DeviceBuffers dev;
    if (!upload_case(data, dev)) {
        release_device(dev);
        return false;
    }
    if (!reset_input(data, dev)) {
        release_device(dev);
        return false;
    }
    call_solution(data, dev);
    CUDA_CHECK_BOOL(cudaStreamSynchronize(dev.stream));
    CUDA_CHECK_BOOL(cudaGetLastError());

    std::vector<uint64_t> got;
    if (!fetch_output(data, dev, got)) {
        release_device(dev);
        return false;
    }
    bool ok = full_compare(data, got, shape.name);
    release_device(dev);
    return ok;
}

static bool run_timed_case(const Shape& shape, uint64_t seed, double* median_ms) {
    CaseData data;
    if (!generate_case(data, shape, seed)) return false;

    DeviceBuffers dev;
    if (!upload_case(data, dev)) {
        release_device(dev);
        return false;
    }

    // Warmup -- input must be reset between runs because solve mutates in place.
    for (int i = 0; i < N_WARMUP; ++i) {
        if (!reset_input(data, dev)) {
            release_device(dev);
            return false;
        }
        call_solution(data, dev);
        CUDA_CHECK_BOOL(cudaStreamSynchronize(dev.stream));
        CUDA_CHECK_BOOL(cudaGetLastError());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK_BOOL(cudaEventCreate(&start));
    CUDA_CHECK_BOOL(cudaEventCreate(&stop));

    std::vector<double> times;
    times.reserve(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        if (!reset_input(data, dev)) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            release_device(dev);
            return false;
        }
        CUDA_CHECK_BOOL(cudaStreamSynchronize(dev.stream));
        CUDA_CHECK_BOOL(cudaEventRecord(start, dev.stream));
        call_solution(data, dev);
        CUDA_CHECK_BOOL(cudaEventRecord(stop, dev.stream));
        CUDA_CHECK_BOOL(cudaEventSynchronize(stop));
        CUDA_CHECK_BOOL(cudaGetLastError());
        float ms = 0.f;
        CUDA_CHECK_BOOL(cudaEventElapsedTime(&ms, start, stop));
        times.push_back((double)ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Output of the LAST timed run is what we check (agent could have produced different outputs
    // on different runs; we score the actual final output). Full + late-slice probe.
    std::vector<uint64_t> got;
    if (!fetch_output(data, dev, got)) {
        release_device(dev);
        return false;
    }
    bool ok_full = full_compare(data, got, shape.name);
    bool ok_late = late_slice_probe(data, got, seed);
    release_device(dev);

    if (!ok_full || !ok_late) return false;

    std::sort(times.begin(), times.end());
    *median_ms = times[times.size() / 2];
    return true;
}

static Shape scored_shape() {
    return {"scored", 256, 1 << 16};
}

}  // namespace

int main(int argc, char** argv) {
    const char* mode = (argc >= 2) ? argv[1] : "--benchmark";

    if (std::strcmp(mode, "--verify") == 0) {
        // Public, small, runtime-random seed so the agent cannot hardcode expected output.
        uint64_t seed = (uint64_t)time(nullptr) ^ ((uint64_t)getpid() << 32) ^ 0xC0FFEE123456ULL;
        const Shape cases[] = {
            {"verify-tiny", 2, 256},
            {"verify-small", 4, 1024},
            {"verify-medium", 8, 4096},
        };
        bool ok = true;
        for (int i = 0; i < 3; ++i) {
            uint64_t row_seed = seed ^ ((uint64_t)(i + 1) * 0x9e3779b97f4a7c15ULL);
            ok = run_correctness_case(cases[i], row_seed) && ok;
        }
        if (ok) {
            std::printf("PASS verify=ok cases=3\n");
            std::fprintf(stderr, "__VERIFIER_CORRECTNESS__=PASS\n");
            return 0;
        }
        std::printf("FAIL verify\n");
        return 1;
    }

    if (std::strcmp(mode, "--benchmark") == 0) {
        // Public, fixed seed -- unscored sanity benchmark for the agent's iteration loop.
        Shape s = scored_shape();
        double median_ms = 0.0;
        if (!run_timed_case(s, 0xDEADBEEFCAFEBABEULL, &median_ms)) {
            std::printf("FAIL benchmark\n");
            return 1;
        }
        std::printf("result=ok time_ms=%.6f batch=%d n=%d\n", median_ms, s.batch, s.n);
        return 0;
    }

    if (std::strcmp(mode, "--benchmark-verify") == 0) {
        // Hidden seed -- comes from env var that test.sh sets right before launch.
        const char* sv = std::getenv("NTT_BENCH_SEED");
        if (!sv || !*sv) {
            std::fprintf(stderr, "ERROR: --benchmark-verify requires NTT_BENCH_SEED env\n");
            return 2;
        }
        uint64_t seed = std::strtoull(sv, nullptr, 0);

        Shape s = scored_shape();
        double median_ms = 0.0;
        if (!run_timed_case(s, seed, &median_ms)) {
            std::printf("FAIL benchmark-verify\n");
            return 1;
        }
        std::printf("result=ok batch=%d n=%d median_ms=%.6f\n", s.batch, s.n, median_ms);
        std::fprintf(stderr, "__VERIFIER_CORRECTNESS__=PASS\n");
        std::fprintf(stderr, "__VERIFIER_BENCHMARK__=PASS\n");
        std::fprintf(stderr, "__VERIFIER_SCORE__=%.6f\n", median_ms);
        return 0;
    }

    std::fprintf(stderr, "usage: %s [--verify|--benchmark|--benchmark-verify]\n", argv[0]);
    return 2;
}
