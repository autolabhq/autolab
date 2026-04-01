// main.cpp — benchmark driver and verifier.
//
// DO NOT MODIFY THIS FILE.
//
// Scene: N_TRI random triangles in the unit cube, seeded for reproducibility.
// Camera: origin at (0.5, 0.5, -2), looking toward +z (at the scene).
// Render: W×H primary rays, one per pixel.
//
// Usage:
//   ./solve                     → benchmark mode: 5 renders, print median seconds
//   ./solve --verify            → verify mode: print checksum (for diff testing)

#include "solve.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <chrono>

// ─── Scene parameters ─────────────────────────────────────────────────────────
static const int N_TRI = 4096;   // number of random triangles
static const int W     = 638;    // image width  (not a power-of-2 to stress BVH)
static const int H     = 638;    // image height

// ─── Reproducible PRNG (xorshift32, fixed seed) ───────────────────────────────
static uint32_t rng_state;
static float rng() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    // Map to [0, 1)
    return (rng_state >> 8) * (1.0f / (1 << 24));
}
static float3 rng3() { return { rng(), rng(), rng() }; }

// ─── Camera ───────────────────────────────────────────────────────────────────
static const float3 CAM_O = { 0.5f, 0.5f, -2.0f };

static Ray MakeRay(int px, int py) {
    // Map pixel to a point on the scene face z=0 (the near face of the unit cube)
    float u = (px + 0.5f) / W;   // [0, 1]
    float v = (py + 0.5f) / H;   // [0, 1]
    float3 target = { u, v, 0.5f };
    float3 d = target - CAM_O;
    float  len = std::sqrt(dot(d, d));
    Ray ray;
    ray.O = CAM_O;
    ray.D = d * (1.f / len);
    ray.t = 1e30f;
    return ray;
}

// ─── Scene construction ───────────────────────────────────────────────────────
static Tri g_scene[N_TRI];

static void BuildScene() {
    rng_state = 0x12345678u;   // fixed seed — do not change
    for (int i = 0; i < N_TRI; i++) {
        float3 v0 = rng3();
        float3 v1 = v0 + rng3() * 0.1f;
        float3 v2 = v0 + rng3() * 0.1f;
        g_scene[i] = { v0, v1, v2, (v0 + v1 + v2) * (1.f / 3.f) };
    }
}

// ─── Checksum ─────────────────────────────────────────────────────────────────
// XOR of (bucket(t) XOR pixel_index_hash) for all pixels.
// bucket: encode t to 2 decimal places (stable across scalar vs SIMD rounding).
static uint64_t ComputeChecksum() {
    static const uint64_t SALT = 0x9E3779B97F4A7C15ULL;
    uint64_t csum = 0;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            Ray ray = MakeRay(x, y);
            Intersect(ray);
            uint32_t bucket = (ray.t < 1e30f)
                ? static_cast<uint32_t>(ray.t * 100.f)
                : 0xFFFFFFFFu;
            csum ^= static_cast<uint64_t>(bucket) ^ (static_cast<uint64_t>(y * W + x) * SALT);
        }
    }
    return csum;
}

// ─── Full render (benchmark payload) ─────────────────────────────────────────
static uint64_t Render() {
    uint64_t hits = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            Ray ray = MakeRay(x, y);
            if (Intersect(ray)) hits++;
        }
    return hits;
}

// ─── Entry point ─────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    bool verify = (argc > 1 && strcmp(argv[1], "--verify") == 0);

    BuildScene();
    Build(g_scene, N_TRI);

    if (verify) {
        uint64_t cs = ComputeChecksum();
        printf("checksum=%llu\n", (unsigned long long)cs);
        return 0;
    }

    // ── Benchmark: 5 full renders, report median ──────────────────────────────
    double times[5];
    for (int run = 0; run < 5; run++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t hits = Render();
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        times[run] = elapsed;
        fprintf(stderr, "run %d: %.3fs  hits=%llu\n",
                run + 1, elapsed, (unsigned long long)hits);
    }
    std::sort(times, times + 5);
    printf("%.6f\n", times[2]);   // median → consumed by test.sh
    return 0;
}
