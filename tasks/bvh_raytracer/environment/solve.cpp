// solve.cpp — OPTIMIZE THIS FILE.
//
// Current implementation: brute-force O(N) ray–triangle intersection.
// Every Intersect() call tests the ray against all N triangles in sequence.
//
// Rules:
//   - Optimize this file only.
//   - Do NOT modify main.cpp, solve.h, or the Makefile.
//   - Do NOT modify anything under /tests/.
//   - C++ standard library is available; SIMD intrinsics (<immintrin.h>) are allowed.
//   - No multithreading (the benchmark is single-threaded).
//   - No new external libraries or build system changes.
//   - Intersect() must produce correct results — wrong t values score 0.

#include "solve.h"

static Tri* g_tris;
static int  g_N;

// Möller–Trumbore ray–triangle intersection.
// If the ray intersects the triangle at distance t > 0, updates ray.t
// with the smaller of the current ray.t and t.
static inline void IntersectTri(Ray& ray, const Tri& tri) {
    const float3 edge1 = tri.vertex1 - tri.vertex0;
    const float3 edge2 = tri.vertex2 - tri.vertex0;
    const float3 h     = cross(ray.D, edge2);
    const float  a     = dot(edge1, h);
    if (a > -1e-7f && a < 1e-7f) return;   // ray parallel to triangle
    const float  f     = 1.f / a;
    const float3 s     = ray.O - tri.vertex0;
    const float  u     = f * dot(s, h);
    if (u < 0.f || u > 1.f) return;
    const float3 q     = cross(s, edge1);
    const float  v     = f * dot(ray.D, q);
    if (v < 0.f || u + v > 1.f) return;
    const float  t     = f * dot(edge2, q);
    if (t > 1e-7f) ray.t = std::min(ray.t, t);
}

// Build: nothing to do for brute-force; just store the triangle array.
void Build(Tri* tris, int N) {
    g_tris = tris;
    g_N    = N;
}

// Intersect: test ray against every triangle — O(N) per ray.
bool Intersect(Ray& ray) {
    for (int i = 0; i < g_N; i++)
        IntersectTri(ray, g_tris[i]);
    return ray.t < 1e30f;
}
