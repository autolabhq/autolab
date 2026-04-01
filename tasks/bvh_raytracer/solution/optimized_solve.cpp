// optimized_solve.cpp — reference solution for bvh_raytracer.
//
// Implements a Bounding Volume Hierarchy (BVH) with the Surface Area Heuristic
// (SAH) for split selection.  Key improvements over the baseline:
//
//  Level 1 — BVH with midpoint split:
//    Replace the O(N) per-ray brute-force loop with a BVH traversal.  Each ray
//    visits only O(log N) nodes instead of all N triangles.  The midpoint split
//    divides the longest axis at the centre of the bounding box — fast to build
//    but leaves unbalanced trees on clustered geometry.
//    Expected speedup: ~37× over brute-force for N=4096.
//
//  Level 2 — SAH split selection:
//    Instead of splitting at the bounding-box midpoint, evaluate all candidate
//    split planes and choose the one minimising the Surface Area Heuristic cost:
//
//        SAH = (left_count * left_area + right_count * right_area) / parent_area
//
//    "Binned SAH" (B=8 bins per axis) keeps build time at O(B×N log N).
//    SAH trees typically reduce the average nodes visited per ray by 30–50%.
//    Expected additional speedup: ~1.5–2× over midpoint BVH.
//
//  Level 3 (not shown here) — 4-wide SIMD AABB test:
//    Pack four BVH node AABBs into SSE/AVX registers and test all four against
//    the ray in one pass.  Eliminates most branch overhead in the traversal hot
//    loop.  See Wald, Boulos & Shirley (2007) "Ray Tracing Deformable Scenes".
//    Expected additional speedup: ~2× over scalar SAH BVH.

#include "solve.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cfloat>

// ─── BVH node (32 bytes — half a cache line) ──────────────────────────────────
struct BVHNode {
    float3 aabbMin, aabbMax;   // 24 bytes
    int    leftFirst;          // leaf: first tri in g_idx; internal: left child index
    int    triCount;           // 0 = internal node, >0 = leaf
    bool   isLeaf() const { return triCount > 0; }
};

static Tri*     g_tris;
static int      g_N;
static int*     g_idx;        // index permutation — reordered by BVH construction
static BVHNode* g_bvh;
static int      g_nodeCount;

// ─── AABB operations ─────────────────────────────────────────────────────────

static inline float HalfArea(const float3& mn, const float3& mx) {
    float3 e = mx - mn;
    return e.x * e.y + e.y * e.z + e.z * e.x;
}

// Slab-method AABB ray intersection.
// Returns nearest hit distance, or 1e30f if no intersection in (0, ray.t).
static inline float IntersectAABB(const Ray& ray,
                                   const float3& bmin, const float3& bmax) {
    float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
    float tmin = std::min(tx1, tx2), tmax = std::max(tx1, tx2);
    float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
    tmin = std::max(tmin, std::min(ty1, ty2));
    tmax = std::min(tmax, std::max(ty1, ty2));
    float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
    tmin = std::max(tmin, std::min(tz1, tz2));
    tmax = std::min(tmax, std::max(tz1, tz2));
    return (tmax >= tmin && tmax > 0.f && tmin < ray.t) ? tmin : 1e30f;
}

// ─── Ray–triangle intersection (Möller–Trumbore) ─────────────────────────────
static inline void IntersectTri(Ray& ray, const Tri& tri) {
    const float3 edge1 = tri.vertex1 - tri.vertex0;
    const float3 edge2 = tri.vertex2 - tri.vertex0;
    const float3 h = cross(ray.D, edge2);
    const float  a = dot(edge1, h);
    if (a > -1e-7f && a < 1e-7f) return;
    const float  f = 1.f / a;
    const float3 s = ray.O - tri.vertex0;
    const float  u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return;
    const float3 q = cross(s, edge1);
    const float  v = f * dot(ray.D, q);
    if (v < 0.f || u + v > 1.f) return;
    const float  t = f * dot(edge2, q);
    if (t > 1e-7f) ray.t = std::min(ray.t, t);
}

// ─── BVH construction helpers ─────────────────────────────────────────────────

static void RecomputeBounds(int nodeIdx) {
    BVHNode& n = g_bvh[nodeIdx];
    n.aabbMin = { 1e30f,  1e30f,  1e30f };
    n.aabbMax = {-1e30f, -1e30f, -1e30f };
    for (int i = n.leftFirst; i < n.leftFirst + n.triCount; i++) {
        const Tri& t = g_tris[g_idx[i]];
        n.aabbMin = fminf3(n.aabbMin, fminf3(t.vertex0, fminf3(t.vertex1, t.vertex2)));
        n.aabbMax = fmaxf3(n.aabbMax, fmaxf3(t.vertex0, fmaxf3(t.vertex1, t.vertex2)));
    }
}

// Binned-SAH split: divide the axis range into BINS equal bins; sweep to find
// the cheapest split.  O(BINS × triCount) per call.
static const int BINS = 8;

struct Bin { float3 mn, mx; int count; };

static void Subdivide(int nodeIdx) {
    BVHNode& node = g_bvh[nodeIdx];
    if (node.triCount <= 2) return;

    float3 extent = node.aabbMax - node.aabbMin;
    float  parentArea = HalfArea(node.aabbMin, node.aabbMax);
    float  leafCost   = (float)node.triCount;   // cost of keeping as leaf
    float  bestCost   = leafCost;
    int    bestAxis   = -1;
    float  bestSplit  = 0.f;

    for (int axis = 0; axis < 3; axis++) {
        if (extent[axis] < 1e-6f) continue;

        float scale = BINS / extent[axis];
        Bin bins[BINS];
        for (auto& b : bins) {
            b.mn = { 1e30f, 1e30f, 1e30f };
            b.mx = {-1e30f,-1e30f,-1e30f };
            b.count = 0;
        }

        // Bin the triangles.
        for (int i = node.leftFirst; i < node.leftFirst + node.triCount; i++) {
            const Tri& tri = g_tris[g_idx[i]];
            int binIdx = std::min(BINS - 1, (int)((tri.centroid[axis] - node.aabbMin[axis]) * scale));
            Bin& b = bins[binIdx];
            b.count++;
            b.mn = fminf3(b.mn, fminf3(tri.vertex0, fminf3(tri.vertex1, tri.vertex2)));
            b.mx = fmaxf3(b.mx, fmaxf3(tri.vertex0, fmaxf3(tri.vertex1, tri.vertex2)));
        }

        // Left-to-right prefix: grow left child AABB and count.
        float3 lmn[BINS-1], lmx[BINS-1];
        int    lcnt[BINS-1];
        float3 mn = { 1e30f, 1e30f, 1e30f }, mx = {-1e30f,-1e30f,-1e30f };
        int cnt = 0;
        for (int i = 0; i < BINS - 1; i++) {
            cnt += bins[i].count;
            mn = fminf3(mn, bins[i].mn);
            mx = fmaxf3(mx, bins[i].mx);
            lmn[i] = mn; lmx[i] = mx; lcnt[i] = cnt;
        }

        // Right-to-left prefix: evaluate SAH at each candidate boundary.
        mn = { 1e30f, 1e30f, 1e30f }; mx = {-1e30f,-1e30f,-1e30f }; cnt = 0;
        for (int i = BINS - 2; i >= 0; i--) {
            cnt += bins[i + 1].count;
            mn = fminf3(mn, bins[i+1].mn);
            mx = fmaxf3(mx, bins[i+1].mx);
            float cost = (lcnt[i] * HalfArea(lmn[i], lmx[i]) +
                          cnt      * HalfArea(mn, mx)) / parentArea;
            if (cost < bestCost) {
                bestCost  = cost;
                bestAxis  = axis;
                bestSplit = node.aabbMin[axis] + (i + 1) * extent[axis] / BINS;
            }
        }
    }

    if (bestAxis < 0) return;  // no split beats the leaf cost

    // Partition triangles around bestSplit on bestAxis.
    int i = node.leftFirst;
    int j = i + node.triCount - 1;
    while (i <= j) {
        if (g_tris[g_idx[i]].centroid[bestAxis] < bestSplit) i++;
        else std::swap(g_idx[i], g_idx[j--]);
    }
    int leftCount = i - node.leftFirst;
    if (leftCount == 0 || leftCount == node.triCount) return;

    // Allocate child nodes.
    int leftIdx  = g_nodeCount++;
    int rightIdx = g_nodeCount++;
    g_bvh[leftIdx]  = BVHNode{ {},{}, node.leftFirst,             leftCount              };
    g_bvh[rightIdx] = BVHNode{ {},{}, i,                          node.triCount-leftCount };
    node.leftFirst   = leftIdx;
    node.triCount    = 0;

    RecomputeBounds(leftIdx);
    RecomputeBounds(rightIdx);
    Subdivide(leftIdx);
    Subdivide(rightIdx);
}

// ─── BVH traversal ───────────────────────────────────────────────────────────
static void TraverseBVH(Ray& ray, int nodeIdx) {
    const BVHNode& node = g_bvh[nodeIdx];
    if (IntersectAABB(ray, node.aabbMin, node.aabbMax) >= ray.t) return;
    if (node.isLeaf()) {
        for (int i = node.leftFirst; i < node.leftFirst + node.triCount; i++)
            IntersectTri(ray, g_tris[g_idx[i]]);
    } else {
        TraverseBVH(ray, node.leftFirst);
        TraverseBVH(ray, node.leftFirst + 1);
    }
}

// ─── Public interface ─────────────────────────────────────────────────────────

void Build(Tri* tris, int N) {
    g_tris = tris;
    g_N    = N;
    g_idx  = new int[N];
    for (int i = 0; i < N; i++) g_idx[i] = i;
    g_bvh  = new BVHNode[2 * N];
    g_nodeCount = 1;
    g_bvh[0] = BVHNode{ {},{}, 0, N };
    RecomputeBounds(0);
    Subdivide(0);
}

bool Intersect(Ray& ray) {
    TraverseBVH(ray, 0);
    return ray.t < 1e30f;
}
