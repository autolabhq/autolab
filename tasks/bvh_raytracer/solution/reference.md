# BVH Ray Tracer — Reference

## Background
BVH-accelerated ray tracing is the foundational acceleration structure in modern renderers (path tracers, game engines, ray-tracing GPUs). Without spatial indexing, rendering cost scales as O(N) triangle tests per ray; a BVH reduces this to O(log N) by culling entire subtrees whose bounding boxes miss the ray. At 638×638 pixels and 4096 triangles, the brute-force baseline must execute ~1.67 billion Möller–Trumbore tests per frame.

## Baseline Approach
The unoptimized implementation calls `IntersectTri()` for every triangle on every ray — 4096 triangle tests per ray with no spatial culling. On a 638×638 image this yields ~406 million rays × 4096 tests = ~1.67 billion intersections per render, running in ~3.8 seconds.

## Possible Optimization Directions
1. **BVH with midpoint split** — Build a binary tree partitioning triangles by the midpoint of the longest AABB axis; each ray visits only O(log N) ≈ 24 nodes instead of 4096 triangles. Expected speedup: ~37× (~0.10s).
2. **Binned Surface Area Heuristic (SAH) split** — Replace midpoint split with the split plane minimising `(left_count × left_area + right_count × right_area) / parent_area` evaluated over B=8–16 bins per axis. Reduces average nodes visited per ray by 30–50% vs midpoint. Expected additional speedup: ~1.5–2× over midpoint BVH.
3. **4-wide SIMD AABB tests** — Pack four BVH node AABBs into SSE/AVX registers and test all four against the ray simultaneously, eliminating branch overhead in the traversal hot loop. Expected additional speedup: ~2–3× over scalar SAH BVH.
4. **Precomputed ray direction reciprocals** — Cache `1/ray.D.x`, `1/ray.D.y`, `1/ray.D.z` to replace divisions with multiplications in the slab-method AABB test.
5. **Ordered traversal** — Visit the nearer child first and skip the farther child if the nearer hit already updates `ray.t` below the farther AABB's entry distance.

## Reference Solution
Implements a BVH with binned SAH split selection (8 bins per axis, evaluated over all three axes). Construction reorders a permutation index array `g_idx[]` rather than moving triangles, so leaves point into contiguous index ranges. Traversal uses the slab-method AABB test (`IntersectAABB`) to prune subtrees early; leaf nodes run Möller–Trumbore per triangle. The SAH evaluator compares split cost against leaf cost and skips the split when no improvement is found.

## Source
- Jacco Bikker, *How to Build a BVH* tutorial series, Parts 1–5 (2022)
- Kay & Kajiya, *Rendering Complex Scenes with Memory-Coherent Ray Tracing* (1986) — slab-method AABB intersection
- Goldsmith & Salmon, *Automatic Creation of Object Hierarchies for Ray Tracing* (1987) — Surface Area Heuristic
- Wald, Boulos & Shirley, *Ray Tracing Deformable Scenes Using Dynamic BVHs* (2007) — SIMD packet traversal
- Pharr, Jakob & Humphreys, *Physically Based Rendering* (4th ed.), §7.3
