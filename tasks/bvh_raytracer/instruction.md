# BVH Ray Tracer

Optimize `Intersect()` in `/app/solve.cpp` to be as fast as possible.

## Problem

A primary ray tracer fires one ray per pixel through a **638×638** camera against a scene of **4096 triangles**. The baseline tests all 4096 triangles per ray (1.67 billion tests per render). Build a BVH in `Build()` to reduce this to O(log N) per ray.

Interface (must not change):

```cpp
void Build(Tri* tris, int N);  // called once before rendering; tris is writable
bool Intersect(Ray& ray);      // update ray.t with nearest hit; return true if hit
```

`Build()` is not timed — only `Intersect()` is benchmarked (median of 5 full renders).

## Files

| Path | Permission |
|------|-----------|
| `/app/solve.cpp` | **Edit** |
| `/app/main.cpp` | Read-only |
| `/app/solve.h` | Read-only |
| `/app/Makefile` | Read-only |

## Rules

- Do not modify `Makefile`. Build configuration is fixed.
- Edit `/app/solve.cpp` only.
- Standard library and SIMD intrinsics (`<immintrin.h>`) are allowed.
- No external libraries. No multithreading.
- `ray.t` must be correct — errors > 0.01 units score 0.
- Time budget: 2 hours
