// solve.h — public interface shared by main.cpp and solve.cpp.
//
// DO NOT MODIFY THIS FILE.
//
// This header defines the data types and the two functions you must implement
// in solve.cpp:
//
//   Build(tris, N)    — called once; may build any acceleration structure
//   Intersect(ray)    — called per-ray; must set ray.t to closest hit
//
// You may add #includes and helper declarations in solve.cpp, but the
// signatures of Build() and Intersect() must not change.

#pragma once
#include <cmath>
#include <algorithm>

// ─── 3-component float vector ─────────────────────────────────────────────────

struct float3 {
    float x, y, z;
    float3() : x(0), y(0), z(0) {}
    float3(float a, float b, float c) : x(a), y(b), z(c) {}
    float& operator[](int i)       { return (&x)[i]; }
    float  operator[](int i) const { return (&x)[i]; }
};

inline float3 operator+(const float3& a, const float3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
inline float3 operator-(const float3& a, const float3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline float3 operator*(const float3& a, float s)         { return {a.x*s,   a.y*s,   a.z*s};   }
inline float3 operator*(float s, const float3& a)         { return a * s; }
inline float  dot(const float3& a, const float3& b)       { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline float3 cross(const float3& a, const float3& b) {
    return { a.y*b.z - a.z*b.y,
             a.z*b.x - a.x*b.z,
             a.x*b.y - a.y*b.x };
}
inline float3 fminf3(const float3& a, const float3& b) {
    return { std::min(a.x,b.x), std::min(a.y,b.y), std::min(a.z,b.z) };
}
inline float3 fmaxf3(const float3& a, const float3& b) {
    return { std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z) };
}

// ─── Scene primitives ─────────────────────────────────────────────────────────

// Triangle with precomputed centroid (useful for BVH construction).
// vertex0/1/2 are the three corners; centroid = (v0+v1+v2)/3.
struct Tri {
    float3 vertex0, vertex1, vertex2, centroid;
};

// A ray.  Callers initialize t = 1e30f (meaning "no hit yet").
// Intersect() updates ray.t when it finds a closer hit.
struct Ray {
    float3 O, D;   // origin, direction (unit length)
    float  t;      // closest hit distance so far; 1e30f = miss
};

// ─── Interface you must implement ─────────────────────────────────────────────

// Called exactly once, before any Intersect() calls.
// tris[0..N-1] is the full triangle array (you may reorder it).
// Use this to build any acceleration structure you like.
void Build(Tri* tris, int N);

// Test ray against the scene.  Updates ray.t with the closest intersection
// distance if one is found.  Returns true if any hit was found.
// Must produce correct results for any valid ray.
bool Intersect(Ray& ray);
