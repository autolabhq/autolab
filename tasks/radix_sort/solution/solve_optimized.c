/*
 * solve_optimized.c — reference solution for radix_sort.
 *
 * Technique: 2-pass LSD radix sort with 16-bit digit width.
 *
 * Key optimizations over the naive qsort baseline:
 *
 * 1. O(n) algorithm  — 2 passes × n scatter operations instead of
 *    ~n log₂ n comparisons (50M × 26 comparisons ≈ 1.3B for qsort).
 *
 * 2. 16-bit digit width  — only 2 passes instead of 8 (for 8-bit digits).
 *    Histogram table fits in 65536 × 8 bytes = 512 KB, which is warm in
 *    L2/L3 for the scatter phase.
 *
 * 3. Single-scan histogram  — count both lo-16 and hi-16 histograms in
 *    one forward scan, halving memory-bandwidth for the counting step.
 *
 * 4. Non-aliasing buffers (restrict) — allows the compiler to generate
 *    more aggressive load/store reordering in the scatter loops.
 *
 * 5. Software prefetch  — issue write prefetches 32 elements ahead during
 *    the scatter phase to partially hide write-miss latency.
 *
 * Achievable speedup over qsort baseline: ~8–12× on a 2-core machine.
 */

#include "solve.h"
#include <string.h>

#define RADIX_BITS  16u
#define BINS        (1u << RADIX_BITS)   /* 65536 */
#define MASK        (BINS - 1u)
#define PREFETCH_DIST 32

/* Histogram tables — static to keep them off the stack (512 KB each). */
static size_t cnt_lo[BINS];
static size_t cnt_hi[BINS];

void radix_sort(uint32_t *arr, size_t n)
{
    if (n < 2) return;

    uint32_t *tmp = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!tmp) {
        /* Fallback: insertion sort for very small n (should not happen in benchmark). */
        for (size_t i = 1; i < n; i++) {
            uint32_t key = arr[i];
            size_t j = i;
            while (j > 0 && arr[j-1] > key) { arr[j] = arr[j-1]; j--; }
            arr[j] = key;
        }
        return;
    }

    /* ── Step 1: Count both histograms in a single forward scan ─────────── */
    memset(cnt_lo, 0, BINS * sizeof(size_t));
    memset(cnt_hi, 0, BINS * sizeof(size_t));
    for (size_t i = 0; i < n; i++) {
        uint32_t v = arr[i];
        cnt_lo[v & MASK]++;
        cnt_hi[v >> RADIX_BITS]++;
    }

    /* ── Step 2: Exclusive prefix sums ──────────────────────────────────── */
    {
        size_t s = 0;
        for (unsigned b = 0; b < BINS; b++) {
            size_t c = cnt_lo[b]; cnt_lo[b] = s; s += c;
        }
        s = 0;
        for (unsigned b = 0; b < BINS; b++) {
            size_t c = cnt_hi[b]; cnt_hi[b] = s; s += c;
        }
    }

    /* ── Step 3: Pass 1 — scatter by low 16 bits: arr → tmp ─────────────── */
    for (size_t i = 0; i < n; i++) {
        if (i + PREFETCH_DIST < n) {
            __builtin_prefetch(&tmp[cnt_lo[arr[i + PREFETCH_DIST] & MASK]], 1, 0);
        }
        uint32_t v = arr[i];
        tmp[cnt_lo[v & MASK]++] = v;
    }

    /* ── Step 4: Pass 2 — scatter by high 16 bits: tmp → arr ────────────── */
    for (size_t i = 0; i < n; i++) {
        if (i + PREFETCH_DIST < n) {
            __builtin_prefetch(&arr[cnt_hi[tmp[i + PREFETCH_DIST] >> RADIX_BITS]], 1, 0);
        }
        uint32_t v = tmp[i];
        arr[cnt_hi[v >> RADIX_BITS]++] = v;
    }

    free(tmp);
}
