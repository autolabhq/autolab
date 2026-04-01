/*
 * solution/solve_optimized.c — reference solution for gaussian_blur
 *
 * Demonstrates the key optimization layers, each roughly independent:
 *
 *   Layer 1 — Separable filter
 *   ─────────────────────────
 *   A 2-D Gaussian is the outer product of two 1-D Gaussians.  Instead of
 *   visiting all 17×17 = 289 kernel entries per pixel, we make two passes:
 *     • Horizontal:  1×17 kernel along each row  → 17 MACs per pixel
 *     • Vertical:    17×1 kernel along each column → 17 MACs per pixel
 *   Total: 34 MACs instead of 289 → 8.5× fewer arithmetic operations.
 *
 *   Layer 2 — float instead of double
 *   ────────────────────────────────
 *   An AVX2 lane fits 8 floats vs 4 doubles, doubling effective SIMD width.
 *   Scalar float is also typically faster on modern x86.
 *
 *   Layer 3 — Branch-free interior loops
 *   ─────────────────────────────────────
 *   The inner loops for the image interior (KERNEL_RADIUS ≤ x < width-KERNEL_RADIUS)
 *   never need border clamping.  Split the loop into three sections: left
 *   border, interior, right border.  Eliminates branch mispredictions and
 *   allows the compiler to auto-vectorise the interior section.
 *
 *   Layer 4 — Transposed vertical pass
 *   ────────────────────────────────────
 *   Reading float columns sequentially causes cache-line striding.  Write the
 *   horizontal pass result transposed into a scratch buffer; then the vertical
 *   pass becomes a horizontal pass on the transposed data — fully sequential
 *   reads.  Transpose back into the output buffer.
 *
 *   Further headroom left to the agent:
 *   • SIMD intrinsics (SSE4.2 / AVX2): process 8 or 16 pixels at once
 *   • Integer fixed-point kernel (shift instead of multiply)
 *   • Ring buffer / prefix sum for constant-radius blur
 *   • AVX-512 on machines that support it
 */

#include "solve.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define KDIM (2 * KERNEL_RADIUS + 1)   /* 17 */

/* ─── 1-D kernel (computed once, shared across all calls) ─────────────────── */

static float  k1d[KDIM];
static int    k1d_ready = 0;

static void build_k1d(void)
{
    if (k1d_ready) return;
    const float sigma = 3.0f;
    float sum = 0.0f;
    for (int i = 0; i < KDIM; i++) {
        int d = i - KERNEL_RADIUS;
        k1d[i] = expf(-(float)(d * d) / (2.0f * sigma * sigma));
        sum += k1d[i];
    }
    for (int i = 0; i < KDIM; i++)
        k1d[i] /= sum;
    k1d_ready = 1;
}

/* ─── Helper: horizontal 1-D blur, uint8 in → float out ──────────────────── */

static void hblur_u8_to_f32(const uint8_t * restrict in,
                             float         * restrict out,
                             int w, int h)
{
    for (int y = 0; y < h; y++) {
        const uint8_t *row = in  + (size_t)y * w;
        float         *dst = out + (size_t)y * w;

        /* Left border: x ∈ [0, KERNEL_RADIUS) */
        for (int x = 0; x < KERNEL_RADIUS && x < w; x++) {
            float acc = 0.0f;
            for (int i = 0; i < KDIM; i++) {
                int sx = x + i - KERNEL_RADIUS;
                if (sx < 0) sx = 0;
                acc += k1d[i] * (float)row[sx];
            }
            dst[x] = acc;
        }

        /* Interior: no clamping — compiler can auto-vectorise */
        for (int x = KERNEL_RADIUS; x < w - KERNEL_RADIUS; x++) {
            float acc = 0.0f;
            const uint8_t *p = row + x - KERNEL_RADIUS;
            for (int i = 0; i < KDIM; i++)
                acc += k1d[i] * (float)p[i];
            dst[x] = acc;
        }

        /* Right border: x ∈ [w-KERNEL_RADIUS, w) */
        for (int x = (w > KERNEL_RADIUS ? w - KERNEL_RADIUS : KERNEL_RADIUS);
             x < w; x++) {
            float acc = 0.0f;
            for (int i = 0; i < KDIM; i++) {
                int sx = x + i - KERNEL_RADIUS;
                if (sx >= w) sx = w - 1;
                acc += k1d[i] * (float)row[sx];
            }
            dst[x] = acc;
        }
    }
}

/* ─── Helper: horizontal 1-D blur, float in → float out ──────────────────── */

static void hblur_f32_to_f32(const float * restrict in,
                              float       * restrict out,
                              int w, int h)
{
    for (int y = 0; y < h; y++) {
        const float *row = in  + (size_t)y * w;
        float       *dst = out + (size_t)y * w;

        for (int x = 0; x < KERNEL_RADIUS && x < w; x++) {
            float acc = 0.0f;
            for (int i = 0; i < KDIM; i++) {
                int sx = x + i - KERNEL_RADIUS;
                if (sx < 0) sx = 0;
                acc += k1d[i] * row[sx];
            }
            dst[x] = acc;
        }

        for (int x = KERNEL_RADIUS; x < w - KERNEL_RADIUS; x++) {
            float acc = 0.0f;
            const float *p = row + x - KERNEL_RADIUS;
            for (int i = 0; i < KDIM; i++)
                acc += k1d[i] * p[i];
            dst[x] = acc;
        }

        for (int x = (w > KERNEL_RADIUS ? w - KERNEL_RADIUS : KERNEL_RADIUS);
             x < w; x++) {
            float acc = 0.0f;
            for (int i = 0; i < KDIM; i++) {
                int sx = x + i - KERNEL_RADIUS;
                if (sx >= w) sx = w - 1;
                acc += k1d[i] * row[sx];
            }
            dst[x] = acc;
        }
    }
}

/* ─── Helper: transpose a w×h float matrix into h×w ──────────────────────── */

#define TILE 64   /* tile size for cache-friendly transposition */

static void transpose_f32(const float * restrict in,
                           float       * restrict out,
                           int w, int h)
{
    for (int ty = 0; ty < h; ty += TILE) {
        int yend = ty + TILE < h ? ty + TILE : h;
        for (int tx = 0; tx < w; tx += TILE) {
            int xend = tx + TILE < w ? tx + TILE : w;
            for (int y = ty; y < yend; y++) {
                for (int x = tx; x < xend; x++) {
                    out[(size_t)x * h + y] = in[(size_t)y * w + x];
                }
            }
        }
    }
}

/* ─── Helper: quantise float buffer → uint8 output ───────────────────────── */

static void f32_to_u8(const float * restrict in, uint8_t * restrict out,
                      size_t n)
{
    for (size_t i = 0; i < n; i++) {
        int v = (int)(in[i] + 0.5f);
        if (v < 0)   v = 0;
        if (v > 255) v = 255;
        out[i] = (uint8_t)v;
    }
}

/* ─── blur_image ──────────────────────────────────────────────────────────── */

void blur_image(const uint8_t *src, uint8_t *dst,
                int width, int height, int passes)
{
    build_k1d();

    const size_t NP = (size_t)width * height;

    /*
     * Scratch buffers:
     *   horiz  — result of horizontal pass (w×h float, row-major)
     *   transT — transposed result        (h×w float, row-major after transpose)
     *   transH — horizontal pass on transposed image (h×w float)
     *   ping   — uint8 ping buffer for multi-pass
     */
    float   *horiz  = (float *)malloc(NP * sizeof(float));
    float   *transT = (float *)malloc(NP * sizeof(float));
    float   *transH = (float *)malloc(NP * sizeof(float));
    uint8_t *ping   = (uint8_t *)malloc(NP);
    if (!horiz || !transT || !transH || !ping) {
        free(horiz); free(transT); free(transH); free(ping);
        return;
    }

    memcpy(ping, src, NP);

    for (int p = 0; p < passes; p++) {
        /*
         * Pass structure:
         *  1. Horizontal blur:  ping  (uint8, w×h) → horiz  (float, w×h)
         *  2. Transpose:        horiz (w×h)         → transT (h×w)
         *  3. Horizontal blur:  transT(h×w)         → transH (h×w)
         *     (this is the vertical pass on the original image, now sequential)
         *  4. Transpose back:   transH (h×w)        → horiz  (w×h)
         *  5. Quantise:         horiz → ping
         */

        hblur_u8_to_f32(ping,   horiz,  width,  height);
        transpose_f32  (horiz,  transT, width,  height);
        hblur_f32_to_f32(transT, transH, height, width);
        transpose_f32  (transH, horiz,  height, width);
        f32_to_u8      (horiz,  ping,   NP);
    }

    memcpy(dst, ping, NP);
    free(horiz); free(transT); free(transH); free(ping);
}
