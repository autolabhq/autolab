/*
 * solve.c
 *
 * Strategy
 * ────────
 * For each output pixel we loop over ALL 289 entries of a 17×17 kernel and
 * accumulate a double-precision weighted sum, then round and clamp to uint8.
 *
 * YOUR TASK
 * ─────────
 * Rewrite blur_image() in this file to make it as fast as possible.
 * You must NOT modify solve.h, main.c, or the Makefile.
 */

#include "solve.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── Kernel ──────────────────────────────────────────────────────────────── */

#define KDIM (2 * KERNEL_RADIUS + 1)   /* 17 */

static double kernel2d[KDIM][KDIM];
static int    kernel_ready = 0;

static void build_kernel(void)
{
    if (kernel_ready) return;

    const double sigma = 3.0;
    double sum = 0.0;

    for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {
        for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {
            double v = exp(-(double)(dy * dy + dx * dx)
                           / (2.0 * sigma * sigma));
            kernel2d[dy + KERNEL_RADIUS][dx + KERNEL_RADIUS] = v;
            sum += v;
        }
    }

    /* Normalise so the kernel sums to exactly 1.0 */
    for (int i = 0; i < KDIM; i++)
        for (int j = 0; j < KDIM; j++)
            kernel2d[i][j] /= sum;

    kernel_ready = 1;
}

/* ─── blur_image ──────────────────────────────────────────────────────────── */

void blur_image(const uint8_t *src, uint8_t *dst,
                int width, int height, int passes)
{
    build_kernel();

    uint8_t *ping = (uint8_t *)malloc((size_t)width * height);
    uint8_t *pong = (uint8_t *)malloc((size_t)width * height);
    if (!ping || !pong) { free(ping); free(pong); return; }

    memcpy(ping, src, (size_t)width * height);

    for (int p = 0; p < passes; p++) {

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {

                double acc = 0.0;

                for (int dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy++) {

                    /* Clamp row — branch inside the inner loop on purpose */
                    int ny = y + dy;
                    if      (ny < 0)      ny = 0;
                    else if (ny >= height) ny = height - 1;

                    for (int dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx++) {

                        /* Clamp column */
                        int nx = x + dx;
                        if      (nx < 0)     nx = 0;
                        else if (nx >= width) nx = width - 1;

                        acc += kernel2d[dy + KERNEL_RADIUS][dx + KERNEL_RADIUS]
                               * (double)ping[ny * width + nx];
                    }
                }

                /* Round to nearest and clamp to [0, 255] */
                int val = (int)(acc + 0.5);
                if (val < 0)   val = 0;
                if (val > 255) val = 255;
                pong[y * width + x] = (uint8_t)val;
            }
        }

        /* Swap ping-pong buffers */
        uint8_t *tmp = ping;
        ping = pong;
        pong = tmp;
    }

    memcpy(dst, ping, (size_t)width * height);
    free(ping);
    free(pong);
}
