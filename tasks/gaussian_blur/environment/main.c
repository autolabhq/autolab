/*
 * main.c — benchmark driver for the gaussian_blur task.  DO NOT MODIFY.
 * _POSIX_C_SOURCE exposes clock_gettime / CLOCK_MONOTONIC under -std=c99.
 *
 * Usage:
 *   ./blur <input.raw> [output.raw] [width] [height]
 *
 *   <input.raw>   — raw grayscale image: exactly width*height bytes, one byte
 *                   per pixel, row-major (row y, col x → byte y*width+x).
 *   [output.raw]  — optional: write the blurred output image to this file.
 *                   Pass "-" to skip writing.
 *   [width]       — image width in pixels  (default: IMG_WIDTH  = 4096)
 *   [height]      — image height in pixels (default: IMG_HEIGHT = 4096)
 *
 * Output to stdout (one line):
 *   time=<seconds>  checksum=<uint64>
 *
 * The benchmark performs one warm-up call (to allow kernel initialisation and
 * page-faulting to settle) and then one timed call.  The warm-up result is
 * discarded; the timed result's output is checksummed and optionally saved.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "solve.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <input.raw> [output.raw] [width] [height]\n",
                argv[0]);
        return 1;
    }

    const char *input_path  = argv[1];
    const char *output_path = (argc >= 3 && strcmp(argv[2], "-") != 0)
                              ? argv[2] : NULL;
    int width  = (argc >= 4) ? atoi(argv[3]) : IMG_WIDTH;
    int height = (argc >= 5) ? atoi(argv[4]) : IMG_HEIGHT;

    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Invalid dimensions: %d x %d\n", width, height);
        return 1;
    }

    const size_t NPIXELS = (size_t)width * height;

    /* ── Load input image ───────────────────────────────────────────────── */
    FILE *f = fopen(input_path, "rb");
    if (!f) { perror("fopen(input)"); return 1; }

    uint8_t *src = (uint8_t *)malloc(NPIXELS);
    uint8_t *dst = (uint8_t *)malloc(NPIXELS);
    if (!src || !dst) { fputs("OOM\n", stderr); return 1; }

    if (fread(src, 1, NPIXELS, f) != NPIXELS) {
        fprintf(stderr, "Short read: expected %zu bytes from %s\n",
                NPIXELS, input_path);
        return 1;
    }
    fclose(f);

    /* ── Warm-up (excluded from timing) ─────────────────────────────────── */
    blur_image(src, dst, width, height, BLUR_PASSES);

    /* ── Timed run ──────────────────────────────────────────────────────── */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    blur_image(src, dst, width, height, BLUR_PASSES);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (double)(t1.tv_sec  - t0.tv_sec)
                   + (double)(t1.tv_nsec - t0.tv_nsec) * 1.0e-9;

    /* ── Checksum ───────────────────────────────────────────────────────── */
    uint64_t checksum = 0;
    for (size_t i = 0; i < NPIXELS; i++)
        checksum += dst[i];

    printf("time=%.6f checksum=%llu\n",
           elapsed, (unsigned long long)checksum);

    /* ── Optional: save output image ───────────────────────────────────── */
    if (output_path) {
        FILE *out = fopen(output_path, "wb");
        if (!out) { perror("fopen(output)"); return 1; }
        fwrite(dst, 1, NPIXELS, out);
        fclose(out);
    }

    free(src);
    free(dst);
    return 0;
}
