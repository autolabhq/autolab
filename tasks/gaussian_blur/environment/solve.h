/*
 * solve.h — interface for the gaussian_blur task.  DO NOT MODIFY.
 *
 * Your entire submission lives in solve.c.  You may not change this file,
 * main.c, or the Makefile.
 */

#ifndef SOLVE_H
#define SOLVE_H

#include <stdint.h>

/* ─── Benchmark parameters ────────────────────────────────────────────────── */

#define IMG_WIDTH     4096   /* pixels per row   */
#define IMG_HEIGHT    4096   /* rows per image   */
#define BLUR_PASSES      5   /* consecutive Gaussian blur passes */
#define KERNEL_RADIUS    8   /* half-width: full kernel is 17×17 */

/* ─── Public interface ────────────────────────────────────────────────────── */

/*
 * blur_image() — apply a 17×17 Gaussian blur (σ = 3.0) exactly BLUR_PASSES
 * times to a grayscale image.
 *
 *   src    – input image.  Row-major, one byte per pixel:
 *             pixel (row y, col x) is at src[y * width + x].  READ ONLY.
 *   dst    – output image.  Same layout as src.  Must NOT alias src.
 *   width  – columns per row   (== IMG_WIDTH   for the benchmark run)
 *   height – rows in the image (== IMG_HEIGHT  for the benchmark run)
 *   passes – number of blur passes to apply (== BLUR_PASSES for the benchmark)
 *
 * Border convention:
 *   When sampling outside the image, clamp the coordinate to [0, dim-1].
 *   i.e., the pixel at (clamp(y+dy, 0, height-1), clamp(x+dx, 0, width-1)).
 *
 * Each pass feeds into the next: the output of pass k is the input to pass k+1.
 *
 * The correctness verifier compares your output to a reference implementation
 * and allows a maximum per-pixel absolute deviation of CORRECTNESS_TOLERANCE.
 */
void blur_image(const uint8_t *src, uint8_t *dst,
                int width, int height, int passes);

/* Per-pixel tolerance used by tests/verify_correctness.py */
#define CORRECTNESS_TOLERANCE 4

#endif /* SOLVE_H */
