/*
 * solve.c — Scaled dot-product attention  (BASELINE — intentionally slow)
 *
 * This implementation is correct but unoptimised.  It serves as the
 * starting point that you must improve.
 *
 * ── Known performance problems (profile and fix them) ────────────────────
 *
 *  1. The baseline materializes large intermediate n×n buffers.
 *  2. The hot path uses expensive scalar math and multiple full-memory passes.
 *  3. The implementation does much more memory traffic than necessary.
 *
 * ── Your task ──────────────────────────────────────────────────────────────
 * Rewrite attention() in this file to fix as many problems as you can.
 * You may add helper functions, static arrays, and #include directives.
 * Do NOT change the function signature declared in solve.h.
 */

#include "solve.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void attention(
    const float * restrict Q,
    const float * restrict K,
    const float * restrict V,
    float       * restrict output,
    int n, int d)
{
    const double scale = 1.0 / sqrt((double)d);

    /* Allocate the full n×n score and probability matrices in double. */
    double *S = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *P = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!S || !P) { free(S); free(P); return; }

    /* ── Step 1: S[i][j] = dot(Q[i], K[j]) / sqrt(d) ────────────────── */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int k = 0; k < d; k++)
                dot += (double)Q[i * d + k] * (double)K[j * d + k];
            S[(size_t)i * n + j] = dot * scale;
        }
    }

    /* ── Step 2: P = row-wise softmax(S) — three separate passes ──────── */
    for (int i = 0; i < n; i++) {
        const double *row_s = S + (size_t)i * n;
        double       *row_p = P + (size_t)i * n;

        /* Pass (a): find row maximum for numerical stability */
        double mx = row_s[0];
        for (int j = 1; j < n; j++)
            if (row_s[j] > mx) mx = row_s[j];

        /* Pass (b): exponentiate and accumulate sum */
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            row_p[j] = exp(row_s[j] - mx);   /* bottleneck: scalar double exp */
            sum += row_p[j];
        }

        /* Pass (c): normalize */
        double inv = 1.0 / sum;
        for (int j = 0; j < n; j++)
            row_p[j] *= inv;
    }

    /* ── Step 3: output[i] = sum_j P[i][j] * V[j] ─────────────────────── */
    memset(output, 0, (size_t)n * (size_t)d * sizeof(float));
    for (int i = 0; i < n; i++) {
        const double *pi = P + (size_t)i * n;
        float        *oi = output + i * d;
        for (int j = 0; j < n; j++) {
            const float  *vj  = V + j * d;
            const double  pij = pi[j];
            for (int k = 0; k < d; k++)
                oi[k] += (float)(pij * (double)vj[k]);
        }
    }

    free(S);
    free(P);
}
