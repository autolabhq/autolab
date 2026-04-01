/*
 * solve.h — public interface for the flash_attention task.
 *
 * This is the ONLY contract you must satisfy.  main.c and the test harness
 * depend on this declaration — do NOT change it.
 */
#pragma once

#include <stddef.h>

/*
 * attention() — scaled dot-product attention
 *
 *   output = softmax(Q K^T / sqrt(d)) V
 *
 * Arguments
 * ---------
 * Q, K, V   row-major float32 input matrices, each of shape [n × d]
 * output    row-major float32 output matrix of shape [n × d]  (caller-alloc)
 * n         sequence length
 * d         head dimension
 *
 * All four arrays are 32-byte aligned (allocated via posix_memalign).
 * Caller owns all memory; this function must not free any argument.
 */
void attention(
    const float * restrict Q,
    const float * restrict K,
    const float * restrict V,
    float       * restrict output,
    int n, int d);
