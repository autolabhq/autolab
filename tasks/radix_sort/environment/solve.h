/*
 * solve.h — public interface for the radix_sort benchmark.
 *
 * The only file you need to implement is solve.c.
 * Do NOT change this file.
 */
#pragma once

#include <stdint.h>
#include <stdlib.h>

/*
 * Sort arr[0..n-1] in ascending order, in place.
 *
 * Requirements:
 *   - Correctness: arr must be fully sorted when the function returns.
 *   - No side effects on memory outside arr (no writes beyond arr[n-1]).
 *   - Must handle n == 0 and n == 1 without error.
 */
void radix_sort(uint32_t *arr, size_t n);
