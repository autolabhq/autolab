/*
 * solve.c — sort implementation (naive stdlib qsort).
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  OPTIMIZE THIS FILE.                                                    │
 * │  Do NOT modify solve.h, main.c, Makefile, or anything under tests/.    │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * The baseline uses the standard library's comparison-based qsort
 */

#include "solve.h"
#include <stdlib.h>

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

void radix_sort(uint32_t *arr, size_t n)
{
    qsort(arr, n, sizeof(uint32_t), cmp_u32);
}
