/*
 * solve.c
 *
 * Strategy
 * ────────
 * For every row in the probe table S we scan the entire build table R
 * looking for matching keys.  This is a classic O(|R| × |S|) nested-loop
 * join — correct but far too slow for large inputs.
 *
 * YOUR TASK
 * ─────────
 * Rewrite hash_join() in this file to make it as fast as possible.
 * You must NOT modify solve.h, main.c, or the Makefile.
 */

#include "solve.h"

void hash_join(const Row *r, int r_count,
               const Row *s, int s_count,
               JoinResult *out)
{
    int64_t  count    = 0;
    uint64_t checksum = 0;

    /*
     * Outer loop: iterate over every probe-side row in S.
     * Inner loop: linear scan of the entire build-side R.
     *
     * The compiler will keep r_count, sk, and the inner loop counter in
     * registers, but r[] (640 KB) must be re-read from L3 cache on every
     * outer iteration after the first few passes warm it up.
     */
    for (int j = 0; j < s_count; j++) {
        const int32_t sk = s[j].key;
        const int32_t sp = s[j].payload;

        for (int i = 0; i < r_count; i++) {
            if (r[i].key == sk) {
                count++;
                /* Widen first, then add — no 32-bit wrap-around. */
                checksum += (uint64_t)(uint32_t)r[i].payload
                          + (uint64_t)(uint32_t)sp;
            }
        }
    }

    out->match_count = count;
    out->checksum    = checksum;
}
