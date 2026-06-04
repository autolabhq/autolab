#include "solve.h"
#include <stdint.h>

/*
 * Myers O(lb) bit-vector edit-distance algorithm.
 *
 * Uses a single uint64_t to track all la rows simultaneously.
 * For la ≤ 64 (our guaranteed bound), this covers the full pattern.
 *
 * Key details vs. naïve implementation:
 *   - pv is initialized to ~0 (all bits), not (1<<la)-1.
 *   - After computing ph, shift left and OR with 1 to encode the
 *     invariant that d[0][j]=j (row-0 horizontal delta is always +1).
 *   - Both adjustments come from Myers (1999), Lemma 3.
 *
 * Reference: Myers, "A Fast Bit-Vector Algorithm for Approximate
 * String Matching Based on Dynamic Programming",
 * JACM 46(3):395–415, 1999.
 */
static int myers64(const char *a, int la, const char *b, int lb)
{
    uint64_t pm[256] = {0};
    for (int i = 0; i < la; i++)
        pm[(unsigned char)a[i]] |= (uint64_t)1 << i;

    uint64_t pv = ~(uint64_t)0;
    uint64_t mv = 0;
    int score = la;

    for (int j = 0; j < lb; j++) {
        uint64_t eq = pm[(unsigned char)b[j]];
        uint64_t xv = eq | mv;
        uint64_t xh = (((eq & pv) + pv) ^ pv) | xv;
        uint64_t ph = mv | ~(xh | pv);
        uint64_t mh = pv & xh;
        score += (int)((ph >> (la - 1)) & 1) - (int)((mh >> (la - 1)) & 1);
        ph = (ph << 1) | 1;         /* encode d[0][j]=j boundary */
        pv = (mh << 1) | ~(xh | ph);
        mv = xh & ph;
    }
    return score;
}

int levenshtein(const char *a, int la, const char *b, int lb)
{
    if (la == 0) return lb;
    if (lb == 0) return la;
    /* Keep the pattern (a) as the shorter string; distance is symmetric. */
    if (la > lb) {
        const char *t = a; a = b; b = t;
        int ti = la; la = lb; lb = ti;
    }
    return myers64(a, la, b, lb);
}
