#include "solve.h"
#include <string.h>

/*
 * Baseline: classic two-row rolling dynamic-programming Levenshtein.
 * Time: O(la × lb)   Space: O(lb)
 */

static inline int min2(int a, int b) { return a < b ? a : b; }
static inline int min3(int a, int b, int c) { return min2(a, min2(b, c)); }

int levenshtein(const char *a, int la, const char *b, int lb)
{
    int prev[65], curr[65];
    int i, j;

    for (j = 0; j <= lb; j++) prev[j] = j;

    for (i = 1; i <= la; i++) {
        curr[0] = i;
        for (j = 1; j <= lb; j++) {
            int sub = prev[j-1] + (a[i-1] != b[j-1]);
            curr[j] = min3(prev[j] + 1, curr[j-1] + 1, sub);
        }
        memcpy(prev, curr, (size_t)(lb + 1) * sizeof(int));
    }
    return prev[lb];
}
