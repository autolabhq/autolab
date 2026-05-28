/* Phase 21.1 v5: 128-bit specialized arithmetic.
 *
 * For 80-bit primes, values fit in __int128. Hand-rolled modular
 * arithmetic avoids GMP's overhead (no malloc, no limb management).
 * Expected 3-5x speedup over GMP.
 *
 * Build: clang -O3 -o phase21_rho_128 phase21_rho_128.c -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>

typedef unsigned __int128 u128;
typedef __int128 i128;

#define NUM_PARTITIONS 16

typedef struct {
    u128 p;
    u128 a;
    u128 b;
} curve_t;

typedef struct {
    u128 x, y;
    int is_infinity;
} point_t;

/* Modular multiplication: (a * b) mod p, with a, b < p < 2^81.
 * Schoolbook: a * b fits in u256 (2 × u128). We need careful overflow. */
static inline u128 mod_mul(u128 a, u128 b, u128 p) {
    /* For 81-bit a, b: a * b < 2^162. We split a as a = ah * 2^64 + al.
     * Use Barrett or just iterative reduction. */
    /* Simpler: use 64-bit halves. */
    uint64_t ah = (uint64_t)(a >> 64);
    uint64_t al = (uint64_t)a;
    uint64_t bh = (uint64_t)(b >> 64);
    uint64_t bl = (uint64_t)b;

    /* a * b = ah*bh*2^128 + (ah*bl + al*bh)*2^64 + al*bl */
    u128 ll = (u128)al * bl;
    u128 lh = (u128)al * bh;
    u128 hl = (u128)ah * bl;
    u128 hh = (u128)ah * bh;

    /* Combine: result = (hh << 128) + ((lh + hl) << 64) + ll
     * For 81-bit p, ah and bh are at most 2^17 - 1, so hh < 2^34.
     * Total: < 2^162. We need this mod p. */

    /* Naive: do iterative reduction. Slow but correct. */
    /* Better: use double-word arithmetic. */
    u128 lo = ll;
    u128 mid = lh + hl;  /* could overflow, but it doesn't because each < 2^145 ... wait */
    /* lh < 2^64 * 2^17 = 2^81; hl similar. So mid < 2^82, fits in u128. */
    u128 high_part_of_mid = mid >> 64;
    u128 low_part_of_mid = mid & ((u128)((uint64_t)-1));

    /* result = (hh + high_part_of_mid) * 2^128 + (low_part_of_mid * 2^64) + lo */
    /* But we want result mod p, where p < 2^81. */
    /* We'll iteratively reduce starting from highest. */

    u128 top = hh + high_part_of_mid;  /* < 2^34 + 2^18 ≈ 2^34 */
    u128 mid_64 = low_part_of_mid;     /* < 2^64 */

    /* Represent result as top * 2^128 + mid_64 * 2^64 + lo, all mod p */

    /* Reduce top * 2^128 mod p: precompute 2^128 mod p? Too expensive per call. */
    /* Simpler: shift and subtract iteratively. */
    /* result starts as (top << 128) + (mid_64 << 64) + lo, but we represent it as
     * cur, then for each bit pull off, multiply by 2, reduce mod p. */
    /* Actually use GMP-style schoolbook reduction: */
    /* For 81-bit prime, this is at most a few hundred subtractions. */

    /* Take top: high 34 bits.
     * top * 2^128 mod p: compute by iterative doubling.
     * top_reduced = top mod p (< p, so just top since top < 2^34 < p)
     * For i = 1..128: top_reduced = (top_reduced * 2) mod p
     * That's 128 conditional subtractions; expensive but correct.
     */

    /* Simpler approach: use 256-bit arithmetic via u128 pairs and Barrett. */
    /* For now, naive: split result into 64-bit chunks and reduce bit by bit. */
    u128 result = 0;
    /* result = 0; for each bit b in (top, mid_64, lo) from high to low:
     *   result = (result * 2 + b) mod p
     */
    /* Top has at most 34 bits */
    int top_bits = 0;
    if (top > 0) {
        u128 t = top;
        while (t > 0) { top_bits++; t >>= 1; }
    }
    for (int i = top_bits - 1; i >= 0; i--) {
        result = (result << 1) | ((top >> i) & 1);
        if (result >= p) result -= p;
    }
    for (int i = 63; i >= 0; i--) {
        result = (result << 1) | ((mid_64 >> i) & 1);
        if (result >= p) result -= p;
    }
    for (int i = 127; i >= 0; i--) {
        result = (result << 1) | ((lo >> i) & 1);
        if (result >= p) result -= p;
    }
    return result;
}

static inline u128 mod_add(u128 a, u128 b, u128 p) {
    u128 r = a + b;
    if (r >= p) r -= p;
    return r;
}

static inline u128 mod_sub(u128 a, u128 b, u128 p) {
    if (a >= b) return a - b;
    return p - (b - a);
}

/* Modular inverse via extended Euclidean (binary GCD) */
static u128 mod_inv(u128 a, u128 p) {
    /* Use Fermat's little theorem: a^(p-2) mod p */
    /* But that's O(log p) multiplications, each O(81) bit reductions = slow */
    /* For benchmarking, just use Fermat */
    u128 result = 1;
    u128 base = a;
    u128 exp = p - 2;
    while (exp > 0) {
        if (exp & 1) result = mod_mul(result, base, p);
        base = mod_mul(base, base, p);
        exp >>= 1;
    }
    return result;
}

static void point_double(point_t *r, const point_t *P, const curve_t *c) {
    if (P->is_infinity || P->y == 0) { r->is_infinity = 1; return; }
    u128 lambda_num = mod_add(mod_mul(P->x, P->x, c->p), 0, c->p);
    lambda_num = mod_mul(lambda_num, 3, c->p);
    lambda_num = mod_add(lambda_num, c->a, c->p);
    u128 lambda_den = mod_add(P->y, P->y, c->p);
    u128 lambda = mod_mul(lambda_num, mod_inv(lambda_den, c->p), c->p);
    u128 x3 = mod_sub(mod_mul(lambda, lambda, c->p), P->x, c->p);
    x3 = mod_sub(x3, P->x, c->p);
    u128 y3 = mod_mul(lambda, mod_sub(P->x, x3, c->p), c->p);
    y3 = mod_sub(y3, P->y, c->p);
    r->x = x3; r->y = y3; r->is_infinity = 0;
}

static void point_add(point_t *r, const point_t *P, const point_t *Q, const curve_t *c) {
    if (P->is_infinity) { *r = *Q; return; }
    if (Q->is_infinity) { *r = *P; return; }
    if (P->x == Q->x) {
        if (mod_add(P->y, Q->y, c->p) == 0) { r->is_infinity = 1; return; }
        point_double(r, P, c); return;
    }
    u128 lambda_num = mod_sub(Q->y, P->y, c->p);
    u128 lambda_den = mod_sub(Q->x, P->x, c->p);
    u128 lambda = mod_mul(lambda_num, mod_inv(lambda_den, c->p), c->p);
    u128 x3 = mod_sub(mod_mul(lambda, lambda, c->p), P->x, c->p);
    x3 = mod_sub(x3, Q->x, c->p);
    u128 y3 = mod_mul(lambda, mod_sub(P->x, x3, c->p), c->p);
    y3 = mod_sub(y3, P->y, c->p);
    r->x = x3; r->y = y3; r->is_infinity = 0;
}

static int point_partition(const point_t *P) {
    if (P->is_infinity) return 0;
    return (int)((uint64_t)P->x & (NUM_PARTITIONS - 1));
}

static void apply_neg(point_t *P, const curve_t *c) {
    if (P->is_infinity) return;
    u128 half = c->p >> 1;
    if (P->y > half) P->y = c->p - P->y;
}

typedef struct {
    int tid, num_steps;
    const curve_t *c;
    const point_t *M;
} thread_t;

static void *worker(void *arg) {
    thread_t *T = (thread_t *)arg;
    point_t cur = T->M[T->tid % NUM_PARTITIONS];
    point_t next;
    for (int i = 0; i < T->num_steps; i++) {
        int part = point_partition(&cur);
        point_add(&next, &cur, &T->M[part], T->c);
        apply_neg(&next, T->c);
        cur = next;
    }
    return NULL;
}

static u128 parse_u128(const char *s) {
    u128 r = 0;
    while (*s >= '0' && *s <= '9') {
        r = r * 10 + (*s - '0');
        s++;
    }
    return r;
}

int main(int argc, char *argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s p a b xP yP num_steps [num_threads]\n", argv[0]);
        return 1;
    }
    curve_t c;
    c.p = parse_u128(argv[1]);
    c.a = parse_u128(argv[2]);
    c.b = parse_u128(argv[3]);
    point_t M[NUM_PARTITIONS];
    M[0].x = parse_u128(argv[4]);
    M[0].y = parse_u128(argv[5]);
    M[0].is_infinity = 0;
    for (int i = 1; i < NUM_PARTITIONS; i++) {
        point_double(&M[i], &M[i-1], &c);
    }

    int num_steps = atoi(argv[6]);
    int n_threads = (argc >= 8) ? atoi(argv[7]) : 1;
    thread_t T[16];
    pthread_t tids[16];

    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int i = 0; i < n_threads; i++) {
        T[i].tid = i; T[i].num_steps = num_steps; T[i].c = &c; T[i].M = M;
        pthread_create(&tids[i], NULL, worker, &T[i]);
    }
    for (int i = 0; i < n_threads; i++) pthread_join(tids[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double wall = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    long total = (long)n_threads * num_steps;
    fprintf(stderr, "%d threads x %d = %ld ops in %.3f s = %.2e ops/sec\n",
            n_threads, num_steps, total, wall, total / wall);
    printf("%.2e\n", total / wall);
    return 0;
}
