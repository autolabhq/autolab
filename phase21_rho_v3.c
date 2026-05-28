/* Phase 21.1 v3: C-extension Pollard rho with affine + pthreads.
 *
 * Affine arithmetic (1 inverse per addition) is FASTER than Jacobian
 * at 81-bit because GMP's mpz_invert is competitive at this size.
 * Just add pthreads for multi-CPU scaling.
 *
 * Build: gcc -O3 -o phase21_rho_v3 phase21_rho_v3.c -lgmp -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <gmp.h>

#define NUM_PARTITIONS 16

typedef struct {
    mpz_t p, a, b;
} curve_t;

typedef struct {
    mpz_t x, y;
    int is_infinity;
} point_t;

static void point_init(point_t *P) {
    mpz_init(P->x); mpz_init(P->y); P->is_infinity = 0;
}
static void point_clear(point_t *P) {
    mpz_clear(P->x); mpz_clear(P->y);
}
static void point_set(point_t *o, const point_t *i) {
    mpz_set(o->x, i->x); mpz_set(o->y, i->y); o->is_infinity = i->is_infinity;
}

static void point_double(point_t *r, const point_t *P, const curve_t *c,
                          mpz_t t1, mpz_t t2, mpz_t lambda) {
    if (P->is_infinity || mpz_sgn(P->y) == 0) { r->is_infinity = 1; return; }
    mpz_mul(t1, P->x, P->x); mpz_mul_ui(t1, t1, 3);
    mpz_add(t1, t1, c->a); mpz_mod(t1, t1, c->p);
    mpz_add(t2, P->y, P->y); mpz_invert(t2, t2, c->p);
    mpz_mul(lambda, t1, t2); mpz_mod(lambda, lambda, c->p);
    mpz_mul(t1, lambda, lambda); mpz_sub(t1, t1, P->x); mpz_sub(t1, t1, P->x);
    mpz_mod(t1, t1, c->p);
    mpz_sub(t2, P->x, t1); mpz_mul(t2, t2, lambda); mpz_sub(t2, t2, P->y);
    mpz_mod(t2, t2, c->p);
    mpz_set(r->x, t1); mpz_set(r->y, t2); r->is_infinity = 0;
}

static void point_add(point_t *r, const point_t *P, const point_t *Q,
                       const curve_t *c, mpz_t t1, mpz_t t2, mpz_t lambda) {
    if (P->is_infinity) { point_set(r, Q); return; }
    if (Q->is_infinity) { point_set(r, P); return; }
    if (mpz_cmp(P->x, Q->x) == 0) {
        mpz_add(t1, P->y, Q->y); mpz_mod(t1, t1, c->p);
        if (mpz_sgn(t1) == 0) { r->is_infinity = 1; return; }
        point_double(r, P, c, t1, t2, lambda); return;
    }
    mpz_sub(t1, Q->y, P->y);
    mpz_sub(t2, Q->x, P->x);
    mpz_invert(t2, t2, c->p);
    mpz_mul(lambda, t1, t2); mpz_mod(lambda, lambda, c->p);
    mpz_mul(t1, lambda, lambda); mpz_sub(t1, t1, P->x); mpz_sub(t1, t1, Q->x);
    mpz_mod(t1, t1, c->p);
    mpz_sub(t2, P->x, t1); mpz_mul(t2, t2, lambda); mpz_sub(t2, t2, P->y);
    mpz_mod(t2, t2, c->p);
    mpz_set(r->x, t1); mpz_set(r->y, t2); r->is_infinity = 0;
}

static int point_partition(const point_t *P, int n) {
    if (P->is_infinity) return 0;
    return (int)(mpz_get_ui(P->x) & (n - 1));
}

static void apply_neg(point_t *P, const curve_t *c, mpz_t t) {
    if (P->is_infinity) return;
    mpz_tdiv_q_2exp(t, c->p, 1);
    if (mpz_cmp(P->y, t) > 0) mpz_sub(P->y, c->p, P->y);
}

typedef struct {
    int tid;
    int num_steps;
    const curve_t *c;
    const point_t *M;  /* [NUM_PARTITIONS] */
    long ops_done;
} thread_t;

static void *worker(void *arg) {
    thread_t *T = (thread_t *)arg;
    point_t cur, next;
    point_init(&cur); point_init(&next);
    point_set(&cur, &T->M[T->tid % NUM_PARTITIONS]);
    mpz_t t1, t2, lambda;
    mpz_init(t1); mpz_init(t2); mpz_init(lambda);
    for (int i = 0; i < T->num_steps; i++) {
        int part = point_partition(&cur, NUM_PARTITIONS);
        point_add(&next, &cur, &T->M[part], T->c, t1, t2, lambda);
        apply_neg(&next, T->c, t1);
        point_set(&cur, &next);
    }
    T->ops_done = T->num_steps;
    point_clear(&cur); point_clear(&next);
    mpz_clear(t1); mpz_clear(t2); mpz_clear(lambda);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s p a b xP yP num_steps [num_threads]\n", argv[0]);
        return 1;
    }
    curve_t c;
    mpz_init_set_str(c.p, argv[1], 10);
    mpz_init_set_str(c.a, argv[2], 10);
    mpz_init_set_str(c.b, argv[3], 10);
    point_t M[NUM_PARTITIONS];
    mpz_t t1, t2, lambda;
    mpz_init(t1); mpz_init(t2); mpz_init(lambda);
    /* Bootstrap M[i] by doubling */
    point_init(&M[0]);
    mpz_set_str(M[0].x, argv[4], 10);
    mpz_set_str(M[0].y, argv[5], 10);
    M[0].is_infinity = 0;
    for (int i = 1; i < NUM_PARTITIONS; i++) {
        point_init(&M[i]);
        point_double(&M[i], &M[i-1], &c, t1, t2, lambda);
    }
    int num_steps = atoi(argv[6]);
    int n_threads = (argc >= 8) ? atoi(argv[7]) : 4;

    thread_t T[16];
    pthread_t tids[16];
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int i = 0; i < n_threads; i++) {
        T[i].tid = i; T[i].num_steps = num_steps; T[i].c = &c; T[i].M = M; T[i].ops_done = 0;
        pthread_create(&tids[i], NULL, worker, &T[i]);
    }
    long total = 0;
    for (int i = 0; i < n_threads; i++) {
        pthread_join(tids[i], NULL);
        total += T[i].ops_done;
    }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double wall = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) * 1e-9;
    double rate = total / wall;
    fprintf(stderr, "%d threads x %d steps = %ld ops in %.3f s = %.2e ops/sec\n",
            n_threads, num_steps, total, wall, rate);
    printf("%.2e\n", rate);

    for (int i = 0; i < NUM_PARTITIONS; i++) point_clear(&M[i]);
    mpz_clear(t1); mpz_clear(t2); mpz_clear(lambda);
    mpz_clear(c.p); mpz_clear(c.a); mpz_clear(c.b);
    return 0;
}
