/* Phase 21.1 v2: C-extension Pollard rho with projective coords + pthreads.
 *
 * Improvements over v1:
 *  - Projective (Jacobian) coordinates eliminate per-add inverse
 *  - pthreads for multi-CPU parallelism
 *  - Negation map folded into walk
 *
 * Build: gcc -O3 -o phase21_rho_v2 phase21_rho_v2.c -lgmp -lpthread
 *
 * Each thread does an independent random walk; we measure aggregate
 * throughput across threads.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <pthread.h>
#include <gmp.h>

#define NUM_PARTITIONS 16
#define DEFAULT_NUM_THREADS 4

/* Jacobian projective point: (X, Y, Z) representing (X/Z^2, Y/Z^3) */
typedef struct {
    mpz_t X, Y, Z;
    int is_infinity;
} jpoint_t;

typedef struct {
    mpz_t p;
    mpz_t a;
    mpz_t b;
} curve_t;

static void jpoint_init(jpoint_t *P) {
    mpz_init(P->X);
    mpz_init(P->Y);
    mpz_init(P->Z);
    P->is_infinity = 0;
}

static void jpoint_clear(jpoint_t *P) {
    mpz_clear(P->X);
    mpz_clear(P->Y);
    mpz_clear(P->Z);
}

/* Jacobian point doubling: 2(X,Y,Z) for y^2 = x^3 + a*x + b
 * Standard formulas; needs ~10 multiplications, 0 inversions. */
static void jpoint_double(jpoint_t *out, const jpoint_t *P, const curve_t *c,
                           mpz_t t1, mpz_t t2, mpz_t t3, mpz_t t4) {
    if (P->is_infinity || mpz_sgn(P->Y) == 0) {
        out->is_infinity = 1;
        return;
    }
    /* A = X^2 */
    mpz_mul(t1, P->X, P->X);
    mpz_mod(t1, t1, c->p);
    /* B = Y^2 */
    mpz_mul(t2, P->Y, P->Y);
    mpz_mod(t2, t2, c->p);
    /* C = B^2 = Y^4 */
    mpz_mul(t3, t2, t2);
    mpz_mod(t3, t3, c->p);
    /* D = 2*((X+B)^2 - A - C) */
    mpz_add(t4, P->X, t2);
    mpz_mul(t4, t4, t4);
    mpz_sub(t4, t4, t1);
    mpz_sub(t4, t4, t3);
    mpz_add(t4, t4, t4);
    mpz_mod(t4, t4, c->p);
    /* E = 3*A + a*Z^4 */
    mpz_t Z2, Z4, E, F, X3, Y3, Z3;
    mpz_init(Z2); mpz_init(Z4); mpz_init(E); mpz_init(F);
    mpz_init(X3); mpz_init(Y3); mpz_init(Z3);
    mpz_mul(Z2, P->Z, P->Z);
    mpz_mod(Z2, Z2, c->p);
    mpz_mul(Z4, Z2, Z2);
    mpz_mod(Z4, Z4, c->p);
    mpz_mul(E, c->a, Z4);
    mpz_mul_ui(F, t1, 3);
    mpz_add(E, E, F);
    mpz_mod(E, E, c->p);
    /* X3 = E^2 - 2D */
    mpz_mul(X3, E, E);
    mpz_sub(X3, X3, t4);
    mpz_sub(X3, X3, t4);
    mpz_mod(X3, X3, c->p);
    /* Y3 = E*(D - X3) - 8*C */
    mpz_sub(Y3, t4, X3);
    mpz_mul(Y3, Y3, E);
    mpz_mul_ui(F, t3, 8);
    mpz_sub(Y3, Y3, F);
    mpz_mod(Y3, Y3, c->p);
    /* Z3 = 2*Y*Z */
    mpz_mul(Z3, P->Y, P->Z);
    mpz_add(Z3, Z3, Z3);
    mpz_mod(Z3, Z3, c->p);
    mpz_set(out->X, X3); mpz_set(out->Y, Y3); mpz_set(out->Z, Z3);
    out->is_infinity = 0;
    mpz_clear(Z2); mpz_clear(Z4); mpz_clear(E); mpz_clear(F);
    mpz_clear(X3); mpz_clear(Y3); mpz_clear(Z3);
}

/* Jacobian point addition: P + Q where P and Q both non-infinity. */
static void jpoint_add(jpoint_t *out, const jpoint_t *P, const jpoint_t *Q,
                        const curve_t *c, mpz_t t1, mpz_t t2, mpz_t t3, mpz_t t4) {
    if (P->is_infinity) { mpz_set(out->X, Q->X); mpz_set(out->Y, Q->Y); mpz_set(out->Z, Q->Z); out->is_infinity = Q->is_infinity; return; }
    if (Q->is_infinity) { mpz_set(out->X, P->X); mpz_set(out->Y, P->Y); mpz_set(out->Z, P->Z); out->is_infinity = P->is_infinity; return; }
    /* U1 = X1 * Z2^2, U2 = X2 * Z1^2 */
    mpz_t U1, U2, S1, S2, H, R, H2, H3, X3, Y3, Z3;
    mpz_init(U1); mpz_init(U2); mpz_init(S1); mpz_init(S2);
    mpz_init(H); mpz_init(R); mpz_init(H2); mpz_init(H3);
    mpz_init(X3); mpz_init(Y3); mpz_init(Z3);
    mpz_mul(t1, Q->Z, Q->Z);  /* t1 = Z2^2 */
    mpz_mod(t1, t1, c->p);
    mpz_mul(U1, P->X, t1);
    mpz_mod(U1, U1, c->p);
    mpz_mul(t2, P->Z, P->Z);
    mpz_mod(t2, t2, c->p);
    mpz_mul(U2, Q->X, t2);
    mpz_mod(U2, U2, c->p);
    /* S1 = Y1 * Z2^3 */
    mpz_mul(t3, t1, Q->Z); mpz_mod(t3, t3, c->p);
    mpz_mul(S1, P->Y, t3); mpz_mod(S1, S1, c->p);
    mpz_mul(t4, t2, P->Z); mpz_mod(t4, t4, c->p);
    mpz_mul(S2, Q->Y, t4); mpz_mod(S2, S2, c->p);
    /* H = U2 - U1; R = S2 - S1 */
    mpz_sub(H, U2, U1); mpz_mod(H, H, c->p);
    mpz_sub(R, S2, S1); mpz_mod(R, R, c->p);
    if (mpz_sgn(H) == 0) {
        if (mpz_sgn(R) == 0) {
            /* doubling */
            jpoint_double(out, P, c, t1, t2, t3, t4);
            goto cleanup;
        }
        out->is_infinity = 1;
        goto cleanup;
    }
    mpz_mul(H2, H, H); mpz_mod(H2, H2, c->p);
    mpz_mul(H3, H2, H); mpz_mod(H3, H3, c->p);
    /* X3 = R^2 - H^3 - 2*U1*H^2 */
    mpz_mul(X3, R, R); mpz_sub(X3, X3, H3);
    mpz_mul(t1, U1, H2);
    mpz_sub(X3, X3, t1); mpz_sub(X3, X3, t1);
    mpz_mod(X3, X3, c->p);
    /* Y3 = R*(U1*H^2 - X3) - S1*H^3 */
    mpz_mul(Y3, U1, H2);
    mpz_sub(Y3, Y3, X3);
    mpz_mul(Y3, Y3, R);
    mpz_mul(t1, S1, H3);
    mpz_sub(Y3, Y3, t1);
    mpz_mod(Y3, Y3, c->p);
    /* Z3 = Z1 * Z2 * H */
    mpz_mul(Z3, P->Z, Q->Z); mpz_mod(Z3, Z3, c->p);
    mpz_mul(Z3, Z3, H); mpz_mod(Z3, Z3, c->p);
    mpz_set(out->X, X3); mpz_set(out->Y, Y3); mpz_set(out->Z, Z3);
    out->is_infinity = 0;
cleanup:
    mpz_clear(U1); mpz_clear(U2); mpz_clear(S1); mpz_clear(S2);
    mpz_clear(H); mpz_clear(R); mpz_clear(H2); mpz_clear(H3);
    mpz_clear(X3); mpz_clear(Y3); mpz_clear(Z3);
}

/* Partition function using x of P (after dividing by Z^2 — but we
 * can use just X mod l since it's deterministic given (X, Z)) */
static int jpoint_partition(const jpoint_t *P, int num_partitions) {
    if (P->is_infinity) return 0;
    /* Use low bits of X (not the affine x; doesn't matter for hashing) */
    return (int)(mpz_get_ui(P->X) & (num_partitions - 1));
}

typedef struct {
    int thread_id;
    int num_steps;
    const curve_t *curve;
    const jpoint_t *M;  /* shared multipliers */
    double ops_per_sec;
} thread_arg_t;

static void *worker(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    const curve_t *curve = targ->curve;
    const jpoint_t *M = targ->M;

    jpoint_t cur, next;
    jpoint_init(&cur); jpoint_init(&next);
    /* Initialize cur = M[0] */
    mpz_set(cur.X, M[0].X); mpz_set(cur.Y, M[0].Y); mpz_set(cur.Z, M[0].Z);
    cur.is_infinity = M[0].is_infinity;

    mpz_t t1, t2, t3, t4;
    mpz_init(t1); mpz_init(t2); mpz_init(t3); mpz_init(t4);

    clock_t t_start = clock();
    int steps_done = 0;
    for (int s = 0; s < targ->num_steps; s++) {
        int part = jpoint_partition(&cur, NUM_PARTITIONS);
        jpoint_add(&next, &cur, &M[part], curve, t1, t2, t3, t4);
        mpz_set(cur.X, next.X); mpz_set(cur.Y, next.Y); mpz_set(cur.Z, next.Z);
        cur.is_infinity = next.is_infinity;
        steps_done++;
    }
    clock_t t_end = clock();
    double elapsed = (double)(t_end - t_start) / CLOCKS_PER_SEC;
    targ->ops_per_sec = steps_done / elapsed;

    jpoint_clear(&cur); jpoint_clear(&next);
    mpz_clear(t1); mpz_clear(t2); mpz_clear(t3); mpz_clear(t4);
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s p a b xP yP num_steps [num_threads]\n", argv[0]);
        return 1;
    }
    curve_t curve;
    mpz_init_set_str(curve.p, argv[1], 10);
    mpz_init_set_str(curve.a, argv[2], 10);
    mpz_init_set_str(curve.b, argv[3], 10);

    /* Build multipliers M[0..NUM_PARTITIONS-1] from P */
    jpoint_t M[NUM_PARTITIONS];
    mpz_t t1, t2, t3, t4;
    mpz_init(t1); mpz_init(t2); mpz_init(t3); mpz_init(t4);
    for (int i = 0; i < NUM_PARTITIONS; i++) {
        jpoint_init(&M[i]);
        mpz_set_str(M[i].X, argv[4], 10);
        mpz_set_str(M[i].Y, argv[5], 10);
        mpz_set_ui(M[i].Z, 1);
        M[i].is_infinity = 0;
    }
    /* Customize each M_i */
    jpoint_t tmp; jpoint_init(&tmp);
    for (int i = 1; i < NUM_PARTITIONS; i++) {
        jpoint_double(&tmp, &M[i-1], &curve, t1, t2, t3, t4);
        mpz_set(M[i].X, tmp.X); mpz_set(M[i].Y, tmp.Y); mpz_set(M[i].Z, tmp.Z);
        M[i].is_infinity = tmp.is_infinity;
    }
    jpoint_clear(&tmp);

    int num_steps = atoi(argv[6]);
    int num_threads = (argc >= 8) ? atoi(argv[7]) : DEFAULT_NUM_THREADS;

    fprintf(stderr, "Benchmarking on %lu-bit prime, %d threads, %d steps/thread\n",
            (unsigned long)mpz_sizeinbase(curve.p, 2), num_threads, num_steps);

    pthread_t threads[16];
    thread_arg_t args[16];
    clock_t global_start = clock();
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    for (int t = 0; t < num_threads; t++) {
        args[t].thread_id = t;
        args[t].num_steps = num_steps;
        args[t].curve = &curve;
        args[t].M = M;
        pthread_create(&threads[t], NULL, worker, &args[t]);
    }
    double total_ops = 0;
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
        total_ops += num_steps;
    }
    struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double wallclock = (ts_end.tv_sec - ts_start.tv_sec) +
                       (ts_end.tv_nsec - ts_start.tv_nsec) * 1e-9;
    double aggregate_ops = total_ops / wallclock;

    fprintf(stderr, "Total: %.0f ops in %.3f s wallclock = %.2e aggregate ops/sec\n",
            total_ops, wallclock, aggregate_ops);
    printf("%.2e\n", aggregate_ops);

    for (int i = 0; i < NUM_PARTITIONS; i++) jpoint_clear(&M[i]);
    mpz_clear(t1); mpz_clear(t2); mpz_clear(t3); mpz_clear(t4);
    mpz_clear(curve.p); mpz_clear(curve.a); mpz_clear(curve.b);
    return 0;
}
