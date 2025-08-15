#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/mm.h"

static double now_seconds(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv) {
    int n = 1024;
    const char* variant = "naive";
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) variant = argv[2];

    double** A = mm_alloc_matrix(n);
    double** B = mm_alloc_matrix(n);
    double** C = mm_alloc_matrix(n);
    if (!A || !B || !C) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    mm_fill_random(A, n, 1u);
    mm_fill_random(B, n, 2u);

    // Warm-up
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = 0.0;
    if (strcmp(variant, "naive") == 0) mm_naive(A,B,C,n);
    else if (strcmp(variant, "blocked") == 0) mm_blocked(A,B,C,n,64,64,64);
    else if (strcmp(variant, "microkernel_avx") == 0) mm_microkernel_avx(A,B,C,n,8,8);
    else mm_naive(A,B,C,n);

    // Timed
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = 0.0;
    double t0 = now_seconds();
    if (strcmp(variant, "naive") == 0) mm_naive(A,B,C,n);
    else if (strcmp(variant, "blocked") == 0) mm_blocked(A,B,C,n,128,128,128);
    else if (strcmp(variant, "microkernel_avx") == 0) mm_microkernel_avx(A,B,C,n,8,8);
    else mm_naive(A,B,C,n);
    double t1 = now_seconds();

    mm_print_csv(n, t1 - t0, variant, 1);

    mm_free_matrix(A, n);
    mm_free_matrix(B, n);
    mm_free_matrix(C, n);
    return 0;
}
