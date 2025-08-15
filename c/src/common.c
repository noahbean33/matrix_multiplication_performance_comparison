#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/mm.h"

static void* aligned_malloc(size_t size, size_t alignment) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
    return ptr;
#endif
}

static void aligned_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

double** mm_alloc_matrix(int n) {
    double** M = (double**)malloc(sizeof(double*) * n);
    if (!M) return NULL;
    for (int i = 0; i < n; ++i) {
        M[i] = (double*)aligned_malloc(sizeof(double) * n, 64);
        if (!M[i]) {
            for (int k = 0; k < i; ++k) aligned_free(M[k]);
            free(M);
            return NULL;
        }
        memset(M[i], 0, sizeof(double) * n);
    }
    return M;
}

void mm_free_matrix(double** M, int n) {
    if (!M) return;
    for (int i = 0; i < n; ++i) aligned_free(M[i]);
    free(M);
}

void mm_fill_random(double** M, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M[i][j] = (double)rand() / RAND_MAX;
}

void mm_print_csv(int n, double seconds, const char* variant, int threads) {
    if (!variant) variant = "unknown";
    // n,time_seconds,variant,threads
    printf("%d,%.6f,%s,%d\n", n, seconds, variant, threads);
}
