#ifndef MM_H
#define MM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n;              // matrix dimension (n x n)
    int threads;        // OpenMP threads (if used)
    const char* variant;// name: naive|blocked|microkernel_avx
} mm_config_t;

// Allocate n x n double matrix, aligned to 64 bytes if possible
double** mm_alloc_matrix(int n);
void     mm_free_matrix(double** M, int n);
void     mm_fill_random(double** M, int n, unsigned seed);

// Variants (all compute C = A * B)
void mm_naive(double** A, double** B, double** C, int n);
void mm_blocked(double** A, double** B, double** C, int n, int Mc, int Nc, int Kc);
void mm_microkernel_avx(double** A, double** B, double** C, int n, int Mr, int Nr);

// CSV helper
void mm_print_csv(int n, double seconds, const char* variant, int threads);

#ifdef __cplusplus
}
#endif

#endif // MM_H
