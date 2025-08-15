#include "../include/mm.h"

// Placeholder: call blocked for now. Later replace with AVX2/AVX-512 micro-kernel.
void mm_microkernel_avx(double** A, double** B, double** C, int n, int Mr, int Nr) {
    (void)Mr; (void)Nr;
    mm_blocked(A, B, C, n, 128, 128, 128);
}
