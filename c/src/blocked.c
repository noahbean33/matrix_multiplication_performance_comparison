#include "../include/mm.h"

#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

// Simple blocked version; tune Mc,Nc,Kc later
void mm_blocked(double** A, double** B, double** C, int n, int Mc, int Nc, int Kc) {
    if (Mc <= 0) Mc = 64;
    if (Nc <= 0) Nc = 64;
    if (Kc <= 0) Kc = 64;
    for (int ii = 0; ii < n; ii += Mc)
    for (int kk = 0; kk < n; kk += Kc)
    for (int jj = 0; jj < n; jj += Nc) {
        int i_end = MIN(ii + Mc, n);
        int k_end = MIN(kk + Kc, n);
        int j_end = MIN(jj + Nc, n);
        for (int i = ii; i < i_end; ++i)
            for (int k = kk; k < k_end; ++k) {
                double aik = A[i][k];
                for (int j = jj; j < j_end; ++j) {
                    C[i][j] += aik * B[k][j];
                }
            }
    }
}
