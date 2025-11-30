/**
 * Optimized Matrix Multiplication - Compiler Optimization Testing
 * Same code as baseline, but compiled with different optimization flags
 * Used to measure impact of compiler optimizations (-O1, -O2, -O3, -Ofast)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#ifdef _WIN32
#include <winsock2.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <unistd.h>
#endif

// Get current time in milliseconds
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// Get hostname
void get_hostname(char *hostname, size_t size) {
    if (gethostname(hostname, size) != 0) {
        strncpy(hostname, "unknown", size);
    }
}

// Calculate GFLOPS
double calculate_gflops(int N, double time_ms) {
    double ops = 2.0 * N * N * N;
    double gflops = ops / (time_ms * 1e6);
    return gflops;
}

// Get optimization level from preprocessor
const char* get_opt_level() {
#if defined(__OPTIMIZE__)
    #if __OPTIMIZE__ >= 3
        return "O3";
    #elif __OPTIMIZE__ >= 2
        return "O2";
    #elif __OPTIMIZE__ >= 1
        return "O1";
    #else
        return "O0";
    #endif
#elif defined(__FAST_MATH__)
    return "Ofast";
#else
    return "O0";
#endif
}

// Initialize matrix with random values
void init_matrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Matrix multiplication: C = A * B
void matrix_multiply(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    // Parse matrix size from command line
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0 || N > 10000) {
            fprintf(stderr, "Error: Matrix size must be between 1 and 10000\n");
            return 1;
        }
    }
    
    // Seed random number generator
    srand(42);
    
    // Allocate matrices
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));
    
    if (!A || !B || !C) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(A); free(B); free(C);
        return 1;
    }
    
    // Initialize matrices
    init_matrix(A, N);
    init_matrix(B, N);
    
    // Get hostname
    char hostname[256];
    get_hostname(hostname, sizeof(hostname));
    
    // Get timestamp
    time_t now = time(NULL);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    // Get optimization level
    const char* opt_level = get_opt_level();
    
    // Create implementation name
    char impl_name[64];
    snprintf(impl_name, sizeof(impl_name), "optimized_%s", opt_level);
    
    // Perform matrix multiplication and measure time
    double start_time = get_time_ms();
    matrix_multiply(A, B, C, N);
    double end_time = get_time_ms();
    
    double elapsed_ms = end_time - start_time;
    double gflops = calculate_gflops(N, elapsed_ms);
    
    // Output in CSV format
    printf("%s,%s,%d,%.3f,%.3f,%.3f,0.000,0.000,N/A,%s,N/A\n",
           timestamp, impl_name, N, elapsed_ms, gflops, elapsed_ms, hostname);
    
    // Cleanup
    free(A);
    free(B);
    free(C);
    
    return 0;
}
