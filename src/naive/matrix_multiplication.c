/**
 * @file matrix_multiplication.c
 * @brief Naive matrix multiplication implementation with performance benchmarking
 * 
 * This implementation uses the standard O(n^3) algorithm for matrix multiplication
 * without any optimizations. It serves as a baseline for performance comparison.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

/**
 * @brief Allocates and initializes an n x n matrix with random values in [0, 1)
 * @param n The dimension of the square matrix
 * @return Pointer to the allocated matrix, or exits on allocation failure
 */
double** generate_random_matrix(int n) {
    double** M = (double**)malloc(n * sizeof(double*));
    if (M == NULL) {
        perror("Failed to allocate memory for row pointers");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < n; i++) {
        M[i] = (double*)malloc(n * sizeof(double));
        if (M[i] == NULL) {
            perror("Failed to allocate memory for matrix row");
            for (int j = 0; j < i; j++) {
                free(M[j]);
            }
            free(M);
            exit(EXIT_FAILURE);
        }
        
        for (int j = 0; j < n; j++) {
            M[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
    return M;
}

/**
 * @brief Performs naive matrix multiplication: C = A * B
 * @param A First input matrix (n x n)
 * @param B Second input matrix (n x n)
 * @param C Output matrix (n x n), assumed to be pre-allocated
 * @param n Dimension of the square matrices
 */
void matrix_multiply(double** A, double** B, double** C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

/**
 * @brief Frees a dynamically allocated n x n matrix
 * @param M Pointer to the matrix to be freed
 * @param n Dimension of the square matrix
 */
void free_matrix(double** M, int n) {
    for (int i = 0; i < n; i++) {
        free(M[i]);
    }
    free(M);
}

/**
 * @brief Gets high-resolution time in milliseconds
 * @return Current time in milliseconds with microsecond precision
 */
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/**
 * @brief Calculates GFLOPS for matrix multiplication
 * @param n Matrix dimension
 * @param time_ms Execution time in milliseconds
 * @return GFLOPS (billions of floating point operations per second)
 */
double calculate_gflops(int n, double time_ms) {
    double operations = 2.0 * n * n * n;
    double time_seconds = time_ms / 1000.0;
    return operations / (time_seconds * 1e9);
}

/**
 * @brief Gets the hostname of the current machine
 * @param buffer Buffer to store the hostname
 * @param size Size of the buffer
 */
void get_hostname(char* buffer, size_t size) {
    if (gethostname(buffer, size) != 0) {
        strncpy(buffer, "unknown", size - 1);
        buffer[size - 1] = '\0';
    }
}

/**
 * @brief Benchmarks matrix multiplication across various matrix sizes
 * 
 * Tests matrix multiplication performance according to README specifications:
 * - Matrix sizes: 64, 128, 256, 512, 1024, 2048, 4096
 * - 10 iterations per size
 * - CSV output: timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node
 * 
 * @return 0 on successful completion
 */
int main() {
    const int matrix_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    const int num_sizes = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);
    const int iterations = 10;
    const char* implementation = "naive_ijk";
    const int threads = 1;
    const int processes = 1;
    
    char hostname[256];
    get_hostname(hostname, sizeof(hostname));
    
    srand((unsigned)time(NULL));
    
    printf("timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node\n");

    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
        int n = matrix_sizes[size_idx];
        
        for (int iter = 0; iter < iterations; iter++) {
            double** A = generate_random_matrix(n);
            double** B = generate_random_matrix(n);

            double** C = (double**)malloc(n * sizeof(double*));
            if (C == NULL) {
                perror("Failed to allocate memory for result matrix");
                free_matrix(A, n);
                free_matrix(B, n);
                exit(EXIT_FAILURE);
            }
            
            for (int i = 0; i < n; i++) {
                C[i] = (double*)calloc(n, sizeof(double));
                if (C[i] == NULL) {
                    perror("Failed to allocate memory for result matrix row");
                    for (int j = 0; j < i; j++) {
                        free(C[j]);
                    }
                    free(C);
                    free_matrix(A, n);
                    free_matrix(B, n);
                    exit(EXIT_FAILURE);
                }
            }

            double start_time = get_time_ms();
            matrix_multiply(A, B, C, n);
            double end_time = get_time_ms();

            double elapsed_ms = end_time - start_time;
            double gflops = calculate_gflops(n, elapsed_ms);
            
            time_t now = time(NULL);
            struct tm* tm_info = localtime(&now);
            char timestamp[64];
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

            printf("%s,%s,%d,%.6f,%.6f,%d,%d,%s\n",
                   timestamp, implementation, n, elapsed_ms, gflops,
                   threads, processes, hostname);

            free_matrix(A, n);
            free_matrix(B, n);
            free_matrix(C, n);
        }
    }

    return 0;
}
