/**
 * MPI Matrix Multiplication - Distributed Implementation
 * Uses row-based decomposition to distribute work across MPI processes
 * Each process computes a subset of rows of the result matrix
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>

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

// Initialize matrix with random values (same seed for all processes)
void init_matrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Local matrix multiplication: compute rows [start_row, end_row) of C
void matrix_multiply_rows(const float *A, const float *B, float *C, 
                          int N, int start_row, int num_rows) {
    for (int i = 0; i < num_rows; i++) {
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
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse matrix size from command line
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
        if (N <= 0 || N > 10000) {
            if (rank == 0) {
                fprintf(stderr, "Error: Matrix size must be between 1 and 10000\n");
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    // Calculate rows per process
    int rows_per_proc = N / size;
    int remainder = N % size;
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int num_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Allocate matrices
    float *A_local = (float*)malloc(num_rows * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C_local = (float*)malloc(num_rows * N * sizeof(float));
    float *A_full = NULL;
    float *C_full = NULL;
    
    if (rank == 0) {
        A_full = (float*)malloc(N * N * sizeof(float));
        C_full = (float*)malloc(N * N * sizeof(float));
    }
    
    if (!A_local || !B || !C_local || (rank == 0 && (!A_full || !C_full))) {
        if (rank == 0) {
            fprintf(stderr, "Error: Memory allocation failed\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Initialize matrices on rank 0
    if (rank == 0) {
        srand(42);
        init_matrix(A_full, N);
        init_matrix(B, N);
    }
    
    // Broadcast B to all processes
    MPI_Bcast(B, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Scatter rows of A to all processes
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            int rows = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * N;
            displs[i] = (i * rows_per_proc + (i < remainder ? i : remainder)) * N;
        }
    }
    
    MPI_Scatterv(A_full, sendcounts, displs, MPI_FLOAT,
                 A_local, num_rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    // Perform local matrix multiplication
    matrix_multiply_rows(A_local, B, C_local, N, start_row, num_rows);
    
    // Gather results
    MPI_Gatherv(C_local, num_rows * N, MPI_FLOAT,
                C_full, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Synchronize after computation
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    // Output results from rank 0
    if (rank == 0) {
        double elapsed_ms = (end_time - start_time) * 1000.0;
        double gflops = calculate_gflops(N, elapsed_ms);
        
        // Get hostname
        char hostname[256];
        get_hostname(hostname, sizeof(hostname));
        
        // Get timestamp
        time_t now = time(NULL);
        char timestamp[64];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
        
        // Create implementation name
        char impl_name[64];
        snprintf(impl_name, sizeof(impl_name), "mpi_%dp", size);
        
        // Output in CSV format
        printf("%s,%s,%d,%.3f,%.3f,%.3f,0.000,0.000,%dp,%s,N/A\n",
               timestamp, impl_name, N, elapsed_ms, gflops, elapsed_ms, 
               size, hostname);
    }
    
    // Cleanup
    free(A_local);
    free(B);
    free(C_local);
    if (rank == 0) {
        free(A_full);
        free(C_full);
        free(sendcounts);
        free(displs);
    }
    
    MPI_Finalize();
    return 0;
}
