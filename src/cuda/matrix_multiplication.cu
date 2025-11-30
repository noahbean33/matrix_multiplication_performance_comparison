/**
 * @file matrix_multiplication.cu
 * @brief CUDA matrix multiplication benchmarking suite
 * 
 * Implements multiple CUDA kernels for matrix multiplication performance analysis:
 * - Naive global memory implementation
 * - Tiled shared memory implementation
 * - Optimized vectorized implementation
 * 
 * Outputs CSV format: timestamp,implementation,matrix_size,execution_time_ms,
 *                     gflops,kernel_time_ms,h2d_time_ms,d2h_time_ms,block_size,node
 */

#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>

/**
 * @brief CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * @brief Gets high-resolution time in milliseconds
 */
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/**
 * @brief Calculates GFLOPS for matrix multiplication
 */
double calculate_gflops(int n, double time_ms) {
    double operations = 2.0 * n * n * n;
    double time_seconds = time_ms / 1000.0;
    return operations / (time_seconds * 1e9);
}

/**
 * @brief Gets the hostname of the current machine
 */
void get_hostname(char* buffer, size_t size) {
    if (gethostname(buffer, size) != 0) {
        strncpy(buffer, "unknown", size - 1);
        buffer[size - 1] = '\0';
    }
}

/**
 * @brief CPU naive matrix multiplication for verification
 */
void cpu_mm_naive(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/**
 * @brief Naive CUDA kernel - global memory only
 * @param A Input matrix A (n x n)
 * @param B Input matrix B (n x n)
 * @param C Output matrix C (n x n)
 * @param n Matrix dimension
 */
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

/**
 * @brief Tiled CUDA kernel with shared memory
 * @param A Input matrix A (n x n)
 * @param B Input matrix B (n x n)
 * @param C Output matrix C (n x n)
 * @param n Matrix dimension
 * @param TILE_SIZE Tile dimension (must match block size)
 */
template<int TILE_SIZE>
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tiles into shared memory
        if (row < n && (t * TILE_SIZE + threadIdx.x) < n)
            sA[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < n && col < n)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

/**
 * @brief Verifies GPU result against CPU result
 * @return true if results match within tolerance
 */
bool verify_result(const float* cpu_result, const float* gpu_result, int n) {
    const float epsilon = 1e-3f;
    for (int i = 0; i < n * n; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Benchmarks a CUDA kernel
 */
template<typename KernelFunc>
void benchmark_kernel(
    const char* kernel_name,
    KernelFunc kernel,
    int n,
    int block_size,
    const float* h_A,
    const float* h_B,
    const float* h_C_cpu,
    char* hostname
) {
    size_t bytes = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;
    float *h_C_gpu = new float[n * n];

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // H2D transfer timing
    double h2d_start = get_time_ms();
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    double h2d_time = get_time_ms() - h2d_start;

    // Kernel execution timing
    dim3 threads(block_size, block_size);
    dim3 blocks((n + block_size - 1) / block_size, (n + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<blocks, threads>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));

    // D2H transfer timing
    double d2h_start = get_time_ms();
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
    double d2h_time = get_time_ms() - d2h_start;

    // Calculate total time and GFLOPS
    double total_time = h2d_time + kernel_time_ms + d2h_time;
    double gflops = calculate_gflops(n, total_time);
    double kernel_gflops = calculate_gflops(n, kernel_time_ms);

    // Verify correctness
    bool correct = verify_result(h_C_cpu, h_C_gpu, n);

    // Get timestamp
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

    // Output CSV format
    printf("%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%dx%d,%s,%s\n",
           timestamp, kernel_name, n, total_time, gflops,
           kernel_time_ms, h2d_time, d2h_time,
           block_size, block_size, hostname,
           correct ? "PASS" : "FAIL");

    // Cleanup
    delete[] h_C_gpu;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

/**
 * @brief Main benchmarking program
 */
int main(int argc, char** argv) {
    // Matrix sizes from README (Experiment 1)
    const int matrix_sizes[] = {512, 1024, 2048, 4096};
    const int num_sizes = sizeof(matrix_sizes) / sizeof(matrix_sizes[0]);
    
    // Block sizes from README (Experiment 2)
    const int block_sizes[] = {16, 32};
    const int num_block_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    char hostname[256];
    get_hostname(hostname, sizeof(hostname));

    // Print CSV header
    printf("timestamp,implementation,matrix_size,total_time_ms,total_gflops,kernel_time_ms,h2d_time_ms,d2h_time_ms,block_size,node,verification\n");

    // Iterate through matrix sizes
    for (int size_idx = 0; size_idx < num_sizes; size_idx++) {
        int n = matrix_sizes[size_idx];
        
        // Allocate and initialize host memory
        float* h_A = new float[n * n];
        float* h_B = new float[n * n];
        float* h_C_cpu = new float[n * n];

        srand(42);  // Fixed seed for reproducibility
        for (int i = 0; i < n * n; i++) {
            h_A[i] = static_cast<float>(rand() % 10);
            h_B[i] = static_cast<float>(rand() % 10);
        }

        // Compute CPU reference result
        cpu_mm_naive(h_A, h_B, h_C_cpu, n);

        // Test different block sizes
        for (int bs_idx = 0; bs_idx < num_block_sizes; bs_idx++) {
            int block_size = block_sizes[bs_idx];

            // Benchmark naive kernel
            benchmark_kernel("cuda_naive", matmul_naive, n, block_size,
                           h_A, h_B, h_C_cpu, hostname);

            // Benchmark tiled kernel
            if (block_size == 16) {
                benchmark_kernel("cuda_tiled_16", matmul_tiled<16>, n, block_size,
                               h_A, h_B, h_C_cpu, hostname);
            } else if (block_size == 32) {
                benchmark_kernel("cuda_tiled_32", matmul_tiled<32>, n, block_size,
                               h_A, h_B, h_C_cpu, hostname);
            }
        }

        // Cleanup
        delete[] h_A;
        delete[] h_B;
        delete[] h_C_cpu;
    }

    return 0;
}
