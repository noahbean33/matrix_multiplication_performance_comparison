#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

// Workload C: Sparse Linear Algebra (SpMV)
// GPU/CUDA implementation with CSR format

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CSR Matrix structure
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

// Generate sparse matrix (same as CPU version)
CSRMatrix generate_sparse_matrix(int num_rows, int num_cols, int min_nnz_per_row, int max_nnz_per_row) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> nnz_dist(min_nnz_per_row, max_nnz_per_row);
    std::uniform_int_distribution<int> col_dist(0, num_cols - 1);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    
    CSRMatrix mat;
    mat.num_rows = num_rows;
    mat.num_cols = num_cols;
    mat.row_ptr.resize(num_rows + 1);
    
    int current_nnz = 0;
    mat.row_ptr[0] = 0;
    
    for (int i = 0; i < num_rows; ++i) {
        int nnz_this_row = nnz_dist(rng);
        
        std::vector<int> cols;
        for (int j = 0; j < nnz_this_row; ++j) {
            int col;
            do {
                col = col_dist(rng);
            } while (std::find(cols.begin(), cols.end(), col) != cols.end());
            cols.push_back(col);
        }
        std::sort(cols.begin(), cols.end());
        
        for (int col : cols) {
            mat.col_indices.push_back(col);
            mat.values.push_back(val_dist(rng));
            current_nnz++;
        }
        
        mat.row_ptr[i + 1] = current_nnz;
    }
    
    mat.nnz = current_nnz;
    return mat;
}

// SpMV CUDA kernel - baseline version
// Each thread processes one row
// WARNING: This has poor memory coalescing on x vector!
__global__ void spmv_csr_kernel(
    const float* __restrict__ values,
    const int* __restrict__ col_indices,
    const int* __restrict__ row_ptr,
    const float* __restrict__ x,
    float* __restrict__ y,
    int num_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        // This loop causes UNCOALESCED memory access to x[]
        // because col_indices[j] is irregular/random per row
        for (int j = row_start; j < row_end; ++j) {
            int col = col_indices[j];
            sum += values[j] * x[col];  // IRREGULAR ACCESS!
        }
        
        y[row] = sum;
    }
}

void benchmark_spmv_gpu(int num_rows) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "SpMV GPU Benchmark - CSR Format" << std::endl;
    std::cout << "Matrix Size: " << num_rows << " rows" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Generate matrix
    int num_cols = num_rows;
    int min_nnz = 5;
    int max_nnz = 9;
    
    std::cout << "Generating sparse matrix (nnz/row: " << min_nnz << "-" << max_nnz << ")..." << std::endl;
    auto gen_start = std::chrono::high_resolution_clock::now();
    CSRMatrix A = generate_sparse_matrix(num_rows, num_cols, min_nnz, max_nnz);
    auto gen_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gen_time = gen_end - gen_start;
    
    std::cout << "Matrix generated in " << gen_time.count() << " s" << std::endl;
    std::cout << "Total non-zeros: " << A.nnz << std::endl;
    std::cout << "Avg nnz/row: " << (double)A.nnz / A.num_rows << std::endl;
    
    // Host vectors
    std::vector<float> h_x(num_cols, 1.0f);
    std::vector<float> h_y(num_rows, 0.0f);
    
    // Device pointers
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptr;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_values, A.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, A.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, num_rows * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_values, A.values.data(), A.nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_indices, A.col_indices.data(), A.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), num_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Kernel configuration
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    
    std::cout << "\nKernel Config: " << grid_size << " blocks, " << block_size << " threads/block" << std::endl;
    
    // Warmup
    spmv_csr_kernel<<<grid_size, block_size>>>(d_values, d_col_indices, d_row_ptr, d_x, d_y, num_rows);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_runs = 10;
    std::cout << "Running " << num_runs << " iterations..." << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < num_runs; ++run) {
        spmv_csr_kernel<<<grid_size, block_size>>>(d_values, d_col_indices, d_row_ptr, d_x, d_y, num_rows);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double avg_time = (milliseconds / 1000.0) / num_runs;
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate metrics
    double bytes_per_spmv = A.nnz * sizeof(float)          // values
                          + A.nnz * sizeof(int)            // col_indices
                          + (A.num_rows + 1) * sizeof(int) // row_ptr
                          + A.nnz * sizeof(float)          // x reads (UNCOALESCED!)
                          + A.num_rows * sizeof(float);    // y writes
    
    double gbytes_per_sec = (bytes_per_spmv / avg_time) / 1e9;
    
    double flops = 2.0 * A.nnz;
    double gflops_per_sec = (flops / avg_time) / 1e9;
    
    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n----- Performance Metrics -----" << std::endl;
    std::cout << "Avg Time per SpMV: " << avg_time * 1000.0 << " ms" << std::endl;
    std::cout << "Throughput:        " << gflops_per_sec << " GFLOP/s" << std::endl;
    std::cout << "Bandwidth:         " << gbytes_per_sec << " GB/s" << std::endl;
    std::cout << "Memory Reads:      " << bytes_per_spmv / 1e6 << " MB" << std::endl;
    std::cout << "Arithmetic Intensity: " << flops / bytes_per_spmv << " FLOP/Byte" << std::endl;
    
    std::cout << "\n----- Memory Access Pattern -----" << std::endl;
    std::cout << "WARNING: Irregular col_indices[] causes UNCOALESCED memory access!" << std::endl;
    std::cout << "x[col_indices[j]] reads are scattered across memory." << std::endl;
    std::cout << "Expected: Poor cache hit rate and memory throughput." << std::endl;
    
    // Verification
    std::cout << "\n----- Verification -----" << std::endl;
    std::cout << "Result checksum: " << h_y[0] + h_y[num_rows/2] + h_y[num_rows-1] << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload C: Sparse Matrix-Vector Multiplication" << std::endl;
    std::cout << "Architecture: GPU (CUDA)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    
    // From benchmarks.yml: sizes [1e5, 1e6]
    std::vector<int> sizes = {100000, 1000000};
    
    for (int size : sizes) {
        benchmark_spmv_gpu(size);
    }
    
    return 0;
}
