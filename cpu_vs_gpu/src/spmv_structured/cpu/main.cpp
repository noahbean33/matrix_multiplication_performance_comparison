#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>

// Workload C: Sparse Linear Algebra (SpMV)
// CPU implementation with CSR format

// CSR Matrix structure
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<float> values;      // Non-zero values
    std::vector<int> col_indices;   // Column indices
    std::vector<int> row_ptr;       // Row pointers (size = num_rows + 1)
};

// Generate a structured sparse matrix with controlled sparsity
// Pattern: 5-9 non-zeros per row (irregular like real sparse matrices)
CSRMatrix generate_sparse_matrix(int num_rows, int num_cols, int min_nnz_per_row, int max_nnz_per_row) {
    std::mt19937 rng(42); // Fixed seed for reproducibility
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
        
        // Generate unique column indices for this row
        std::vector<int> cols;
        for (int j = 0; j < nnz_this_row; ++j) {
            int col;
            do {
                col = col_dist(rng);
            } while (std::find(cols.begin(), cols.end(), col) != cols.end());
            cols.push_back(col);
        }
        std::sort(cols.begin(), cols.end());
        
        // Add to CSR structure
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

// SpMV kernel: y = A * x
void spmv_csr(const CSRMatrix& A, const std::vector<float>& x, std::vector<float>& y) {
    for (int i = 0; i < A.num_rows; ++i) {
        float sum = 0.0f;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = sum;
    }
}

void benchmark_spmv(int num_rows) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "SpMV CPU Benchmark - CSR Format" << std::endl;
    std::cout << "Matrix Size: " << num_rows << " rows" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Generate matrix (assume square for simplicity)
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
    
    // Initialize input and output vectors
    std::vector<float> x(num_cols, 1.0f);
    std::vector<float> y(num_rows, 0.0f);
    
    // Warmup run
    spmv_csr(A, x, y);
    
    // Benchmark runs
    const int num_runs = 10;
    std::cout << "\nRunning " << num_runs << " iterations..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < num_runs; ++run) {
        spmv_csr(A, x, y);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / num_runs;
    
    // Calculate metrics
    // Data movement: read all values, read corresponding x elements, write y
    double bytes_per_spmv = A.nnz * sizeof(float)        // values
                          + A.nnz * sizeof(int)          // col_indices
                          + (A.num_rows + 1) * sizeof(int) // row_ptr (amortized)
                          + A.nnz * sizeof(float)        // x reads (irregular!)
                          + A.num_rows * sizeof(float);  // y writes
    
    double gbytes_per_sec = (bytes_per_spmv / avg_time) / 1e9;
    
    // FLOPs: 2 operations (mul + add) per non-zero
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
    
    // Verification
    std::cout << "\n----- Verification -----" << std::endl;
    std::cout << "Result checksum: " << y[0] + y[num_rows/2] + y[num_rows-1] << std::endl;
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload C: Sparse Matrix-Vector Multiplication" << std::endl;
    std::cout << "Architecture: CPU" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // From benchmarks.yml: sizes [1e5, 1e6]
    std::vector<int> sizes = {100000, 1000000};
    
    for (int size : sizes) {
        benchmark_spmv(size);
    }
    
    return 0;
}
