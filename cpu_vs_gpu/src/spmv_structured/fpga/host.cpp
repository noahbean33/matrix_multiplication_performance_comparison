#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cstring>

#ifdef XILINX_FPGA
#include "xcl2.hpp"
#endif

#include "spmv_kernel.h"

// CSR Matrix structure
struct CSRMatrix {
    int num_rows;
    int num_cols;
    int nnz;
    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

// Generate sparse matrix (same as CPU/GPU versions)
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

// Software reference implementation
void spmv_csr_sw(const CSRMatrix& A, const std::vector<float>& x, std::vector<float>& y) {
    for (int i = 0; i < A.num_rows; ++i) {
        float sum = 0.0f;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = sum;
    }
}

void benchmark_spmv_fpga_sw(int num_rows) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "SpMV FPGA Benchmark - Software Emulation" << std::endl;
    std::cout << "Matrix Size: " << num_rows << " rows" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Generate matrix
    int num_cols = num_rows;
    int min_nnz = 5;
    int max_nnz = 9;
    
    std::cout << "Generating sparse matrix (nnz/row: " << min_nnz << "-" << max_nnz << ")..." << std::endl;
    CSRMatrix A = generate_sparse_matrix(num_rows, num_cols, min_nnz, max_nnz);
    
    std::cout << "Total non-zeros: " << A.nnz << std::endl;
    std::cout << "Avg nnz/row: " << (double)A.nnz / A.num_rows << std::endl;
    
    // Initialize vectors
    std::vector<float> x(num_cols, 1.0f);
    std::vector<float> y(num_rows, 0.0f);
    std::vector<float> y_ref(num_rows, 0.0f);
    
    // Software reference
    spmv_csr_sw(A, x, y_ref);
    
    // Call HLS kernel (software emulation)
    std::cout << "\nRunning HLS kernel (software simulation)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    spmv_csr_kernel(
        A.values.data(),
        A.col_indices.data(),
        A.row_ptr.data(),
        x.data(),
        y.data(),
        A.num_rows,
        A.nnz
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Verify results
    bool correct = true;
    float max_error = 0.0f;
    for (int i = 0; i < num_rows; ++i) {
        float error = std::abs(y[i] - y_ref[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-3) {
            correct = false;
            if (i < 10) {  // Print first few errors
                std::cout << "Mismatch at " << i << ": " << y[i] << " vs " << y_ref[i] << std::endl;
            }
        }
    }
    
    std::cout << "\n----- Verification -----" << std::endl;
    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    
    // Performance metrics (for SW emulation)
    double bytes_per_spmv = A.nnz * sizeof(float) + A.nnz * sizeof(int) +
                           (A.num_rows + 1) * sizeof(int) + A.nnz * sizeof(float) +
                           A.num_rows * sizeof(float);
    double flops = 2.0 * A.nnz;
    
    std::cout << "\n----- Software Emulation Metrics -----" << std::endl;
    std::cout << "Time: " << elapsed.count() * 1000.0 << " ms" << std::endl;
    std::cout << "Note: Hardware synthesis required for actual FPGA performance" << std::endl;
    
    std::cout << "\n----- Expected FPGA Characteristics -----" << std::endl;
    std::cout << "Pipeline II: 1 cycle (inner loop)" << std::endl;
    std::cout << "BRAM Usage: ~" << (1024 * sizeof(float)) / 1024 << " KB (for x_buffer)" << std::endl;
    std::cout << "Memory Ports: 5 independent AXI ports for parallel access" << std::endl;
    std::cout << "Irregular Access Handling: Custom arbitration logic" << std::endl;
    std::cout << "Dataflow Potential: Task-level pipelining across rows" << std::endl;
}

#ifdef XILINX_FPGA
void benchmark_spmv_fpga_hw(int num_rows, const std::string& xclbin_file) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "SpMV FPGA Benchmark - Hardware" << std::endl;
    std::cout << "Matrix Size: " << num_rows << " rows" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // This section would contain actual FPGA hardware execution
    // Using Xilinx XRT runtime for PCIe FPGA cards
    
    std::cout << "Hardware execution requires:" << std::endl;
    std::cout << "1. Synthesized bitstream (xclbin file)" << std::endl;
    std::cout << "2. Xilinx XRT runtime installed" << std::endl;
    std::cout << "3. Compatible FPGA board (e.g., Alveo U250)" << std::endl;
    std::cout << "\nPlease run synthesis first: make build" << std::endl;
}
#endif

int main(int argc, char** argv) {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload C: Sparse Matrix-Vector Multiplication" << std::endl;
    std::cout << "Architecture: FPGA (Xilinx Vitis HLS)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    bool hw_mode = false;
    std::string xclbin_file = "";
    
    if (argc > 1 && std::string(argv[1]) == "--hw") {
        hw_mode = true;
        if (argc > 2) {
            xclbin_file = argv[2];
        }
    }
    
    std::vector<int> sizes = {100000, 1000000};
    
    if (hw_mode) {
#ifdef XILINX_FPGA
        for (int size : sizes) {
            benchmark_spmv_fpga_hw(size, xclbin_file);
        }
#else
        std::cout << "Hardware mode not available. Rebuild with Xilinx tools." << std::endl;
        return 1;
#endif
    } else {
        std::cout << "\nRunning in SOFTWARE EMULATION mode" << std::endl;
        std::cout << "Use --hw <xclbin> for hardware execution\n" << std::endl;
        
        for (int size : sizes) {
            benchmark_spmv_fpga_sw(size);
        }
    }
    
    return 0;
}
