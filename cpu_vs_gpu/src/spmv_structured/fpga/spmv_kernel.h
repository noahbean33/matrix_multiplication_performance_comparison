#ifndef SPMV_KERNEL_H
#define SPMV_KERNEL_H

// SpMV FPGA Kernel Headers

// Baseline pipelined version
void spmv_csr_kernel(
    const float* values,
    const int* col_indices,
    const int* row_ptr,
    const float* x,
    float* y,
    int num_rows,
    int nnz
);

// Optimized version with BRAM buffering
void spmv_csr_kernel_optimized(
    const float* values,
    const int* col_indices,
    const int* row_ptr,
    const float* x,
    float* y,
    int num_rows,
    int nnz
);

// Dataflow version with task-level pipelining
void spmv_csr_kernel_dataflow(
    const float* values,
    const int* col_indices,
    const int* row_ptr,
    const float* x,
    float* y,
    int num_rows,
    int nnz
);

#endif // SPMV_KERNEL_H
