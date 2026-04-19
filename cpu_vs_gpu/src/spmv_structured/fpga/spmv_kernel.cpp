#include "spmv_kernel.h"

// FPGA Kernel for SpMV using CSR format
// This demonstrates custom dataflow architecture for irregular memory access

// Pipelined SpMV kernel with burst reads
void spmv_csr_kernel(
    const float* values,        // Non-zero values array
    const int* col_indices,     // Column indices array
    const int* row_ptr,         // Row pointer array
    const float* x,             // Input vector
    float* y,                   // Output vector
    int num_rows,               // Number of rows
    int nnz                     // Total non-zeros
) {
#pragma HLS INTERFACE m_axi port=values offset=slave bundle=gmem0 depth=10000000
#pragma HLS INTERFACE m_axi port=col_indices offset=slave bundle=gmem1 depth=10000000
#pragma HLS INTERFACE m_axi port=row_ptr offset=slave bundle=gmem2 depth=1000000
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem3 depth=1000000
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem4 depth=1000000
#pragma HLS INTERFACE s_axilite port=num_rows
#pragma HLS INTERFACE s_axilite port=nnz
#pragma HLS INTERFACE s_axilite port=return

    // Local buffers for row processing
    // Strategy: Pipeline row processing with buffered reads
    
    ROW_LOOP:
    for (int row = 0; row < num_rows; row++) {
#pragma HLS LOOP_TRIPCOUNT min=100000 max=1000000 avg=500000
        
        // Read row boundaries
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        int row_nnz = row_end - row_start;
        
        float sum = 0.0f;
        
        // Process non-zeros in this row
        // This is where irregular access happens - but FPGA can:
        // 1. Pipeline the loop
        // 2. Use custom memory arbitration
        // 3. Build specialized caching/buffering
        NNZ_LOOP:
        for (int j = row_start; j < row_end; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=5 max=9 avg=7
            
            int col = col_indices[j];
            float val = values[j];
            float x_val = x[col];  // Irregular access - but pipelined!
            
            sum += val * x_val;
        }
        
        // Write result
        y[row] = sum;
    }
}

// Optimized version with BRAM buffering for x vector
// This version shows how FPGAs can build custom cache structures
void spmv_csr_kernel_optimized(
    const float* values,
    const int* col_indices,
    const int* row_ptr,
    const float* x,
    float* y,
    int num_rows,
    int nnz
) {
#pragma HLS INTERFACE m_axi port=values offset=slave bundle=gmem0 depth=10000000
#pragma HLS INTERFACE m_axi port=col_indices offset=slave bundle=gmem1 depth=10000000
#pragma HLS INTERFACE m_axi port=row_ptr offset=slave bundle=gmem2 depth=1000000
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem3 depth=1000000
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem4 depth=1000000
#pragma HLS INTERFACE s_axilite port=num_rows
#pragma HLS INTERFACE s_axilite port=nnz
#pragma HLS INTERFACE s_axilite port=return

    // BRAM buffer for frequently accessed x elements
    // Trade-off: BRAM usage vs DDR bandwidth
    const int BUFFER_SIZE = 1024;
    float x_buffer[BUFFER_SIZE];
#pragma HLS BIND_STORAGE variable=x_buffer type=RAM_2P impl=BRAM
    
    // This is a simplified example - real implementation would use:
    // - Intelligent prefetching
    // - LRU cache replacement
    // - Burst reads for sequential patterns
    
    ROW_LOOP:
    for (int row = 0; row < num_rows; row++) {
#pragma HLS LOOP_TRIPCOUNT min=100000 max=1000000 avg=500000
        
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        float sum = 0.0f;
        
        NNZ_LOOP:
        for (int j = row_start; j < row_end; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=5 max=9 avg=7
            
            int col = col_indices[j];
            float val = values[j];
            
            // Use buffer if available, otherwise DDR
            float x_val;
            if (col < BUFFER_SIZE) {
                x_val = x_buffer[col];
            } else {
                x_val = x[col];
            }
            
            sum += val * x_val;
        }
        
        y[row] = sum;
    }
}

// Dataflow version - splits computation into stages
// This shows advanced FPGA optimization: task-level pipelining
void spmv_csr_kernel_dataflow(
    const float* values,
    const int* col_indices,
    const int* row_ptr,
    const float* x,
    float* y,
    int num_rows,
    int nnz
) {
#pragma HLS INTERFACE m_axi port=values offset=slave bundle=gmem0 depth=10000000
#pragma HLS INTERFACE m_axi port=col_indices offset=slave bundle=gmem1 depth=10000000
#pragma HLS INTERFACE m_axi port=row_ptr offset=slave bundle=gmem2 depth=1000000
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem3 depth=1000000
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem4 depth=1000000
#pragma HLS INTERFACE s_axilite port=num_rows
#pragma HLS INTERFACE s_axilite port=nnz
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS DATAFLOW

    // In a real implementation, we'd split this into:
    // 1. Read stage (fetch row_ptr, values, col_indices)
    // 2. Compute stage (MAC operations)
    // 3. Write stage (output y)
    // Connected via FIFOs for streaming
    
    // Simplified single-stage for this example
    ROW_LOOP:
    for (int row = 0; row < num_rows; row++) {
#pragma HLS LOOP_TRIPCOUNT min=100000 max=1000000 avg=500000
        
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        float sum = 0.0f;
        
        NNZ_LOOP:
        for (int j = row_start; j < row_end; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=5 max=9 avg=7
            
            int col = col_indices[j];
            sum += values[j] * x[col];
        }
        
        y[row] = sum;
    }
}
