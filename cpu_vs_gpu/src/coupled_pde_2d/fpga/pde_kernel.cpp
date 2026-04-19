#include "pde_kernel.h"

// FPGA Kernel for Coupled PDE (Navier-Stokes + Temperature)
// This workload shows FPGA's BRAM PRESSURE challenge!

// Simple version: All fields in DDR (large grids)
// Shows the problem: 5 fields x N^2 exceeds BRAM capacity
void pde_kernel_ddr(
    const float* u_in,
    const float* v_in,
    const float* p_in,
    const float* T_in,
    float* u_out,
    float* v_out,
    float* p_out,
    float* T_out,
    int nx,
    int ny,
    float dt,
    float nu,
    float alpha,
    float g_beta,
    int jacobi_iters
) {
#pragma HLS INTERFACE m_axi port=u_in offset=slave bundle=gmem0 depth=1048576
#pragma HLS INTERFACE m_axi port=v_in offset=slave bundle=gmem1 depth=1048576
#pragma HLS INTERFACE m_axi port=p_in offset=slave bundle=gmem2 depth=1048576
#pragma HLS INTERFACE m_axi port=T_in offset=slave bundle=gmem3 depth=1048576
#pragma HLS INTERFACE m_axi port=u_out offset=slave bundle=gmem4 depth=1048576
#pragma HLS INTERFACE m_axi port=v_out offset=slave bundle=gmem5 depth=1048576
#pragma HLS INTERFACE m_axi port=p_out offset=slave bundle=gmem6 depth=1048576
#pragma HLS INTERFACE m_axi port=T_out offset=slave bundle=gmem7 depth=1048576
#pragma HLS INTERFACE s_axilite port=nx
#pragma HLS INTERFACE s_axilite port=ny
#pragma HLS INTERFACE s_axilite port=dt
#pragma HLS INTERFACE s_axilite port=nu
#pragma HLS INTERFACE s_axilite port=alpha
#pragma HLS INTERFACE s_axilite port=g_beta
#pragma HLS INTERFACE s_axilite port=jacobi_iters
#pragma HLS INTERFACE s_axilite port=return

    // This is a SIMPLIFIED kernel showing the PROBLEM
    // Full implementation would need multiple passes
    
    // Step 1: Advection (can pipeline reasonably well)
    ADVECT_I: for (int i = 1; i < ny - 1; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=254 max=1022 avg=510
        ADVECT_J: for (int j = 1; j < nx - 1; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=254 max=1022 avg=510
            
            int idx = i * nx + j;
            
            // Read from DDR (slow!)
            float u_val = u_in[idx];
            float v_val = v_in[idx];
            float T_val = T_in[idx];
            
            // Simple advection (central difference)
            float u_right = u_in[i * nx + (j+1)];
            float u_left = u_in[i * nx + (j-1)];
            float u_up = u_in[(i+1) * nx + j];
            float u_down = u_in[(i-1) * nx + j];
            
            float du_dx = 0.5f * (u_right - u_left);
            float du_dy = 0.5f * (u_up - u_down);
            
            u_out[idx] = u_val - dt * (u_val * du_dx + v_val * du_dy);
        }
    }
    
    // Step 2: Diffusion (can pipeline reasonably well)
    DIFFUSE_I: for (int i = 1; i < ny - 1; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=254 max=1022 avg=510
        DIFFUSE_J: for (int j = 1; j < nx - 1; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=254 max=1022 avg=510
            
            int idx = i * nx + j;
            
            // Laplacian (DDR reads)
            float u_center = u_out[idx];
            float u_n = u_out[(i+1) * nx + j];
            float u_s = u_out[(i-1) * nx + j];
            float u_e = u_out[i * nx + (j+1)];
            float u_w = u_out[i * nx + (j-1)];
            
            float laplacian = u_n + u_s + u_e + u_w - 4.0f * u_center;
            u_out[idx] = u_center + dt * nu * laplacian;
        }
    }
    
    // Step 3: Pressure Poisson solve - THE PROBLEM!
    // This cannot be pipelined well due to Jacobi iteration dependency
    
    // Compute divergence first
    DIV_I: for (int i = 1; i < ny - 1; ++i) {
        DIV_J: for (int j = 1; j < nx - 1; ++j) {
#pragma HLS PIPELINE II=1
            int idx = i * nx + j;
            float du_dx = 0.5f * (u_out[i * nx + (j+1)] - u_out[i * nx + (j-1)]);
            float dv_dy = 0.5f * (v_out[(i+1) * nx + j] - v_out[(i-1) * nx + j]);
            // Store divergence temporarily (would need another buffer!)
        }
    }
    
    // Jacobi iteration - CANNOT PIPELINE ACROSS ITERATIONS!
    JACOBI_ITER: for (int iter = 0; iter < jacobi_iters; ++iter) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100 avg=100
        // Each iteration must complete before next starts
        // NO WAY to achieve II=1 here!
        
        JACOBI_I: for (int i = 1; i < ny - 1; ++i) {
            JACOBI_J: for (int j = 1; j < nx - 1; ++j) {
#pragma HLS PIPELINE II=1
                int idx = i * nx + j;
                
                // Read 4 neighbors from DDR (or BRAM if it fits)
                float p_n = p_in[(i+1) * nx + j];
                float p_s = p_in[(i-1) * nx + j];
                float p_e = p_in[i * nx + (j+1)];
                float p_w = p_in[i * nx + (j-1)];
                
                // Would need div[idx] here
                p_out[idx] = 0.25f * (p_n + p_s + p_e + p_w);
            }
        }
        // Swap buffers (need pointer swap logic)
    }
}

// BRAM version (ONLY works for small grids!)
// Shows resource limits
void pde_kernel_bram(
    const float* u_in,
    const float* v_in,
    const float* T_in,
    float* u_out,
    float* v_out,
    float* T_out,
    int nx,
    int ny,
    float dt,
    float nu,
    float alpha
) {
#pragma HLS INTERFACE m_axi port=u_in offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=v_in offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=T_in offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=u_out offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=v_out offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=T_out offset=slave bundle=gmem5
#pragma HLS INTERFACE s_axilite port=nx
#pragma HLS INTERFACE s_axilite port=ny
#pragma HLS INTERFACE s_axilite port=return

    // On-chip buffers (BRAM) - ONLY for small grids!
    const int MAX_SIZE = 256;
    
    float u_buf[MAX_SIZE * MAX_SIZE];
    float v_buf[MAX_SIZE * MAX_SIZE];
    float T_buf[MAX_SIZE * MAX_SIZE];
#pragma HLS BIND_STORAGE variable=u_buf type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=v_buf type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=T_buf type=RAM_2P impl=BRAM
    
    // Load from DDR to BRAM
    LOAD: for (int i = 0; i < nx * ny; ++i) {
#pragma HLS PIPELINE II=1
        u_buf[i] = u_in[i];
        v_buf[i] = v_in[i];
        T_buf[i] = T_in[i];
    }
    
    // Process in BRAM (faster!)
    PROCESS_I: for (int i = 1; i < ny - 1; ++i) {
        PROCESS_J: for (int j = 1; j < nx - 1; ++j) {
#pragma HLS PIPELINE II=1
            int idx = i * nx + j;
            
            // All accesses now in BRAM (1 cycle latency)
            float u_val = u_buf[idx];
            float laplacian = u_buf[(i+1)*nx+j] + u_buf[(i-1)*nx+j] +
                             u_buf[i*nx+(j+1)] + u_buf[i*nx+(j-1)] - 4.0f*u_val;
            
            u_buf[idx] = u_val + dt * nu * laplacian;
        }
    }
    
    // Store back to DDR
    STORE: for (int i = 0; i < nx * ny; ++i) {
#pragma HLS PIPELINE II=1
        u_out[i] = u_buf[i];
        v_out[i] = v_buf[i];
        T_out[i] = T_buf[i];
    }
}

// Streaming advection (can achieve good II)
void pde_kernel_advect_streaming(
    const float* u_in,
    const float* v_in,
    const float* field_in,
    float* field_out,
    int nx,
    int ny,
    float dt
) {
#pragma HLS INTERFACE m_axi port=u_in offset=slave bundle=gmem0 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=v_in offset=slave bundle=gmem1 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=field_in offset=slave bundle=gmem2 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=field_out offset=slave bundle=gmem3 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=nx
#pragma HLS INTERFACE s_axilite port=ny
#pragma HLS INTERFACE s_axilite port=dt
#pragma HLS INTERFACE s_axilite port=return

    // Line buffers for streaming
    float line_buf_u[2][MAX_WIDTH];
    float line_buf_v[2][MAX_WIDTH];
    float line_buf_f[2][MAX_WIDTH];
#pragma HLS ARRAY_PARTITION variable=line_buf_u complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buf_v complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buf_f complete dim=1
    
    // Streaming advection (can achieve II=1)
    STREAM_I: for (int i = 0; i < ny; ++i) {
        STREAM_J: for (int j = 0; j < nx; ++j) {
#pragma HLS PIPELINE II=1
            
            int idx = i * nx + j;
            
            // Stream in data
            float u_val = u_in[idx];
            float v_val = v_in[idx];
            float f_val = field_in[idx];
            
            // Update line buffers
            line_buf_u[0][j] = line_buf_u[1][j];
            line_buf_u[1][j] = u_val;
            line_buf_v[0][j] = line_buf_v[1][j];
            line_buf_v[1][j] = v_val;
            line_buf_f[0][j] = line_buf_f[1][j];
            line_buf_f[1][j] = f_val;
            
            // Compute advection (if not on boundary)
            float result;
            if (i > 0 && i < ny-1 && j > 0 && j < nx-1) {
                float df_dx = 0.5f * (line_buf_f[1][j+1] - line_buf_f[1][j-1]);
                float df_dy = 0.5f * (line_buf_f[1][j] - line_buf_f[0][j]);
                result = f_val - dt * (u_val * df_dx + v_val * df_dy);
            } else {
                result = f_val;
            }
            
            field_out[idx] = result;
        }
    }
}

// Multi-iteration kernel (for small grids that fit in BRAM)
void pde_kernel_multi_iter(
    const float* u_in,
    const float* v_in,
    float* u_out,
    float* v_out,
    int nx,
    int ny,
    int num_iters,
    float dt,
    float nu
) {
#pragma HLS INTERFACE m_axi port=u_in offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=v_in offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=u_out offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=v_out offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=return

    const int MAX_BUF = 256 * 256;
    
    float buf_u_a[MAX_BUF];
    float buf_u_b[MAX_BUF];
    float buf_v_a[MAX_BUF];
    float buf_v_b[MAX_BUF];
#pragma HLS BIND_STORAGE variable=buf_u_a type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=buf_u_b type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=buf_v_a type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=buf_v_b type=RAM_2P impl=BRAM
    
    // Load initial state
    LOAD_INIT: for (int i = 0; i < nx * ny; ++i) {
#pragma HLS PIPELINE II=1
        buf_u_a[i] = u_in[i];
        buf_v_a[i] = v_in[i];
    }
    
    // Process multiple iterations ON-CHIP
    MULTI_ITER: for (int iter = 0; iter < num_iters; ++iter) {
#pragma HLS LOOP_TRIPCOUNT min=10 max=100 avg=50
        
        float* u_src = (iter % 2 == 0) ? buf_u_a : buf_u_b;
        float* u_dst = (iter % 2 == 0) ? buf_u_b : buf_u_a;
        float* v_src = (iter % 2 == 0) ? buf_v_a : buf_v_b;
        float* v_dst = (iter % 2 == 0) ? buf_v_b : buf_v_a;
        
        ITER_I: for (int i = 1; i < ny - 1; ++i) {
            ITER_J: for (int j = 1; j < nx - 1; ++j) {
#pragma HLS PIPELINE II=1
                int idx = i * nx + j;
                
                float u_val = u_src[idx];
                float u_lap = u_src[(i+1)*nx+j] + u_src[(i-1)*nx+j] +
                             u_src[i*nx+(j+1)] + u_src[i*nx+(j-1)] - 4.0f*u_val;
                
                u_dst[idx] = u_val + dt * nu * u_lap;
            }
        }
    }
    
    // Write final result
    float* u_final = (num_iters % 2 == 0) ? buf_u_a : buf_u_b;
    float* v_final = (num_iters % 2 == 0) ? buf_v_a : buf_v_b;
    
    STORE_FINAL: for (int i = 0; i < nx * ny; ++i) {
#pragma HLS PIPELINE II=1
        u_out[i] = u_final[i];
        v_out[i] = v_final[i];
    }
}
