#ifndef PDE_KERNEL_H
#define PDE_KERNEL_H

// Maximum width for line buffers
constexpr int MAX_WIDTH = 2048;

// DDR-based kernel (for large grids)
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
);

// BRAM-based kernel (ONLY for small grids ≤256²)
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
);

// Streaming advection kernel
void pde_kernel_advect_streaming(
    const float* u_in,
    const float* v_in,
    const float* field_in,
    float* field_out,
    int nx,
    int ny,
    float dt
);

// Multi-iteration kernel (process multiple timesteps on-chip)
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
);

#endif // PDE_KERNEL_H
