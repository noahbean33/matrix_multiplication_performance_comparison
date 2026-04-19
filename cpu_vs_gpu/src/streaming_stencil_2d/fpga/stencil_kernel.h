#ifndef STENCIL_KERNEL_H
#define STENCIL_KERNEL_H

// Maximum width for line buffers
// Adjust based on target FPGA BRAM capacity
constexpr int MAX_WIDTH = 2048;

// Simple non-streaming version
void stencil_kernel_simple(
    const float* u_in,
    float* u_out,
    int width,
    int height,
    float alpha,
    float beta
);

// Streaming version with line buffers (THE KEY OPTIMIZATION!)
void stencil_kernel_streaming(
    const float* u_in,
    float* u_out,
    int width,
    int height,
    float alpha,
    float beta
);

// Multi-iteration version (process multiple timesteps on-chip)
void stencil_kernel_multi_iteration(
    const float* u_in,
    float* u_out,
    int width,
    int height,
    int num_iterations,
    float alpha,
    float beta
);

// Dataflow version (pipeline read/compute/write)
void stencil_kernel_dataflow(
    const float* u_in,
    float* u_out,
    int width,
    int height,
    float alpha,
    float beta
);

#endif // STENCIL_KERNEL_H
