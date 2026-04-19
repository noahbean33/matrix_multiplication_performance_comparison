#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

// Workload A: Coupled PDE (2D Navier-Stokes + Temperature)
// GPU/CUDA implementation showing multi-field pressure

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Initialize velocity and temperature fields
__global__ void initialize_kernel(
    float* u, float* v, float* T,
    int nx, int ny
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < ny && j < nx) {
        int idx = i * nx + j;
        u[idx] = 0.1f * sinf(2.0f * M_PI * i / (float)ny);
        v[idx] = 0.1f * cosf(2.0f * M_PI * j / (float)nx);
        float y_frac = (float)i / ny;
        T[idx] = 1.0f - y_frac;
    }
}

// LOCAL OPERATION: Advection kernel
__global__ void advect_kernel(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ field,
    float* __restrict__ field_new,
    int nx, int ny, float dt
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Interior points only
    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1) {
        int idx = i * nx + j;
        
        float u_val = u[idx];
        float v_val = v[idx];
        
        // Central differences
        float df_dx = 0.5f * (field[i * nx + (j+1)] - field[i * nx + (j-1)]);
        float df_dy = 0.5f * (field[(i+1) * nx + j] - field[(i-1) * nx + j]);
        
        field_new[idx] = field[idx] - dt * (u_val * df_dx + v_val * df_dy);
    }
}

// LOCAL OPERATION: Diffusion kernel (Laplacian)
__global__ void diffuse_kernel(
    const float* __restrict__ field,
    float* __restrict__ field_new,
    int nx, int ny, float dt, float coeff
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1) {
        int idx = i * nx + j;
        
        // 5-point Laplacian
        float laplacian = field[(i+1) * nx + j] + field[(i-1) * nx + j] +
                         field[i * nx + (j+1)] + field[i * nx + (j-1)] -
                         4.0f * field[idx];
        
        field_new[idx] = field[idx] + dt * coeff * laplacian;
    }
}

// LOCAL OPERATION: Add buoyancy force
__global__ void add_buoyancy_kernel(
    float* v, const float* T,
    int nx, int ny, float force
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    
    if (idx < total) {
        v[idx] += force * T[idx];
    }
}

// GLOBAL OPERATION: Compute divergence
__global__ void compute_divergence_kernel(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ div,
    int nx, int ny
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1) {
        int idx = i * nx + j;
        
        float du_dx = 0.5f * (u[i * nx + (j+1)] - u[i * nx + (j-1)]);
        float dv_dy = 0.5f * (v[(i+1) * nx + j] - v[(i-1) * nx + j]);
        
        div[idx] = du_dx + dv_dy;
    }
}

// GLOBAL OPERATION: Jacobi iteration for pressure Poisson
// This is the EXPENSIVE kernel that gets called 100 times per timestep!
__global__ void jacobi_iteration_kernel(
    const float* __restrict__ p,
    const float* __restrict__ div,
    float* __restrict__ p_new,
    int nx, int ny
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1) {
        int idx = i * nx + j;
        
        // Laplace(p) = divergence
        p_new[idx] = 0.25f * (p[(i+1) * nx + j] + 
                             p[(i-1) * nx + j] +
                             p[i * nx + (j+1)] + 
                             p[i * nx + (j-1)] -
                             div[idx]);
    }
}

// LOCAL OPERATION: Pressure projection (correct velocity)
__global__ void project_velocity_kernel(
    float* u, float* v,
    const float* __restrict__ p,
    int nx, int ny
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1) {
        int idx = i * nx + j;
        
        float dp_dx = 0.5f * (p[i * nx + (j+1)] - p[i * nx + (j-1)]);
        float dp_dy = 0.5f * (p[(i+1) * nx + j] - p[(i-1) * nx + j]);
        
        u[idx] -= dp_dx;
        v[idx] -= dp_dy;
    }
}

// GPU solver class
class CoupledPDESolverGPU {
public:
    int nx, ny;
    float dt, nu, alpha, g, beta;
    int jacobi_iters;
    
    // Device arrays
    float *d_u, *d_v, *d_p, *d_T, *d_div;
    float *d_u_temp, *d_v_temp, *d_T_temp, *d_p_temp;
    
    dim3 block, grid;
    dim3 block_1d;
    int grid_1d;
    
    CoupledPDESolverGPU(int n) : nx(n), ny(n) {
        dt = 0.001f;
        nu = 0.01f;
        alpha = 0.01f;
        g = 9.8f;
        beta = 0.001f;
        jacobi_iters = 100;
        
        // Allocate device memory
        size_t bytes = nx * ny * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_u, bytes));
        CUDA_CHECK(cudaMalloc(&d_v, bytes));
        CUDA_CHECK(cudaMalloc(&d_p, bytes));
        CUDA_CHECK(cudaMalloc(&d_T, bytes));
        CUDA_CHECK(cudaMalloc(&d_div, bytes));
        CUDA_CHECK(cudaMalloc(&d_u_temp, bytes));
        CUDA_CHECK(cudaMalloc(&d_v_temp, bytes));
        CUDA_CHECK(cudaMalloc(&d_T_temp, bytes));
        CUDA_CHECK(cudaMalloc(&d_p_temp, bytes));
        
        // Zero initialize
        CUDA_CHECK(cudaMemset(d_u, 0, bytes));
        CUDA_CHECK(cudaMemset(d_v, 0, bytes));
        CUDA_CHECK(cudaMemset(d_p, 0, bytes));
        CUDA_CHECK(cudaMemset(d_T, 0, bytes));
        CUDA_CHECK(cudaMemset(d_div, 0, bytes));
        
        // Kernel configuration
        block = dim3(16, 16);
        grid = dim3((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
        
        block_1d = dim3(256);
        grid_1d = (nx * ny + block_1d.x - 1) / block_1d.x;
        
        // Initialize fields
        initialize_kernel<<<grid, block>>>(d_u, d_v, d_T, nx, ny);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    ~CoupledPDESolverGPU() {
        CUDA_CHECK(cudaFree(d_u));
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaFree(d_p));
        CUDA_CHECK(cudaFree(d_T));
        CUDA_CHECK(cudaFree(d_div));
        CUDA_CHECK(cudaFree(d_u_temp));
        CUDA_CHECK(cudaFree(d_v_temp));
        CUDA_CHECK(cudaFree(d_T_temp));
        CUDA_CHECK(cudaFree(d_p_temp));
    }
    
    void timestep() {
        // 1. Advection (LOCAL)
        advect_kernel<<<grid, block>>>(d_u, d_v, d_u, d_u_temp, nx, ny, dt);
        advect_kernel<<<grid, block>>>(d_u, d_v, d_v, d_v_temp, nx, ny, dt);
        advect_kernel<<<grid, block>>>(d_u, d_v, d_T, d_T_temp, nx, ny, dt);
        
        std::swap(d_u, d_u_temp);
        std::swap(d_v, d_v_temp);
        std::swap(d_T, d_T_temp);
        
        // 2. Diffusion (LOCAL)
        diffuse_kernel<<<grid, block>>>(d_u, d_u_temp, nx, ny, dt, nu);
        diffuse_kernel<<<grid, block>>>(d_v, d_v_temp, nx, ny, dt, nu);
        diffuse_kernel<<<grid, block>>>(d_T, d_T_temp, nx, ny, dt, alpha);
        
        std::swap(d_u, d_u_temp);
        std::swap(d_v, d_v_temp);
        std::swap(d_T, d_T_temp);
        
        // 3. Buoyancy (LOCAL)
        float force = g * beta * dt;
        add_buoyancy_kernel<<<grid_1d, block_1d>>>(d_v, d_T, nx, ny, force);
        
        // 4. Pressure solve (GLOBAL - expensive!)
        compute_divergence_kernel<<<grid, block>>>(d_u, d_v, d_div, nx, ny);
        
        // Jacobi iteration - THE BOTTLENECK!
        for (int iter = 0; iter < jacobi_iters; ++iter) {
            jacobi_iteration_kernel<<<grid, block>>>(d_p, d_div, d_p_temp, nx, ny);
            std::swap(d_p, d_p_temp);
        }
        
        // 5. Projection (LOCAL)
        project_velocity_kernel<<<grid, block>>>(d_u, d_v, d_p, nx, ny);
    }
};

void benchmark_coupled_pde_gpu(int grid_size, int num_steps) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "Coupled PDE GPU Benchmark" << std::endl;
    std::cout << "Grid: " << grid_size << "x" << grid_size << std::endl;
    std::cout << "Steps: " << num_steps << std::endl;
    std::cout << "====================================" << std::endl;
    
    CoupledPDESolverGPU solver(grid_size);
    
    // Memory footprint
    size_t memory_bytes = 9 * grid_size * grid_size * sizeof(float);  // 5 main + 4 temp
    double memory_mb = memory_bytes / (1024.0 * 1024.0);
    
    std::cout << "Fields: u, v, p, T, div (5 arrays + 4 temp)" << std::endl;
    std::cout << "GPU Memory: " << memory_mb << " MB" << std::endl;
    std::cout << "Jacobi iters: " << solver.jacobi_iters << std::endl;
    std::cout << "Block size: (" << solver.block.x << ", " << solver.block.y << ")" << std::endl;
    std::cout << "Grid size: (" << solver.grid.x << ", " << solver.grid.y << ")" << std::endl;
    
    // Warmup
    std::cout << "\nWarming up..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        solver.timestep();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    std::cout << "Running " << num_steps << " timesteps..." << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int step = 0; step < num_steps; ++step) {
        solver.timestep();
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double total_time_s = milliseconds / 1000.0;
    double time_per_step = total_time_s / num_steps;
    
    // Calculate metrics
    double total_points = (double)grid_size * grid_size * num_steps;
    double points_per_sec = total_points / total_time_s;
    
    // Estimate FLOPs
    long long flops_per_step = (long long)grid_size * grid_size * (10 + 10 + 10 * solver.jacobi_iters);
    double gflops_per_sec = (flops_per_step * num_steps) / (total_time_s * 1e9);
    
    // Memory bandwidth
    // Each Jacobi iteration: read p (4 neighbors) + div, write p_new
    // Plus all other operations
    double bytes_per_step = memory_bytes * (3 + solver.jacobi_iters * 2);  // Rough estimate
    double gbytes_per_sec = (bytes_per_step * num_steps) / (total_time_s * 1e9);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n----- Performance Metrics -----" << std::endl;
    std::cout << "Total time:      " << total_time_s << " s" << std::endl;
    std::cout << "Time per step:   " << time_per_step * 1000.0 << " ms" << std::endl;
    std::cout << "Points/sec:      " << points_per_sec / 1e6 << " M pts/s" << std::endl;
    std::cout << "Throughput:      " << gflops_per_sec << " GFLOP/s" << std::endl;
    std::cout << "Bandwidth:       " << gbytes_per_sec << " GB/s" << std::endl;
    
    std::cout << "\n----- GPU Analysis -----" << std::endl;
    std::cout << "LOCAL operations (advection, diffusion):" << std::endl;
    std::cout << "  ✓ Good parallelism (each thread independent)" << std::endl;
    std::cout << "  ✓ Coalesced memory access (row-major)" << std::endl;
    std::cout << "  ✓ High GPU utilization" << std::endl;
    
    std::cout << "\nGLOBAL operation (pressure Poisson):" << std::endl;
    std::cout << "  ⚠️  " << solver.jacobi_iters << " kernel launches per timestep!" << std::endl;
    std::cout << "  ⚠️  Cannot overlap iterations (dependency)" << std::endl;
    std::cout << "  ⚠️  Kernel launch overhead adds up" << std::endl;
    std::cout << "  ⚠️  Memory bandwidth intensive" << std::endl;
    
    std::cout << "\n----- Memory Pressure -----" << std::endl;
    std::cout << "Working set: " << memory_mb << " MB in GPU DRAM" << std::endl;
    std::cout << "Jacobi iterations: " << solver.jacobi_iters << " × full grid traversal" << std::endl;
    std::cout << "Each iteration reads ~5 values, writes 1 value per point" << std::endl;
    std::cout << "Total memory traffic per step: ~" << (bytes_per_step / 1e6) << " MB" << std::endl;
    
    std::cout << "\n----- Key Observations -----" << std::endl;
    std::cout << "GPU is GOOD at this workload (high parallelism)" << std::endl;
    std::cout << "But Jacobi iterations show iterative solver overhead:" << std::endl;
    std::cout << "  - " << solver.jacobi_iters << " separate kernel launches" << std::endl;
    std::cout << "  - Cannot pipeline (each depends on previous)" << std::endl;
    std::cout << "  - Memory bandwidth becomes bottleneck" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload A: Coupled PDE" << std::endl;
    std::cout << "Architecture: GPU (CUDA)" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Equations: 2D Incompressible Navier-Stokes + Temperature" << std::endl;
    std::cout << "Boussinesq approximation (buoyancy coupling)" << std::endl;
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
    
    // From benchmarks.yml
    std::vector<int> grid_sizes = {256, 512, 1024};
    int steps = 100;  // Fewer steps for GPU (slower per step due to Jacobi)
    
    for (int size : grid_sizes) {
        benchmark_coupled_pde_gpu(size, steps);
    }
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "Summary: GPU Performance" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "✓ GPU handles multi-field PDEs well" << std::endl;
    std::cout << "✓ Good parallelism for local operations" << std::endl;
    std::cout << "⚠️  Iterative solvers show overhead:" << std::endl;
    std::cout << "   - 100 kernel launches per timestep" << std::endl;
    std::cout << "   - Cannot overlap Jacobi iterations" << std::endl;
    std::cout << "   - Memory bandwidth intensive" << std::endl;
    std::cout << "\nThis shows challenges for FPGAs:" << std::endl;
    std::cout << "   - 5 fields × FP32 × 1024² = 20 MB" << std::endl;
    std::cout << "   - Exceeds typical FPGA BRAM capacity" << std::endl;
    std::cout << "   - Iterative solver hard to pipeline" << std::endl;
    std::cout << "====================================" << std::endl;
    
    return 0;
}
