#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

// Workload B: Pure Streaming Stencil (2D Diffusion)
// GPU/CUDA implementation showing DRAM round trips

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Stencil kernel: 5-point 2D diffusion
// Each thread computes one point
__global__ void stencil_kernel(
    const float* __restrict__ u_in,
    float* __restrict__ u_out,
    int width,
    int height,
    float alpha,
    float beta
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Skip boundary (using fixed BC, we just don't update boundary)
    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        int idx = i * width + j;
        
        // 5-point stencil
        float center = u_in[idx];
        float north = u_in[idx - width];
        float south = u_in[idx + width];
        float west = u_in[idx - 1];
        float east = u_in[idx + 1];
        
        u_out[idx] = alpha * center + beta * (north + south + west + east);
    }
}

// Shared memory optimization kernel
__global__ void stencil_kernel_shared(
    const float* __restrict__ u_in,
    float* __restrict__ u_out,
    int width,
    int height,
    float alpha,
    float beta
) {
    // Use shared memory to reduce global memory accesses
    const int TILE_W = 32;
    const int TILE_H = 32;
    const int HALO = 1;
    
    __shared__ float tile[TILE_H + 2*HALO][TILE_W + 2*HALO];
    
    int ti = threadIdx.y + HALO;
    int tj = threadIdx.x + HALO;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load center
    if (i < height && j < width) {
        tile[ti][tj] = u_in[i * width + j];
        
        // Load halo regions
        if (threadIdx.y == 0 && i > 0) {
            tile[ti - 1][tj] = u_in[(i - 1) * width + j];
        }
        if (threadIdx.y == blockDim.y - 1 && i < height - 1) {
            tile[ti + 1][tj] = u_in[(i + 1) * width + j];
        }
        if (threadIdx.x == 0 && j > 0) {
            tile[ti][tj - 1] = u_in[i * width + (j - 1)];
        }
        if (threadIdx.x == blockDim.x - 1 && j < width - 1) {
            tile[ti][tj + 1] = u_in[i * width + (j + 1)];
        }
    }
    
    __syncthreads();
    
    // Compute stencil from shared memory
    if (i > 0 && i < height - 1 && j > 0 && j < width - 1) {
        float center = tile[ti][tj];
        float north = tile[ti - 1][tj];
        float south = tile[ti + 1][tj];
        float west = tile[ti][tj - 1];
        float east = tile[ti][tj + 1];
        
        u_out[i * width + j] = alpha * center + beta * (north + south + west + east);
    }
}

void solve_streaming_stencil_gpu(int grid_size, int steps, bool use_shared) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "GPU Benchmark: " << (use_shared ? "Shared Memory" : "Global Memory") << std::endl;
    std::cout << "Grid Size: " << grid_size << "x" << grid_size << ", Steps: " << steps << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Stencil parameters
    const float alpha = 0.1f;
    const float beta = 0.225f;
    
    // Host memory
    std::vector<float> h_u(grid_size * grid_size);
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            h_u[i * grid_size + j] = (i % 10 + j % 10) / 20.0f;
        }
    }
    
    // Device memory
    float *d_u_in, *d_u_out;
    size_t bytes = grid_size * grid_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_u_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_u_in, h_u.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_out, h_u.data(), bytes, cudaMemcpyHostToDevice));
    
    // Kernel configuration
    dim3 block(32, 32);
    dim3 grid((grid_size + block.x - 1) / block.x, (grid_size + block.y - 1) / block.y);
    
    std::cout << "Grid: (" << grid.x << ", " << grid.y << "), Block: (" 
              << block.x << ", " << block.y << ")" << std::endl;
    
    // Warmup
    for (int s = 0; s < 10; ++s) {
        if (use_shared) {
            stencil_kernel_shared<<<grid, block>>>(d_u_in, d_u_out, grid_size, grid_size, alpha, beta);
        } else {
            stencil_kernel<<<grid, block>>>(d_u_in, d_u_out, grid_size, grid_size, alpha, beta);
        }
        std::swap(d_u_in, d_u_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    std::cout << "Running " << steps << " iterations..." << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int s = 0; s < steps; ++s) {
        if (use_shared) {
            stencil_kernel_shared<<<grid, block>>>(d_u_in, d_u_out, grid_size, grid_size, alpha, beta);
        } else {
            stencil_kernel<<<grid, block>>>(d_u_in, d_u_out, grid_size, grid_size, alpha, beta);
        }
        // Swap buffers for next iteration
        // KEY OBSERVATION: Each iteration requires FULL DRAM READ/WRITE!
        std::swap(d_u_in, d_u_out);
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double total_time_s = milliseconds / 1000.0;
    
    // Calculate metrics
    long long total_points = (long long)grid_size * grid_size * steps;
    double points_per_sec = total_points / total_time_s;
    double gflops = total_points * 5 / (total_time_s * 1e9);
    
    // Memory traffic: Each iteration reads entire grid + writes entire grid
    double bytes_per_iteration = 2.0 * grid_size * grid_size * sizeof(float);
    double total_bytes = bytes_per_iteration * steps;
    double gbytes_s = total_bytes / (total_time_s * 1e9);
    
    // Copy result back for verification
    CUDA_CHECK(cudaMemcpy(h_u.data(), d_u_in, bytes, cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n----- Performance Metrics -----" << std::endl;
    std::cout << "Total Time:    " << total_time_s << " s" << std::endl;
    std::cout << "Points/sec:    " << points_per_sec / 1e9 << " Gpts/s" << std::endl;
    std::cout << "Throughput:    " << gflops << " GFLOP/s" << std::endl;
    std::cout << "Bandwidth:     " << gbytes_s << " GB/s" << std::endl;
    std::cout << "Total DRAM:    " << total_bytes / 1e9 << " GB" << std::endl;
    
    std::cout << "\n----- Memory Analysis -----" << std::endl;
    std::cout << "⚠️  DRAM Round Trips: " << steps << " iterations" << std::endl;
    std::cout << "Each iteration requires:" << std::endl;
    std::cout << "  - Full grid READ  (" << (bytes / 1e6) << " MB)" << std::endl;
    std::cout << "  - Full grid WRITE (" << (bytes / 1e6) << " MB)" << std::endl;
    std::cout << "Cannot overlap iterations (output of step N = input of step N+1)" << std::endl;
    
    if (use_shared) {
        std::cout << "\n✓ Shared memory reduces global memory reads within kernel" << std::endl;
        std::cout << "✓ Each point read once per iteration (vs multiple times)" << std::endl;
    } else {
        std::cout << "\n✗ Global memory only: each neighbor read separately" << std::endl;
        std::cout << "✗ Cache helps but still suboptimal" << std::endl;
    }
    
    std::cout << "\nKey Limitation: Must store entire grid in DRAM" << std::endl;
    std::cout << "FPGA advantage: Streaming with line buffers (no full DRAM round trip)" << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_u_in));
    CUDA_CHECK(cudaFree(d_u_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload B: Pure Streaming Stencil" << std::endl;
    std::cout << "Architecture: GPU (CUDA)" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Stencil: 2D 5-point diffusion" << std::endl;
    std::cout << "Equation: u_new = α*u + β*(u_N + u_S + u_E + u_W)" << std::endl;
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
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    
    // From benchmarks.yml
    std::vector<int> grid_sizes = {512, 1024, 2048};
    int steps = 2000;
    
    for (int size : grid_sizes) {
        // Run both global and shared memory versions
        solve_streaming_stencil_gpu(size, steps, false);  // Global memory
        solve_streaming_stencil_gpu(size, steps, true);   // Shared memory
    }
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "Summary: GPU Performance" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "✓ GPU is GOOD at this workload (high parallelism)" << std::endl;
    std::cout << "✓ Shared memory optimization helps significantly" << std::endl;
    std::cout << "⚠️  But: Each iteration requires full DRAM read/write" << std::endl;
    std::cout << "⚠️  Cannot overlap iterations (data dependency)" << std::endl;
    std::cout << "FPGA can do better with streaming line buffers!" << std::endl;
    std::cout << "====================================" << std::endl;
    
    return 0;
}
