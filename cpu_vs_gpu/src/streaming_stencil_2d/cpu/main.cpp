#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// Workload B: Pure Streaming Stencil (2D Diffusion)
// CPU implementation using std::vector

void solve_streaming_stencil(int grid_size, int steps) {
    std::cout << "---\nRunning Streaming Stencil on CPU...\n";
    std::cout << "Grid Size: " << grid_size << "x" << grid_size << ", Steps: " << steps << std::endl;

    // 1. Initialize Grids
    std::vector<std::vector<float>> u(grid_size, std::vector<float>(grid_size, 1.0f));
    std::vector<std::vector<float>> u_new(grid_size, std::vector<float>(grid_size, 1.0f));

    // Initialize with a pattern
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            u[i][j] = (i % 10 + j % 10) / 20.0f;
        }
    }

    // 2. Stencil Parameters
    const float alpha = 0.1f;
    const float beta = 0.225f; // 0.9 / 4.0

    // 3. Main Computation Loop
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < steps; ++s) {
        // Apply the 5-point stencil
        for (int i = 1; i < grid_size - 1; ++i) {
            for (int j = 1; j < grid_size - 1; ++j) {
                u_new[i][j] = alpha * u[i][j] + beta * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1]);
            }
        }
        // Swap grids (or copy back)
        u.swap(u_new);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;

    // 4. Report Metrics
    double total_time_s = duration_ms.count() / 1000.0;
    long long total_points = (long long)grid_size * grid_size * steps;
    double points_per_sec = total_points / total_time_s;
    
    // FLOPs: 5 ops per point (1 mul + 4 muls + 4 adds = 9 ops, simplified to 5)
    double gflops = total_points * 5 / (total_time_s * 1e9);
    
    // Bandwidth: read 5 values, write 1 value per point
    double bytes_per_point = 6 * sizeof(float);  // 5 reads + 1 write
    double gbytes_s = (total_points * bytes_per_point) / (total_time_s * 1e9);
    
    // Memory footprint
    double memory_mb = 2.0 * grid_size * grid_size * sizeof(float) / (1024.0 * 1024.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n----- Performance Metrics -----" << std::endl;
    std::cout << "Total Time:    " << total_time_s << " s" << std::endl;
    std::cout << "Points/sec:    " << points_per_sec / 1e9 << " Gpts/s" << std::endl;
    std::cout << "Throughput:    " << gflops << " GFLOP/s" << std::endl;
    std::cout << "Bandwidth:     " << gbytes_s << " GB/s" << std::endl;
    std::cout << "Memory:        " << memory_mb << " MB (2 grids)" << std::endl;
    
    std::cout << "\n----- Analysis -----" << std::endl;
    std::cout << "CPU limited by cache hierarchy and memory bandwidth" << std::endl;
    std::cout << "Working set: " << memory_mb << " MB" << std::endl;
    if (memory_mb < 32) {
        std::cout << "✓ Fits in L3 cache (good performance expected)" << std::endl;
    } else {
        std::cout << "✗ Exceeds L3 cache (DRAM bound)" << std::endl;
    }
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload B: Pure Streaming Stencil" << std::endl;
    std::cout << "Architecture: CPU" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Stencil: 2D 5-point diffusion" << std::endl;
    std::cout << "Equation: u_new = α*u + β*(u_N + u_S + u_E + u_W)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // From benchmarks.yml
    std::vector<int> grid_sizes = {512, 1024, 2048};
    int steps = 2000;

    for (int size : grid_sizes) {
        solve_streaming_stencil(size, steps);
    }

    return 0;
}
