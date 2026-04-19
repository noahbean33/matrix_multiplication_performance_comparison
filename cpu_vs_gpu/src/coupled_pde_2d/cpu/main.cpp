#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Workload A: Coupled PDE (2D Navier-Stokes + Temperature)
// CPU implementation with Boussinesq approximation

// 2D incompressible Navier-Stokes with temperature coupling:
// ∂u/∂t + u·∇u = -∇p + ν∇²u
// ∂v/∂t + u·∇v = -∇p + ν∇²v + gβT  (Boussinesq buoyancy)
// ∇·u = 0  (incompressibility via pressure Poisson)
// ∂T/∂t + u·∇T = α∇²T

struct Grid2D {
    int nx, ny;
    std::vector<float> u;      // x-velocity
    std::vector<float> v;      // y-velocity
    std::vector<float> p;      // pressure
    std::vector<float> T;      // temperature
    std::vector<float> div;    // divergence (for pressure solve)
    
    Grid2D(int n) : nx(n), ny(n), 
                    u(n*n, 0.0f), v(n*n, 0.0f), 
                    p(n*n, 0.0f), T(n*n, 0.0f),
                    div(n*n, 0.0f) {}
    
    inline int idx(int i, int j) const { return i * ny + j; }
};

// Parameters
struct Parameters {
    float dt = 0.001f;           // Time step
    float nu = 0.01f;            // Kinematic viscosity
    float alpha = 0.01f;         // Thermal diffusivity
    float g = 9.8f;              // Gravity
    float beta = 0.001f;         // Thermal expansion coefficient
    int jacobi_iters = 100;      // Pressure Poisson iterations
};

// Initialize with some pattern
void initialize(Grid2D& grid) {
    for (int i = 0; i < grid.nx; ++i) {
        for (int j = 0; j < grid.ny; ++j) {
            int idx = grid.idx(i, j);
            // Initial velocity field (small perturbation)
            grid.u[idx] = 0.1f * std::sin(2.0f * M_PI * i / grid.nx);
            grid.v[idx] = 0.1f * std::cos(2.0f * M_PI * j / grid.ny);
            // Initial temperature (hot bottom)
            float y_frac = (float)i / grid.nx;
            grid.T[idx] = 1.0f - y_frac;
        }
    }
}

// LOCAL OPERATION: Advection (stencil-based)
void advect(const Grid2D& grid, const std::vector<float>& field,
            std::vector<float>& field_new, const Parameters& params) {
    float dt = params.dt;
    
    for (int i = 1; i < grid.nx - 1; ++i) {
        for (int j = 1; j < grid.ny - 1; ++j) {
            int idx = grid.idx(i, j);
            
            // Central differences for advection term
            float u_val = grid.u[idx];
            float v_val = grid.v[idx];
            
            float df_dx = 0.5f * (field[grid.idx(i, j+1)] - field[grid.idx(i, j-1)]);
            float df_dy = 0.5f * (field[grid.idx(i+1, j)] - field[grid.idx(i-1, j)]);
            
            field_new[idx] = field[idx] - dt * (u_val * df_dx + v_val * df_dy);
        }
    }
}

// LOCAL OPERATION: Diffusion (stencil-based, Laplacian)
void diffuse(const Grid2D& grid, std::vector<float>& field,
             float diffusion_coeff, const Parameters& params) {
    std::vector<float> field_new = field;
    float dt = params.dt;
    
    for (int i = 1; i < grid.nx - 1; ++i) {
        for (int j = 1; j < grid.ny - 1; ++j) {
            int idx = grid.idx(i, j);
            
            // 5-point Laplacian
            float laplacian = field[grid.idx(i+1, j)] + field[grid.idx(i-1, j)] +
                             field[grid.idx(i, j+1)] + field[grid.idx(i, j-1)] -
                             4.0f * field[idx];
            
            field_new[idx] = field[idx] + dt * diffusion_coeff * laplacian;
        }
    }
    
    field = field_new;
}

// GLOBAL OPERATION: Compute divergence
void compute_divergence(Grid2D& grid) {
    for (int i = 1; i < grid.nx - 1; ++i) {
        for (int j = 1; j < grid.ny - 1; ++j) {
            int idx = grid.idx(i, j);
            
            float du_dx = 0.5f * (grid.u[grid.idx(i, j+1)] - grid.u[grid.idx(i, j-1)]);
            float dv_dy = 0.5f * (grid.v[grid.idx(i+1, j)] - grid.v[grid.idx(i-1, j)]);
            
            grid.div[idx] = du_dx + dv_dy;
        }
    }
}

// GLOBAL OPERATION: Pressure Poisson solve (Jacobi iteration)
// This is the EXPENSIVE GLOBAL operation!
void solve_pressure(Grid2D& grid, const Parameters& params) {
    std::vector<float> p_new = grid.p;
    
    // Jacobi iteration (fixed number)
    for (int iter = 0; iter < params.jacobi_iters; ++iter) {
        for (int i = 1; i < grid.nx - 1; ++i) {
            for (int j = 1; j < grid.ny - 1; ++j) {
                int idx = grid.idx(i, j);
                
                // Laplace(p) = divergence
                p_new[idx] = 0.25f * (grid.p[grid.idx(i+1, j)] + 
                                     grid.p[grid.idx(i-1, j)] +
                                     grid.p[grid.idx(i, j+1)] + 
                                     grid.p[grid.idx(i, j-1)] -
                                     grid.div[idx]);
            }
        }
        grid.p = p_new;
    }
}

// LOCAL OPERATION: Pressure projection (correct velocity)
void project_velocity(Grid2D& grid, const Parameters& params) {
    for (int i = 1; i < grid.nx - 1; ++i) {
        for (int j = 1; j < grid.ny - 1; ++j) {
            int idx = grid.idx(i, j);
            
            float dp_dx = 0.5f * (grid.p[grid.idx(i, j+1)] - grid.p[grid.idx(i, j-1)]);
            float dp_dy = 0.5f * (grid.p[grid.idx(i+1, j)] - grid.p[grid.idx(i-1, j)]);
            
            grid.u[idx] -= dp_dx;
            grid.v[idx] -= dp_dy;
        }
    }
}

// Add Boussinesq buoyancy force to v-velocity
void add_buoyancy(Grid2D& grid, const Parameters& params) {
    float force = params.g * params.beta * params.dt;
    
    for (int i = 0; i < grid.nx * grid.ny; ++i) {
        grid.v[i] += force * grid.T[i];
    }
}

// One timestep
void timestep(Grid2D& grid, const Parameters& params) {
    // 1. Advection (LOCAL)
    std::vector<float> u_star = grid.u;
    std::vector<float> v_star = grid.v;
    std::vector<float> T_star = grid.T;
    
    advect(grid, grid.u, u_star, params);
    advect(grid, grid.v, v_star, params);
    advect(grid, grid.T, T_star, params);
    
    grid.u = u_star;
    grid.v = v_star;
    grid.T = T_star;
    
    // 2. Diffusion (LOCAL)
    diffuse(grid, grid.u, params.nu, params);
    diffuse(grid, grid.v, params.nu, params);
    diffuse(grid, grid.T, params.alpha, params);
    
    // 3. Buoyancy (LOCAL)
    add_buoyancy(grid, params);
    
    // 4. Pressure solve (GLOBAL - expensive!)
    compute_divergence(grid);
    solve_pressure(grid, params);
    project_velocity(grid, params);
}

void solve_coupled_pde(int grid_size, int num_steps) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "Coupled PDE CPU Benchmark" << std::endl;
    std::cout << "Grid: " << grid_size << "x" << grid_size << std::endl;
    std::cout << "Steps: " << num_steps << std::endl;
    std::cout << "====================================" << std::endl;
    
    Parameters params;
    Grid2D grid(grid_size);
    
    // Initialize
    initialize(grid);
    
    // Memory footprint
    size_t memory_bytes = 5 * grid_size * grid_size * sizeof(float);  // 5 fields
    double memory_mb = memory_bytes / (1024.0 * 1024.0);
    
    std::cout << "Fields: u, v, p, T, div (5 arrays)" << std::endl;
    std::cout << "Memory: " << memory_mb << " MB" << std::endl;
    std::cout << "Jacobi iters: " << params.jacobi_iters << std::endl;
    
    // Warmup
    timestep(grid, params);
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_steps; ++step) {
        timestep(grid, params);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    double time_per_step = elapsed.count() / num_steps;
    double total_points = (double)grid_size * grid_size * num_steps;
    double points_per_sec = total_points / elapsed.count();
    
    // Estimate FLOPs (rough)
    // Advection: ~10 ops/point, Diffusion: ~10 ops/point, Pressure: ~10 ops/point * 100 iters
    long long flops_per_step = (long long)grid_size * grid_size * (10 + 10 + 10 * params.jacobi_iters);
    double gflops_per_sec = (flops_per_step * num_steps) / (elapsed.count() * 1e9);
    
    // Memory bandwidth estimate
    // Each step: read all fields multiple times, write all fields
    double bytes_per_step = memory_bytes * 10;  // Rough estimate
    double gbytes_per_sec = (bytes_per_step * num_steps) / (elapsed.count() * 1e9);
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n----- Performance Metrics -----" << std::endl;
    std::cout << "Total time:      " << elapsed.count() << " s" << std::endl;
    std::cout << "Time per step:   " << time_per_step * 1000.0 << " ms" << std::endl;
    std::cout << "Points/sec:      " << points_per_sec / 1e6 << " M pts/s" << std::endl;
    std::cout << "Throughput:      " << gflops_per_sec << " GFLOP/s" << std::endl;
    std::cout << "Bandwidth:       " << gbytes_per_sec << " GB/s" << std::endl;
    
    std::cout << "\n----- Operation Breakdown -----" << std::endl;
    std::cout << "LOCAL operations (advection, diffusion): Fast" << std::endl;
    std::cout << "GLOBAL operation (pressure Poisson):     Expensive!" << std::endl;
    std::cout << "  - " << params.jacobi_iters << " Jacobi iterations" << std::endl;
    std::cout << "  - Full grid traversal each iteration" << std::endl;
    std::cout << "  - Dominates compute time (~90%)" << std::endl;
    
    if (memory_mb < 32) {
        std::cout << "\n✓ Working set fits in L3 cache" << std::endl;
    } else {
        std::cout << "\n✗ Working set exceeds L3 cache (DRAM-bound)" << std::endl;
    }
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload A: Coupled PDE" << std::endl;
    std::cout << "Architecture: CPU" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Equations: 2D Incompressible Navier-Stokes + Temperature" << std::endl;
    std::cout << "Boussinesq approximation (buoyancy coupling)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // From benchmarks.yml
    std::vector<int> grid_sizes = {256, 512, 1024};
    int steps = 1000;
    
    for (int size : grid_sizes) {
        solve_coupled_pde(size, steps);
    }
    
    return 0;
}
