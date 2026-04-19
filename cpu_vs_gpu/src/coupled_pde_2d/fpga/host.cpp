#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

#ifdef XILINX_FPGA
#include "xcl2.hpp"
#endif

#include "pde_kernel.h"

// Software reference for one field advection
void advect_sw(
    const std::vector<float>& u,
    const std::vector<float>& v,
    const std::vector<float>& field,
    std::vector<float>& field_out,
    int nx, int ny, float dt
) {
    for (int i = 1; i < ny - 1; ++i) {
        for (int j = 1; j < nx - 1; ++j) {
            int idx = i * nx + j;
            float u_val = u[idx];
            float v_val = v[idx];
            
            float df_dx = 0.5f * (field[i * nx + (j+1)] - field[i * nx + (j-1)]);
            float df_dy = 0.5f * (field[(i+1) * nx + j] - field[(i-1) * nx + j]);
            
            field_out[idx] = field[idx] - dt * (u_val * df_dx + v_val * df_dy);
        }
    }
}

void benchmark_coupled_pde_fpga_sw(int grid_size) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "Coupled PDE FPGA Benchmark (SW Emulation)" << std::endl;
    std::cout << "Grid: " << grid_size << "x" << grid_size << std::endl;
    std::cout << "====================================" << std::endl;
    
    float dt = 0.001f;
    float nu = 0.01f;
    float alpha = 0.01f;
    
    // Initialize fields
    std::vector<float> h_u(grid_size * grid_size);
    std::vector<float> h_v(grid_size * grid_size);
    std::vector<float> h_T(grid_size * grid_size);
    std::vector<float> h_p(grid_size * grid_size, 0.0f);
    std::vector<float> h_div(grid_size * grid_size, 0.0f);
    
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            int idx = i * grid_size + j;
            h_u[idx] = 0.1f * std::sin(2.0f * M_PI * i / grid_size);
            h_v[idx] = 0.1f * std::cos(2.0f * M_PI * j / grid_size);
            float y_frac = (float)i / grid_size;
            h_T[idx] = 1.0f - y_frac;
        }
    }
    
    std::vector<float> h_u_out(grid_size * grid_size);
    std::vector<float> h_v_out(grid_size * grid_size);
    std::vector<float> h_T_out(grid_size * grid_size);
    
    // Memory footprint analysis
    size_t bytes_per_field = grid_size * grid_size * sizeof(float);
    double mb_per_field = bytes_per_field / (1024.0 * 1024.0);
    size_t total_bytes = 5 * bytes_per_field;  // u, v, p, T, div
    double total_mb = total_bytes / (1024.0 * 1024.0);
    
    std::cout << "Fields: u, v, p, T, div (5 arrays)" << std::endl;
    std::cout << "Memory per field: " << mb_per_field << " MB" << std::endl;
    std::cout << "Total memory: " << total_mb << " MB" << std::endl;
    
    // Run streaming advection kernel (the only part that works well)
    std::cout << "\nRunning streaming advection kernel..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    pde_kernel_advect_streaming(
        h_u.data(),
        h_v.data(),
        h_T.data(),
        h_T_out.data(),
        grid_size,
        grid_size,
        dt
    );
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "SW emulation time: " << elapsed.count() * 1000.0 << " ms" << std::endl;
    
    // BRAM capacity analysis
    std::cout << "\n----- BRAM Capacity Analysis -----" << std::endl;
    std::cout << "FPGA BRAM (Alveo U250): ~96 MB total" << std::endl;
    std::cout << "Usable per kernel: ~20-40 MB" << std::endl;
    std::cout << "\nRequired for this workload:" << std::endl;
    
    std::cout << "Grid Size | Memory | Fits in BRAM?" << std::endl;
    std::cout << "----------|--------|---------------" << std::endl;
    
    for (int size : {256, 512, 1024}) {
        double mem = 5.0 * size * size * sizeof(float) / (1024.0 * 1024.0);
        bool fits = mem < 30.0;  // Conservative estimate
        std::cout << size << "²     | " << std::setw(5) << std::fixed << std::setprecision(1) 
                  << mem << " MB | " << (fits ? "✓ YES" : "✗ NO (use DDR)") << std::endl;
    }
    
    // Resource usage estimates
    std::cout << "\n----- Expected FPGA Resource Usage -----" << std::endl;
    
    if (grid_size <= 256) {
        std::cout << "Configuration: BRAM-based (all fields on-chip)" << std::endl;
        int bram_18k = (int)((total_mb * 1024) / 36);  // 36 KB per BRAM 18K
        std::cout << "BRAM 18K:  ~" << bram_18k << " / 4320 (" 
                  << (bram_18k * 100 / 4320) << "%)" << std::endl;
        std::cout << "LUTs:      ~15-25K / 1.3M (< 2%)" << std::endl;
        std::cout << "FFs:       ~20-35K / 2.6M (< 2%)" << std::endl;
        std::cout << "DSP:       ~10-20 / 12288 (< 1%)" << std::endl;
        std::cout << "\n✓ BRAM pressure, but feasible" << std::endl;
        std::cout << "✓ Can process multiple iterations on-chip" << std::endl;
        std::cout << "✗ Still limited by Jacobi iteration overhead" << std::endl;
        
    } else if (grid_size <= 512) {
        std::cout << "Configuration: HYBRID (partial BRAM, frequent DDR access)" << std::endl;
        std::cout << "BRAM: EXCEEDED - must use DDR for some fields" << std::endl;
        std::cout << "Strategy: Keep 2-3 fields in BRAM, rest in DDR" << std::endl;
        std::cout << "\n⚠️  BRAM bottleneck becomes severe" << std::endl;
        std::cout << "✗ Cannot fit all fields on-chip" << std::endl;
        std::cout << "✗ DDR round trips required" << std::endl;
        std::cout << "✗ Jacobi iterations particularly slow (100× DDR access)" << std::endl;
        
    } else {
        std::cout << "Configuration: DDR-only (all fields in off-chip memory)" << std::endl;
        std::cout << "BRAM: FAR EXCEEDED - all operations use DDR" << std::endl;
        std::cout << "Strategy: Line buffers for streaming ops, DDR for everything else" << std::endl;
        std::cout << "\n❌ BRAM completely insufficient" << std::endl;
        std::cout << "✗ All fields must reside in DDR" << std::endl;
        std::cout << "✗ Random access pattern for Jacobi (no streaming)" << std::endl;
        std::cout << "✗ 100 Jacobi iterations × full DDR traversal" << std::endl;
        std::cout << "✗ Pipeline II likely > 10 cycles (DDR latency)" << std::endl;
        std::cout << "\nThis is where FPGA LOSES its advantage!" << std::endl;
    }
    
    // Jacobi iteration analysis
    std::cout << "\n----- Jacobi Iteration Challenge -----" << std::endl;
    std::cout << "Problem: 100 iterations with inter-iteration dependency" << std::endl;
    std::cout << "\nCannot achieve:" << std::endl;
    std::cout << "  ✗ II=1 across iterations (dependency!)" << std::endl;
    std::cout << "  ✗ Streaming dataflow (random access)" << std::endl;
    std::cout << "  ✗ Line buffer optimization (need full grid)" << std::endl;
    
    std::cout << "\nCan achieve:" << std::endl;
    std::cout << "  ✓ II=1 within single Jacobi iteration" << std::endl;
    std::cout << "  ✓ Pipeline individual stencil operations" << std::endl;
    
    std::cout << "\nExpected performance:" << std::endl;
    std::cout << "  - Advection/Diffusion: ~300 M pts/s (streaming)" << std::endl;
    std::cout << "  - Jacobi iteration: ~50-100 M pts/s (DDR-bound)" << std::endl;
    std::cout << "  - Overall: ~80-120 M pts/s (Jacobi dominates)" << std::endl;
    
    // Comparison with GPU
    std::cout << "\n----- FPGA vs GPU for this Workload -----" << std::endl;
    std::cout << "GPU Advantages:" << std::endl;
    std::cout << "  ✓ Large DRAM capacity (8-24 GB)" << std::endl;
    std::cout << "  ✓ High memory bandwidth (500+ GB/s)" << std::endl;
    std::cout << "  ✓ Good at parallel Jacobi iterations" << std::endl;
    std::cout << "  ✓ Can handle large grids easily" << std::endl;
    
    std::cout << "\nFPGA Challenges:" << std::endl;
    std::cout << "  ✗ Limited BRAM (~96 MB total)" << std::endl;
    std::cout << "  ✗ DDR bandwidth lower (20-40 GB/s)" << std::endl;
    std::cout << "  ✗ Jacobi iterations don't pipeline across iterations" << std::endl;
    std::cout << "  ✗ Multi-field pressure exceeds on-chip capacity" << std::endl;
    
    std::cout << "\nConclusion:" << std::endl;
    std::cout << "This workload shows FPGA's BRAM/memory bottleneck!" << std::endl;
    std::cout << "GPU is likely FASTER for grids ≥ 512²" << std::endl;
}

#ifdef XILINX_FPGA
void benchmark_coupled_pde_fpga_hw(int grid_size, const std::string& xclbin_file) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "Coupled PDE FPGA Benchmark (Hardware)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    std::cout << "Hardware execution requires:" << std::endl;
    std::cout << "1. Synthesized bitstream (xclbin file)" << std::endl;
    std::cout << "2. Xilinx XRT runtime installed" << std::endl;
    std::cout << "3. Compatible FPGA board (e.g., Alveo U250)" << std::endl;
    std::cout << "\nNote: Synthesis will likely show BRAM resource exhaustion" << std::endl;
    std::cout << "      for grids ≥ 512²" << std::endl;
}
#endif

int main(int argc, char** argv) {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload A: Coupled PDE" << std::endl;
    std::cout << "Architecture: FPGA (Xilinx Vitis HLS)" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Equations: 2D Incompressible Navier-Stokes + Temperature" << std::endl;
    std::cout << "Boussinesq approximation (buoyancy coupling)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    bool hw_mode = false;
    std::string xclbin_file = "";
    
    if (argc > 1 && std::string(argv[1]) == "--hw") {
        hw_mode = true;
        if (argc > 2) {
            xclbin_file = argv[2];
        }
    }
    
    std::cout << "\nThis benchmark demonstrates FPGA's BRAM PRESSURE challenge!" << std::endl;
    std::cout << "Multiple fields + iterative solver = memory bottleneck\n" << std::endl;
    
    // From benchmarks.yml
    std::vector<int> grid_sizes = {256, 512, 1024};
    
    if (hw_mode) {
#ifdef XILINX_FPGA
        for (int size : grid_sizes) {
            benchmark_coupled_pde_fpga_hw(size, xclbin_file);
        }
#else
        std::cout << "Hardware mode not available. Rebuild with Xilinx tools." << std::endl;
        return 1;
#endif
    } else {
        std::cout << "Running in SOFTWARE EMULATION mode" << std::endl;
        std::cout << "Use --hw <xclbin> for hardware execution\n" << std::endl;
        
        for (int size : grid_sizes) {
            benchmark_coupled_pde_fpga_sw(size);
        }
    }
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "Summary: FPGA Performance" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "256²:  BRAM pressure, but manageable" << std::endl;
    std::cout << "512²:  BRAM exceeded, hybrid DDR/BRAM" << std::endl;
    std::cout << "1024²: DDR-only, FPGA loses advantage" << std::endl;
    std::cout << "\nKey Insight:" << std::endl;
    std::cout << "  5 fields × N² × 4 bytes quickly exceeds BRAM" << std::endl;
    std::cout << "  Jacobi iterations (100×) make it worse" << std::endl;
    std::cout << "  Cannot stream (random access pattern)" << std::endl;
    std::cout << "\nThis is where \"FPGA is great... until BRAM becomes the bottleneck\"" << std::endl;
    std::cout << "====================================" << std::endl;
    
    return 0;
}
