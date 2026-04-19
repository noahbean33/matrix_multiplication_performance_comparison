#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>

#ifdef XILINX_FPGA
#include "xcl2.hpp"
#endif

#include "pointer_chase_kernel.h"

// Create predictable chain
std::vector<Node> create_predictable_chain(int length, int stride = 7) {
    std::vector<Node> nodes(length);
    for (int i = 0; i < length; ++i) {
        nodes[i].next_index = (i + stride) % length;
        nodes[i].value = static_cast<float>(i % 100) / 100.0f;
    }
    return nodes;
}

// Create random chain
std::vector<Node> create_random_chain(int length) {
    std::vector<Node> nodes(length);
    std::vector<int> indices(length);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < length - 1; ++i) {
        nodes[indices[i]].next_index = indices[i + 1];
        nodes[indices[i]].value = static_cast<float>(i % 100) / 100.0f;
    }
    nodes[indices[length - 1]].next_index = indices[0];
    nodes[indices[length - 1]].value = static_cast<float>((length - 1) % 100) / 100.0f;
    
    return nodes;
}

// Software reference
float pointer_chase_sw(const std::vector<Node>& nodes, int num_hops, int start_index) {
    float sum = 0.0f;
    int current = start_index;
    for (int hop = 0; hop < num_hops; ++hop) {
        sum += nodes[current].value;
        current = nodes[current].next_index;
    }
    return sum;
}

void benchmark_pointer_chase_fpga_sw(int length, bool predictable) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "Pointer Chase FPGA Benchmark (SW Emulation)" << std::endl;
    std::cout << "Pattern: " << (predictable ? "PREDICTABLE (stride)" : "UNPREDICTABLE (random)") << std::endl;
    std::cout << "Chain Length: " << length << " nodes" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Create chain
    std::cout << "Creating chain..." << std::endl;
    std::vector<Node> nodes = predictable ? 
        create_predictable_chain(length, 7) : 
        create_random_chain(length);
    
    std::cout << "Memory footprint: " << (nodes.size() * sizeof(Node)) / (1024.0 * 1024.0) << " MB" << std::endl;
    
    int num_hops = length / 10;
    float result_sw = pointer_chase_sw(nodes, num_hops, 0);
    
    // Call HLS kernel (baseline version)
    std::cout << "\nRunning HLS kernel (baseline)..." << std::endl;
    float result_fpga = 0.0f;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (predictable) {
        pointer_chase_kernel_prefetch(nodes.data(), num_hops, 0, &result_fpga, true, 7);
    } else {
        pointer_chase_kernel_baseline(nodes.data(), num_hops, 0, &result_fpga);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    // Verify
    bool correct = std::abs(result_fpga - result_sw) < 1e-3;
    
    std::cout << "\n----- Verification -----" << std::endl;
    std::cout << "SW result:   " << result_sw << std::endl;
    std::cout << "FPGA result: " << result_fpga << std::endl;
    std::cout << "Status: " << (correct ? "PASS" : "FAIL") << std::endl;
    
    // Performance metrics
    double hops_per_sec = num_hops / elapsed.count();
    double latency_per_hop_ns = (elapsed.count() * 1e9) / num_hops;
    
    std::cout << "\n----- SW Emulation Metrics -----" << std::endl;
    std::cout << "Time: " << elapsed.count() * 1000.0 << " ms" << std::endl;
    std::cout << "Hops/sec: " << hops_per_sec / 1e6 << " M hops/s" << std::endl;
    std::cout << "Latency/hop: " << latency_per_hop_ns << " ns" << std::endl;
    
    std::cout << "\n----- Expected FPGA Hardware Performance -----" << std::endl;
    if (predictable) {
        std::cout << "Pattern: PREDICTABLE (stride)" << std::endl;
        std::cout << "FPGA Advantage: Custom prefetch logic can hide latency!" << std::endl;
        std::cout << "Expected latency: ~30-80 ns/hop" << std::endl;
        std::cout << "Speedup vs baseline: ~2-3x (with prefetch buffer)" << std::endl;
        std::cout << "✓ FPGA can pipeline prefetch operations" << std::endl;
        std::cout << "✓ Multiple outstanding memory requests" << std::endl;
    } else {
        std::cout << "Pattern: UNPREDICTABLE (random)" << std::endl;
        std::cout << "FPGA Challenge: Cannot prefetch effectively!" << std::endl;
        std::cout << "Expected latency: ~80-150 ns/hop (DDR latency)" << std::endl;
        std::cout << "Performance: Similar to CPU (both memory-bound)" << std::endl;
        std::cout << "✗ Serial dependency limits pipelining" << std::endl;
        std::cout << "✗ Each hop waits for DDR read (~100 ns)" << std::endl;
        std::cout << "\nThis is where FPGAs STOP looking magical!" << std::endl;
    }
    
    std::cout << "\n----- FPGA Resource Estimate -----" << std::endl;
    std::cout << "LUTs: ~5-10K (control logic)" << std::endl;
    std::cout << "FFs: ~8-15K (pipeline registers)" << std::endl;
    std::cout << "BRAM: ~10-50 (prefetch buffers)" << std::endl;
    std::cout << "DSP: 0 (no arithmetic)" << std::endl;
    std::cout << "Bottleneck: Memory latency, NOT compute!" << std::endl;
}

#ifdef XILINX_FPGA
void benchmark_pointer_chase_fpga_hw(int length, bool predictable, const std::string& xclbin_file) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "Pointer Chase FPGA Benchmark (Hardware)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    std::cout << "Hardware execution requires:" << std::endl;
    std::cout << "1. Synthesized bitstream (xclbin file)" << std::endl;
    std::cout << "2. Xilinx XRT runtime installed" << std::endl;
    std::cout << "3. Compatible FPGA board (e.g., Alveo U250)" << std::endl;
    std::cout << "\nPlease run synthesis first: make build" << std::endl;
}
#endif

int main(int argc, char** argv) {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload D: Pointer-Chasing / Graph Walk" << std::endl;
    std::cout << "Architecture: FPGA (Xilinx Vitis HLS)" << std::endl;
    std::cout << "====================================" << std::endl;
    
    bool hw_mode = false;
    std::string xclbin_file = "";
    
    if (argc > 1 && std::string(argv[1]) == "--hw") {
        hw_mode = true;
        if (argc > 2) {
            xclbin_file = argv[2];
        }
    }
    
    int length = 1000000;
    
    std::cout << "\nThis benchmark tests FPGA's custom memory logic." << std::endl;
    std::cout << "Question: Can FPGAs build custom prefetch/arbitration?\n" << std::endl;
    
    if (hw_mode) {
#ifdef XILINX_FPGA
        benchmark_pointer_chase_fpga_hw(length, true, xclbin_file);
        benchmark_pointer_chase_fpga_hw(length, false, xclbin_file);
#else
        std::cout << "Hardware mode not available. Rebuild with Xilinx tools." << std::endl;
        return 1;
#endif
    } else {
        std::cout << "Running in SOFTWARE EMULATION mode" << std::endl;
        std::cout << "Use --hw <xclbin> for hardware execution\n" << std::endl;
        
        // Predictable pattern
        benchmark_pointer_chase_fpga_sw(length, true);
        
        // Unpredictable pattern
        benchmark_pointer_chase_fpga_sw(length, false);
    }
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "Summary: FPGA Performance" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Predictable: OK (custom prefetch helps)" << std::endl;
    std::cout << "Random: POOR (limited by DDR latency)" << std::endl;
    std::cout << "Key Insight: FPGA custom logic helps ONLY when" << std::endl;
    std::cout << "             pattern is predictable or batchable." << std::endl;
    std::cout << "\nThis shows when FPGAs STOP looking magical!" << std::endl;
    std::cout << "====================================" << std::endl;
    
    return 0;
}
