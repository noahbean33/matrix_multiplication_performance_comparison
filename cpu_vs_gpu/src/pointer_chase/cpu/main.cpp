#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>

// Workload D: Pointer-Chasing / Graph Walk
// CPU implementation with predictable and unpredictable patterns

// Node structure for linked list / graph
struct Node {
    int next_index;  // Index of next node to visit
    float value;     // Value to accumulate
};

// Create predictable access pattern: next = f(i)
// Uses a stride pattern that can be prefetched
std::vector<Node> create_predictable_chain(int length, int stride = 7) {
    std::vector<Node> nodes(length);
    
    // Create a strided pattern: next = (current + stride) % length
    // This is predictable and hardware prefetchers can learn it
    for (int i = 0; i < length; ++i) {
        nodes[i].next_index = (i + stride) % length;
        nodes[i].value = static_cast<float>(i % 100) / 100.0f;
    }
    
    return nodes;
}

// Create unpredictable (random) access pattern
// This defeats hardware prefetchers and caches
std::vector<Node> create_random_chain(int length) {
    std::vector<Node> nodes(length);
    
    // Create a random permutation - each node visited exactly once
    std::vector<int> indices(length);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    // Build chain: nodes[i].next = indices[i]
    for (int i = 0; i < length - 1; ++i) {
        nodes[indices[i]].next_index = indices[i + 1];
        nodes[indices[i]].value = static_cast<float>(i % 100) / 100.0f;
    }
    nodes[indices[length - 1]].next_index = indices[0]; // Close the loop
    nodes[indices[length - 1]].value = static_cast<float>((length - 1) % 100) / 100.0f;
    
    return nodes;
}

// Pointer-chasing kernel
float pointer_chase(const std::vector<Node>& nodes, int num_hops, int start_index = 0) {
    float sum = 0.0f;
    int current = start_index;
    
    // Chase pointers for num_hops
    for (int hop = 0; hop < num_hops; ++hop) {
        sum += nodes[current].value;
        current = nodes[current].next_index;  // Memory indirection!
    }
    
    return sum;
}

void benchmark_pointer_chase(int length, bool predictable) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "Pointer Chase CPU Benchmark" << std::endl;
    std::cout << "Pattern: " << (predictable ? "PREDICTABLE (stride)" : "UNPREDICTABLE (random)") << std::endl;
    std::cout << "Chain Length: " << length << " nodes" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Create chain
    std::cout << "Creating chain..." << std::endl;
    auto start_create = std::chrono::high_resolution_clock::now();
    std::vector<Node> nodes = predictable ? 
        create_predictable_chain(length, 7) : 
        create_random_chain(length);
    auto end_create = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> create_time = end_create - start_create;
    
    std::cout << "Chain created in " << create_time.count() << " s" << std::endl;
    std::cout << "Memory footprint: " << (nodes.size() * sizeof(Node)) / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Warm up caches
    int num_hops = length;  // Traverse entire chain
    float result = pointer_chase(nodes, num_hops);
    
    // Benchmark
    const int num_runs = 100;
    std::cout << "\nRunning " << num_runs << " iterations (" << num_hops << " hops each)..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < num_runs; ++run) {
        result += pointer_chase(nodes, num_hops, run % 10);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / num_runs;
    
    // Calculate metrics
    double total_hops = static_cast<double>(num_runs) * num_hops;
    double hops_per_sec = total_hops / elapsed.count();
    double avg_latency_per_hop_ns = (elapsed.count() * 1e9) / total_hops;
    
    // Memory transactions estimate
    // Each hop reads: next_index (4 bytes) + value (4 bytes) = 8 bytes
    // Plus cache line overhead (64 bytes typical)
    double bytes_per_hop = sizeof(Node);  // 8 bytes
    double effective_bytes_per_hop = predictable ? bytes_per_hop : 64.0;  // Random hits cache line
    double total_bytes = total_hops * effective_bytes_per_hop;
    double gbytes_transferred = total_bytes / 1e9;
    double bandwidth_gbs = gbytes_transferred / elapsed.count();
    
    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n----- Performance Metrics -----" << std::endl;
    std::cout << "Total hops:        " << static_cast<long long>(total_hops) << std::endl;
    std::cout << "Total time:        " << elapsed.count() * 1000.0 << " ms" << std::endl;
    std::cout << "Avg time per run:  " << avg_time * 1000.0 << " ms" << std::endl;
    std::cout << "Hops per second:   " << hops_per_sec / 1e6 << " M hops/s" << std::endl;
    std::cout << "Latency per hop:   " << avg_latency_per_hop_ns << " ns" << std::endl;
    
    std::cout << "\n----- Memory Metrics -----" << std::endl;
    std::cout << "Bytes per hop:     " << bytes_per_hop << " bytes (actual data)" << std::endl;
    std::cout << "Effective bytes:   " << effective_bytes_per_hop << " bytes (with cache lines)" << std::endl;
    std::cout << "Total transferred: " << gbytes_transferred << " GB" << std::endl;
    std::cout << "Bandwidth:         " << bandwidth_gbs << " GB/s" << std::endl;
    
    std::cout << "\n----- Analysis -----" << std::endl;
    if (predictable) {
        std::cout << "✓ Predictable pattern allows hardware prefetching" << std::endl;
        std::cout << "✓ Cache can anticipate next access" << std::endl;
        std::cout << "✓ Branch predictor learns the pattern" << std::endl;
        std::cout << "Expected: ~10-30 ns/hop (good cache + prefetch)" << std::endl;
    } else {
        std::cout << "✗ Random pattern defeats prefetchers" << std::endl;
        std::cout << "✗ Each hop likely causes cache miss" << std::endl;
        std::cout << "✗ Memory latency dominates (DDR ~50-100 ns)" << std::endl;
        std::cout << "Expected: ~50-150 ns/hop (memory-bound)" << std::endl;
    }
    
    // Verification
    std::cout << "\n----- Verification -----" << std::endl;
    std::cout << "Accumulated value: " << result << std::endl;
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload D: Pointer-Chasing / Graph Walk" << std::endl;
    std::cout << "Architecture: CPU" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // From benchmarks.yml
    int length = 1000000;
    
    std::cout << "\nThis benchmark tests memory indirection performance." << std::endl;
    std::cout << "It shows when hardware prefetchers and caches help (or don't).\n" << std::endl;
    
    // Profile 1: Predictable (stride pattern)
    benchmark_pointer_chase(length, true);
    
    // Profile 2: Unpredictable (random pattern)
    benchmark_pointer_chase(length, false);
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "Summary: CPU Performance" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Predictable: Good (prefetch helps)" << std::endl;
    std::cout << "Random: OK (decent caches, but memory-bound)" << std::endl;
    std::cout << "CPU has best general-purpose memory subsystem" << std::endl;
    std::cout << "====================================" << std::endl;
    
    return 0;
}
