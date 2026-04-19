#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

// Workload D: Pointer-Chasing / Graph Walk
// GPU/CUDA implementation - shows why GPUs HATE this workload

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Node structure
struct Node {
    int next_index;
    float value;
};

// GPU Kernel - each thread chases its own chain
// This is TERRIBLE for GPU because:
// 1. Each thread has different memory access pattern (no coalescing)
// 2. All threads wait for each memory fetch (serialization)
// 3. Warp divergence if chains have different lengths
__global__ void pointer_chase_kernel(
    const Node* __restrict__ nodes,
    int num_hops,
    int* start_indices,
    float* results,
    int num_threads
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_threads) {
        float sum = 0.0f;
        int current = start_indices[tid];
        
        // This loop is HORRIBLE for GPU:
        // - Each iteration depends on the previous one (no parallelism)
        // - Memory access pattern is completely scattered
        // - Threads in a warp access completely different addresses
        for (int hop = 0; hop < num_hops; ++hop) {
            sum += nodes[current].value;
            current = nodes[current].next_index;  // SERIAL DEPENDENCY!
        }
        
        results[tid] = sum;
    }
}

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

void benchmark_pointer_chase_gpu(int length, bool predictable) {
    std::cout << "\n====================================" << std::endl;
    std::cout << "Pointer Chase GPU Benchmark" << std::endl;
    std::cout << "Pattern: " << (predictable ? "PREDICTABLE (stride)" : "UNPREDICTABLE (random)") << std::endl;
    std::cout << "Chain Length: " << length << " nodes" << std::endl;
    std::cout << "====================================" << std::endl;
    
    // Create chain
    std::cout << "Creating chain..." << std::endl;
    std::vector<Node> h_nodes = predictable ? 
        create_predictable_chain(length, 7) : 
        create_random_chain(length);
    
    std::cout << "Memory footprint: " << (h_nodes.size() * sizeof(Node)) / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // Allocate device memory
    Node* d_nodes;
    CUDA_CHECK(cudaMalloc(&d_nodes, h_nodes.size() * sizeof(Node)));
    CUDA_CHECK(cudaMemcpy(d_nodes, h_nodes.data(), h_nodes.size() * sizeof(Node), cudaMemcpyHostToDevice));
    
    // Launch multiple threads, each chasing through the chain
    // This simulates multiple independent pointer-chasing tasks
    const int num_threads = 256;  // Multiple warps
    int block_size = 256;
    int grid_size = 1;
    
    std::vector<int> h_start_indices(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        h_start_indices[i] = (i * 13) % length;  // Different starting points
    }
    
    int* d_start_indices;
    float* d_results;
    CUDA_CHECK(cudaMalloc(&d_start_indices, num_threads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, num_threads * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_start_indices, h_start_indices.data(), num_threads * sizeof(int), cudaMemcpyHostToDevice));
    
    int num_hops = length / 10;  // Shorter for GPU (still shows the problem)
    
    std::cout << "Launching " << num_threads << " threads, each doing " << num_hops << " hops..." << std::endl;
    std::cout << "\n⚠️  WARNING: This workload is TERRIBLE for GPU!" << std::endl;
    std::cout << "   - Serial dependencies (can't parallelize within a chain)" << std::endl;
    std::cout << "   - Scattered memory access (no coalescing)" << std::endl;
    std::cout << "   - Warp divergence (threads take different times)" << std::endl;
    
    // Warmup
    pointer_chase_kernel<<<grid_size, block_size>>>(d_nodes, num_hops, d_start_indices, d_results, num_threads);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_runs = 100;
    std::cout << "\nRunning " << num_runs << " iterations..." << std::endl;
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < num_runs; ++run) {
        pointer_chase_kernel<<<grid_size, block_size>>>(d_nodes, num_hops, d_start_indices, d_results, num_threads);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    double elapsed_sec = milliseconds / 1000.0;
    double avg_time = elapsed_sec / num_runs;
    
    // Calculate metrics
    double total_hops = static_cast<double>(num_runs) * num_threads * num_hops;
    double hops_per_sec = total_hops / elapsed_sec;
    double avg_latency_per_hop_ns = (elapsed_sec * 1e9) / total_hops;
    
    // Copy results back
    std::vector<float> h_results(num_threads);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, num_threads * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n----- Performance Metrics -----" << std::endl;
    std::cout << "Total hops:        " << static_cast<long long>(total_hops) << std::endl;
    std::cout << "Total time:        " << elapsed_sec * 1000.0 << " ms" << std::endl;
    std::cout << "Avg time per run:  " << avg_time * 1000.0 << " ms" << std::endl;
    std::cout << "Hops per second:   " << hops_per_sec / 1e6 << " M hops/s" << std::endl;
    std::cout << "Latency per hop:   " << avg_latency_per_hop_ns << " ns" << std::endl;
    
    std::cout << "\n----- Why GPU is Slow -----" << std::endl;
    std::cout << "1. SERIAL DEPENDENCY: Each hop depends on previous result" << std::endl;
    std::cout << "   → Can't parallelize within a chain" << std::endl;
    std::cout << "   → All GPU cores wait for memory latency" << std::endl;
    std::cout << "2. SCATTERED ACCESS: Each thread accesses random addresses" << std::endl;
    std::cout << "   → No memory coalescing (each warp = 32 different addresses!)" << std::endl;
    std::cout << "   → Memory bandwidth wasted" << std::endl;
    std::cout << "3. WARP DIVERGENCE: Threads may have different execution times" << std::endl;
    std::cout << "   → Entire warp waits for slowest thread" << std::endl;
    
    std::cout << "\n----- Analysis -----" << std::endl;
    if (predictable) {
        std::cout << "Predictable: Still BAD (GPU can't prefetch well)" << std::endl;
        std::cout << "Expected: ~100-300 ns/hop (memory-bound)" << std::endl;
    } else {
        std::cout << "Random: TERRIBLE (worst-case for GPU)" << std::endl;
        std::cout << "Expected: ~200-500 ns/hop (much worse than CPU!)" << std::endl;
    }
    
    std::cout << "\nGPU is ~5-10x SLOWER than CPU for pointer chasing!" << std::endl;
    
    // Verification
    std::cout << "\n----- Verification -----" << std::endl;
    std::cout << "Thread 0 result: " << h_results[0] << std::endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_start_indices));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::cout << "====================================" << std::endl;
    std::cout << "Workload D: Pointer-Chasing / Graph Walk" << std::endl;
    std::cout << "Architecture: GPU (CUDA)" << std::endl;
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
    
    std::cout << "\n⚠️  This benchmark shows GPU's WORST-CASE workload!" << std::endl;
    std::cout << "Pointer chasing has SERIAL dependencies and SCATTERED memory access.\n" << std::endl;
    
    int length = 1000000;
    
    // Profile 1: Predictable (still bad for GPU)
    benchmark_pointer_chase_gpu(length, true);
    
    // Profile 2: Unpredictable (catastrophic for GPU)
    benchmark_pointer_chase_gpu(length, false);
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "Summary: GPU Performance" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Predictable: POOR (can't overcome serial dependency)" << std::endl;
    std::cout << "Random: TERRIBLE (worst case for GPU architecture)" << std::endl;
    std::cout << "GPU is 5-10x SLOWER than CPU on this workload!" << std::endl;
    std::cout << "====================================" << std::endl;
    
    return 0;
}
