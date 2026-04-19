#include "pointer_chase_kernel.h"

// FPGA Kernel for Pointer Chasing / Graph Walk
// Shows custom prefetch logic for predictable patterns
// But still struggles with truly random access

// Node structure
struct Node {
    int next_index;
    float value;
};

// Baseline pointer-chasing kernel
// This is SERIAL - can't pipeline well even on FPGA
void pointer_chase_kernel_baseline(
    const Node* nodes,
    int num_hops,
    int start_index,
    float* result
) {
#pragma HLS INTERFACE m_axi port=nodes offset=slave bundle=gmem0 depth=1000000
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem1 depth=1
#pragma HLS INTERFACE s_axilite port=num_hops
#pragma HLS INTERFACE s_axilite port=start_index
#pragma HLS INTERFACE s_axilite port=return

    float sum = 0.0f;
    int current = start_index;
    
    // This loop CANNOT be pipelined due to loop-carried dependency
    // Each iteration depends on the result of the previous one
    CHASE_LOOP:
    for (int hop = 0; hop < num_hops; ++hop) {
// Cannot use PIPELINE pragma here - loop-carried dependency!
// #pragma HLS PIPELINE II=1  // This would FAIL synthesis!
#pragma HLS LOOP_TRIPCOUNT min=1000 max=100000 avg=10000
        
        // Read current node (memory latency!)
        Node curr_node = nodes[current];
        sum += curr_node.value;
        current = curr_node.next_index;  // DEPENDENCY!
    }
    
    *result = sum;
}

// Optimized kernel with speculative prefetch
// Only works well if pattern is predictable!
void pointer_chase_kernel_prefetch(
    const Node* nodes,
    int num_hops,
    int start_index,
    float* result,
    bool predictable,
    int stride  // Only used if predictable
) {
#pragma HLS INTERFACE m_axi port=nodes offset=slave bundle=gmem0 depth=1000000 latency=100
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem1 depth=1
#pragma HLS INTERFACE s_axilite port=num_hops
#pragma HLS INTERFACE s_axilite port=start_index
#pragma HLS INTERFACE s_axilite port=predictable
#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=return

    float sum = 0.0f;
    int current = start_index;
    
    if (predictable) {
        // FPGA Advantage: Custom prefetch logic for predictable patterns!
        // We can speculatively fetch ahead using the known pattern
        
        const int PREFETCH_DEPTH = 4;
        Node prefetch_buffer[PREFETCH_DEPTH];
#pragma HLS ARRAY_PARTITION variable=prefetch_buffer complete
        
        // Pre-fill buffer
        int prefetch_indices[PREFETCH_DEPTH];
#pragma HLS ARRAY_PARTITION variable=prefetch_indices complete
        
        PREFILL:
        for (int i = 0; i < PREFETCH_DEPTH; ++i) {
#pragma HLS UNROLL
            prefetch_indices[i] = (start_index + i * stride) % num_hops;
            prefetch_buffer[i] = nodes[prefetch_indices[i]];
        }
        
        int buffer_idx = 0;
        
        // Chase with prefetching
        CHASE_PREFETCH:
        for (int hop = 0; hop < num_hops; ++hop) {
#pragma HLS LOOP_TRIPCOUNT min=1000 max=100000 avg=10000
            
            // Use buffered value
            sum += prefetch_buffer[buffer_idx].value;
            current = prefetch_buffer[buffer_idx].next_index;
            
            // Prefetch next value (predictable offset)
            int next_prefetch = (current + PREFETCH_DEPTH * stride) % num_hops;
            prefetch_buffer[buffer_idx] = nodes[next_prefetch];
            
            buffer_idx = (buffer_idx + 1) % PREFETCH_DEPTH;
        }
        
    } else {
        // Random pattern - no prefetch helps
        // Falls back to serial chasing
        CHASE_RANDOM:
        for (int hop = 0; hop < num_hops; ++hop) {
#pragma HLS LOOP_TRIPCOUNT min=1000 max=100000 avg=10000
            
            Node curr_node = nodes[current];
            sum += curr_node.value;
            current = curr_node.next_index;
        }
    }
    
    *result = sum;
}

// Multi-chain kernel - process multiple independent chains
// This can achieve some parallelism if chains are independent
void pointer_chase_kernel_multichain(
    const Node* nodes,
    int chain_length,
    int num_chains,
    const int* start_indices,
    float* results
) {
#pragma HLS INTERFACE m_axi port=nodes offset=slave bundle=gmem0 depth=1000000
#pragma HLS INTERFACE m_axi port=start_indices offset=slave bundle=gmem1 depth=256
#pragma HLS INTERFACE m_axi port=results offset=slave bundle=gmem2 depth=256
#pragma HLS INTERFACE s_axilite port=chain_length
#pragma HLS INTERFACE s_axilite port=num_chains
#pragma HLS INTERFACE s_axilite port=return

    // Process chains sequentially (could also use DATAFLOW for parallelism)
    CHAIN_LOOP:
    for (int c = 0; c < num_chains; ++c) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=256 avg=32
        
        float sum = 0.0f;
        int current = start_indices[c];
        
        HOP_LOOP:
        for (int hop = 0; hop < chain_length; ++hop) {
#pragma HLS LOOP_TRIPCOUNT min=1000 max=100000 avg=10000
            
            Node curr_node = nodes[current];
            sum += curr_node.value;
            current = curr_node.next_index;
        }
        
        results[c] = sum;
    }
}

// Burst-read optimized kernel (for adjacent reads)
// Only effective if nodes are stored in access order
void pointer_chase_kernel_burst(
    const Node* nodes,
    int num_hops,
    int start_index,
    float* result
) {
#pragma HLS INTERFACE m_axi port=nodes offset=slave bundle=gmem0 depth=1000000 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem1 depth=1
#pragma HLS INTERFACE s_axilite port=num_hops
#pragma HLS INTERFACE s_axilite port=start_index
#pragma HLS INTERFACE s_axilite port=return

    // Buffer for burst reads
    const int BURST_SIZE = 256;
    Node node_buffer[BURST_SIZE];
#pragma HLS BIND_STORAGE variable=node_buffer type=RAM_2P impl=BRAM
    
    float sum = 0.0f;
    int current = start_index;
    int buffer_base = -1;
    
    CHASE_BURST:
    for (int hop = 0; hop < num_hops; ++hop) {
#pragma HLS LOOP_TRIPCOUNT min=1000 max=100000 avg=10000
        
        // Check if current is in buffer
        int buffer_offset = current - buffer_base;
        
        if (buffer_offset < 0 || buffer_offset >= BURST_SIZE) {
            // Refill buffer with burst read
            buffer_base = current;
            
            BURST_READ:
            for (int i = 0; i < BURST_SIZE && (buffer_base + i) < num_hops; ++i) {
#pragma HLS PIPELINE II=1
                node_buffer[i] = nodes[buffer_base + i];
            }
            
            buffer_offset = 0;
        }
        
        // Use buffered node
        sum += node_buffer[buffer_offset].value;
        current = node_buffer[buffer_offset].next_index;
    }
    
    *result = sum;
}
