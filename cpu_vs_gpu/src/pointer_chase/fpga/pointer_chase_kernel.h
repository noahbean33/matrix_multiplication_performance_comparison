#ifndef POINTER_CHASE_KERNEL_H
#define POINTER_CHASE_KERNEL_H

// Node structure
struct Node {
    int next_index;
    float value;
};

// Baseline kernel - shows serial dependency problem
void pointer_chase_kernel_baseline(
    const Node* nodes,
    int num_hops,
    int start_index,
    float* result
);

// Optimized with custom prefetch (works for predictable patterns)
void pointer_chase_kernel_prefetch(
    const Node* nodes,
    int num_hops,
    int start_index,
    float* result,
    bool predictable,
    int stride
);

// Multi-chain processing
void pointer_chase_kernel_multichain(
    const Node* nodes,
    int chain_length,
    int num_chains,
    const int* start_indices,
    float* results
);

// Burst-read optimized
void pointer_chase_kernel_burst(
    const Node* nodes,
    int num_hops,
    int start_index,
    float* result
);

#endif // POINTER_CHASE_KERNEL_H
