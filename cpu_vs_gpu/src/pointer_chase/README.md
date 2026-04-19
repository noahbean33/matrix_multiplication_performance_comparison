# Workload D: Pointer-Chasing / Graph Walk

## Overview

This benchmark implements **pointer-chasing** (linked list traversal) to test the hypothesis: **"Can FPGAs build custom prefetch/arbitration logic?"** 

The answer: **Yes, but ONLY for predictable patterns.**

This workload demonstrates **when FPGAs stop looking magical** — when faced with truly random memory access patterns, all architectures become DDR-bound.

### Key Characteristics

- **Task**: Follow a linked list / graph for K hops, accumulating values
- **Problem Sizes**: 1M nodes (~8 MB)
- **Two Profiles**:
  1. **Predictable**: `next = (current + stride) % length` — can be prefetched
  2. **Unpredictable**: Random permutation — defeats all prefetchers
- **Metrics**: ops/s, latency per hop, DDR transactions
- **Architectural Axis**: **Control + Memory Indirection**

### Expected Results

| Architecture | Predictable | Unpredictable | Notes |
|--------------|-------------|---------------|-------|
| **CPU** | Good (10-30 ns) | OK (50-150 ns) | Hardware prefetch helps |
| **GPU** | Poor (100-300 ns) | Terrible (200-500 ns) | Warp divergence kills |
| **FPGA** | OK (30-80 ns) | Poor (80-150 ns) | Custom prefetch only helps if predictable |

**Key Insight**: For truly random access, everyone is DDR-bound (~100ns latency). No hardware architecture can fix serial dependencies + unpredictable indirection!

---

## The Fundamental Problem

### Serial Dependency

```cpp
for (int hop = 0; hop < num_hops; ++hop) {
    sum += nodes[current].value;
    current = nodes[current].next_index;  // ← DEPENDS on previous iteration!
}
```

**Cannot be parallelized!** Each iteration must wait for the previous one to complete.

### Memory Indirection

```cpp
current = nodes[current].next_index;  // Random memory access
```

- Next address is **data-dependent** (read from memory)
- Hardware prefetchers can't predict (unless pattern is regular)
- Cache effectiveness depends on locality

---

## Directory Structure

```
pointer_chase/
├── cpu/
│   ├── main.cpp           # CPU with stride vs random patterns
│   └── CMakeLists.txt     # Build configuration
├── gpu/
│   ├── main.cu            # CUDA showing warp divergence
│   └── CMakeLists.txt     # CUDA build
├── fpga/
│   ├── pointer_chase_kernel.cpp  # HLS with custom prefetch
│   ├── pointer_chase_kernel.h    # Kernel header
│   ├── host.cpp           # Host code (SW emulation)
│   ├── Makefile           # Vitis build system
│   └── run_hls.tcl        # HLS synthesis script
└── README.md              # This file
```

---

## Building and Running

### CPU Version

**Build:**
```bash
cd cpu
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**Run:**
```bash
./pointer_chase_cpu
```

**Expected Output:**
```
====================================
Pattern: PREDICTABLE (stride)
====================================
Hops per second:   25.123 M hops/s
Latency per hop:   39.802 ns

----- Analysis -----
✓ Predictable pattern allows hardware prefetching
✓ Cache can anticipate next access
Expected: ~10-30 ns/hop (good cache + prefetch)

====================================
Pattern: UNPREDICTABLE (random)
====================================
Hops per second:   8.456 M hops/s
Latency per hop:   118.267 ns

----- Analysis -----
✗ Random pattern defeats prefetchers
✗ Each hop likely causes cache miss
Expected: ~50-150 ns/hop (memory-bound)
```

### GPU Version

**Build:**
```bash
cd gpu
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**Run:**
```bash
./pointer_chase_gpu
```

**Expected Output:**
```
⚠️  This workload is TERRIBLE for GPU!
   - Serial dependencies (can't parallelize within a chain)
   - Scattered memory access (no coalescing)
   - Warp divergence (threads take different times)

====================================
Pattern: PREDICTABLE (stride)
====================================
Latency per hop:   156.234 ns

----- Why GPU is Slow -----
1. SERIAL DEPENDENCY: Can't parallelize within a chain
2. SCATTERED ACCESS: No memory coalescing
3. WARP DIVERGENCE: Threads have different execution times

GPU is ~5-10x SLOWER than CPU for pointer chasing!

====================================
Pattern: UNPREDICTABLE (random)
====================================
Latency per hop:   287.456 ns

Random: TERRIBLE (worst case for GPU architecture)
```

### FPGA Version

**Software Emulation:**
```bash
cd fpga
make sw_emu
make run_sw_emu
```

**Expected Output:**
```
====================================
Pattern: PREDICTABLE (stride)
====================================
FPGA Advantage: Custom prefetch logic can hide latency!
Expected latency: ~30-80 ns/hop
Speedup vs baseline: ~2-3x (with prefetch buffer)
✓ FPGA can pipeline prefetch operations
✓ Multiple outstanding memory requests

====================================
Pattern: UNPREDICTABLE (random)
====================================
FPGA Challenge: Cannot prefetch effectively!
Expected latency: ~80-150 ns/hop (DDR latency)
✗ Serial dependency limits pipelining
✗ Each hop waits for DDR read (~100 ns)

This is where FPGAs STOP looking magical!
```

---

## Performance Metrics

### Metrics Reported

1. **Hops per Second (M hops/s)**
   - How many pointer dereferences per second
   - Higher is better

2. **Latency per Hop (ns)**
   - Time to traverse one link
   - **Most important metric**
   - Theoretical minimum: DDR latency (~50-100 ns)

3. **Memory Bandwidth (GB/s)**
   - Effective bandwidth utilized
   - Usually LOW (pointer chasing is latency-bound, not bandwidth-bound)

4. **DDR Transactions**
   - Number of memory accesses
   - Random pattern: ~1 cache line per hop (64 bytes)
   - Sequential pattern: Better cache utilization

---

## Expected Results

### CPU Performance

| Pattern | Latency/hop | Hops/s | Analysis |
|---------|-------------|--------|----------|
| **Predictable** | 10-30 ns | 33-100 M | Hardware prefetcher learns stride pattern |
| **Random** | 50-150 ns | 6-20 M | Cache misses, but good prefetch queue |

**CPU Wins**: Best general-purpose memory subsystem!

### GPU Performance

| Pattern | Latency/hop | Hops/s | Analysis |
|---------|-------------|--------|----------|
| **Predictable** | 100-300 ns | 3-10 M | Can't overcome serial dependency |
| **Random** | 200-500 ns | 2-5 M | Catastrophic (warp divergence + no coalescing) |

**GPU Loses Badly**: 5-10x slower than CPU!

**Why GPU is terrible:**
- ❌ **Serial dependency**: Each thread chases its own chain → no parallelism
- ❌ **Scattered access**: Threads in a warp access completely different addresses
- ❌ **No coalescing**: Every memory access is uncoalesced
- ❌ **Warp divergence**: Different threads take different times

### FPGA Performance

| Pattern | Latency/hop | Hops/s | Analysis |
|---------|-------------|--------|----------|
| **Predictable** | 30-80 ns | 12-33 M | Custom prefetch logic helps! |
| **Random** | 80-150 ns | 6-12 M | Limited by DDR latency |

**FPGA OK for Predictable, Poor for Random**:
- ✅ **Predictable**: Custom prefetch with 2-3x speedup
- ❌ **Random**: No advantage over CPU

---

## Implementation Details

### CPU Implementation

**Strategy**: Straightforward loop with compiler optimizations

```cpp
float pointer_chase(const Node* nodes, int num_hops, int start) {
    float sum = 0.0f;
    int current = start;
    for (int hop = 0; hop < num_hops; ++hop) {
        sum += nodes[current].value;
        current = nodes[current].next_index;  // Indirection!
    }
    return sum;
}
```

**Why CPU is good:**
- Hardware prefetcher (learns stride patterns)
- Deep memory request queue
- Good branch prediction
- Large caches (L1/L2/L3)

### GPU Implementation

**Strategy**: Each thread chases its own chain

```cuda
__global__ void pointer_chase_kernel(const Node* nodes, ...) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    int current = start_indices[tid];
    
    for (int hop = 0; hop < num_hops; ++hop) {
        sum += nodes[current].value;
        current = nodes[current].next_index;  // SERIAL!
    }
    
    results[tid] = sum;
}
```

**Why GPU fails:**
1. Each thread's loop is **serial** (can't parallelize)
2. Threads access **different addresses** (no coalescing)
3. **Warp divergence** (threads may take different paths)
4. GPU memory system optimized for **coalesced**, **parallel** access

### FPGA Implementation (4 Variants)

#### 1. **Baseline**
- Simple serial loop
- Shows the problem: loop-carried dependency

#### 2. **Custom Prefetch** (Predictable Pattern)
```cpp
const int PREFETCH_DEPTH = 4;
Node prefetch_buffer[PREFETCH_DEPTH];

// Pre-fetch ahead using known pattern
for (int i = 0; i < PREFETCH_DEPTH; ++i) {
    prefetch_buffer[i] = nodes[(start + i * stride) % length];
}

// Chase with prefetching
for (int hop = 0; hop < num_hops; ++hop) {
    sum += prefetch_buffer[buffer_idx].value;
    // Prefetch next (predictable!)
    int next = (current + PREFETCH_DEPTH * stride) % length;
    prefetch_buffer[buffer_idx] = nodes[next];
    buffer_idx = (buffer_idx + 1) % PREFETCH_DEPTH;
}
```

**FPGA Advantage**: Can issue multiple outstanding DDR reads!

#### 3. **Multi-Chain**
- Process multiple independent chains
- Can use DATAFLOW for some parallelism
- Still limited by serial nature of each chain

#### 4. **Burst-Read Optimized**
- Buffer entire regions of memory
- Only works if nodes are stored in access order

---

## Why FPGAs "Stop Looking Magical"

### What FPGAs CAN Do

✅ **Custom prefetch logic**
- Issue multiple outstanding memory requests
- Build specialized prediction hardware
- Overlap computation with memory access

✅ **Predictable patterns**
- Learn stride patterns
- Pipeline prefetch operations
- Hide DDR latency with buffering

### What FPGAs CANNOT Do

❌ **Break serial dependencies**
- `current = nodes[current].next` is **fundamentally serial**
- No amount of custom logic can parallelize this
- Must wait for each DDR read to complete

❌ **Predict random patterns**
- If `next_index` is truly random, no prefetch helps
- DDR latency (~100 ns) becomes the bottleneck
- FPGA has no advantage over CPU

---

## Key Observations

### 1. **Memory Latency Dominates**

All architectures are limited by DDR latency (~50-100 ns):
- **Best case**: ~30 ns (all hits in L1 cache)
- **Worst case**: ~150 ns (every access goes to DDR)

### 2. **Predictability Matters**

| Pattern | CPU | GPU | FPGA |
|---------|-----|-----|------|
| Predictable | ⭐⭐⭐ | ⭐ | ⭐⭐ |
| Random | ⭐⭐ | ❌ | ⭐ |

### 3. **Serial Dependencies Can't Be Fixed**

The loop:
```cpp
for (int hop = 0; hop < num_hops; ++hop) {
    current = nodes[current].next_index;  // SERIAL!
}
```

**Cannot be parallelized** by ANY hardware architecture!

### 4. **GPU's Parallel Model Breaks Down**

- GPU needs **parallel** work
- Pointer chasing is **inherently serial**
- Result: GPU is 5-10x slower than CPU

---

## Profiling and Analysis

### CPU Profiling

```bash
# Perf stat
perf stat -e L1-dcache-load-misses,L1-dcache-loads ./pointer_chase_cpu

# Expected:
# Predictable: ~5-10% L1 miss rate
# Random: ~50-80% L1 miss rate (cold misses)
```

### GPU Profiling

```bash
# Nsight Compute
ncu --set full -o pointer_profile ./pointer_chase_gpu

# Key metrics:
# - Global memory efficiency: <10% (terrible!)
# - Warp execution efficiency: <25% (divergence)
# - Memory throughput: Low (latency-bound)
```

### FPGA Reports

After HLS synthesis:
```
pointer_chase_hls_project/solution1/syn/report/
```

**Expected findings:**
- Loop II: **Cannot pipeline** (loop-carried dependency)
- Latency: `num_hops * DDR_latency`
- Resource usage: Low (this is a memory problem, not compute)

---

## Use Cases (Where This Pattern Appears)

1. **Graph Traversal**
   - Breadth-first search (BFS)
   - Depth-first search (DFS)
   - PageRank

2. **Hash Tables**
   - Collision resolution chains
   - Open addressing with probing

3. **Tree Traversal**
   - Binary search trees
   - B-trees
   - Skip lists

4. **Dynamic Data Structures**
   - Linked lists
   - Adjacency lists
   - Sparse graphs

**Why this matters**: These patterns are common in databases, graph analytics, and pointer-heavy code.

---

## Troubleshooting

### CPU Build Issues

**Error:** `Segmentation fault`
- **Solution:** Chain is likely corrupted. Check random chain generation.

### GPU Build Issues

**Error:** `Kernel timeout`
- **Solution:** Reduce `num_hops` (GPU is very slow on this!)

**Error:** `Poor performance`
- **Solution:** This is EXPECTED! GPU hates pointer chasing.

### FPGA Build Issues

**Error:** `Cannot pipeline loop`
- **Solution:** This is EXPECTED! Loop has serial dependency.

**Error:** `Timing not met`
- **Solution:** Lower frequency or accept longer latency

---

## Extensions

To extend this benchmark:

1. **Add Block-Based Traversal**
   - Process multiple chains in parallel
   - Shows FPGA dataflow benefits

2. **Test Different Node Sizes**
   - Small nodes (8 bytes): Cache-friendly
   - Large nodes (64+ bytes): Cache-unfriendly

3. **Measure Power**
   - GPU wastes power on this workload
   - FPGA might have better energy efficiency

4. **Add Prefetch Distance Tuning**
   - Vary FPGA prefetch buffer size
   - Find optimal depth for different patterns

---

## References

1. **Pointer Chasing:**
   - Hennessy & Patterson. "Computer Architecture: A Quantitative Approach." Chapter 2 (Memory Hierarchy).

2. **GPU Memory Access:**
   - NVIDIA. "CUDA C Programming Guide." Section on Memory Coalescing.

3. **FPGA Memory Systems:**
   - Kapre & DeHon. "Optimistic Parallelization of Floating-Point Accumulation." FPGA 2007.

4. **Cache Performance:**
   - Drepper. "What Every Programmer Should Know About Memory." 2007.

---

## Summary

**This benchmark shows the limits of ALL architectures:**

| Architecture | Strength | Weakness |
|--------------|----------|----------|
| **CPU** | Hardware prefetch, good caches | Still memory-bound on random |
| **GPU** | - | **Catastrophic** on pointer chasing |
| **FPGA** | Custom prefetch logic (predictable) | Can't fix serial dependency |

### Key Takeaways

1. ✅ **CPUs win at pointer chasing** (best general-purpose memory system)
2. ❌ **GPUs are 5-10x slower** (parallel model breaks down)
3. ⚠️ **FPGAs help ONLY if predictable** (custom logic has limits)
4. 💡 **This is where FPGAs STOP looking magical!**

### The Bottom Line

**For truly random memory access with serial dependencies, no hardware architecture can do magic. Everyone is limited by DDR latency (~100ns).**

---

## License

Same as parent repository (see `../../LICENSE`)
