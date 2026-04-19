# Workload A: Coupled PDE (Navier-Stokes + Temperature)

## Overview

This benchmark implements a **2D incompressible Navier-Stokes solver with temperature coupling** (Boussinesq approximation). It demonstrates **where FPGAs encounter BRAM/memory pressure** — the anchor workload showing FPGA limitations.

**Key Message:** *"FPGA is great... until BRAM/memory becomes the bottleneck."*

### Key Characteristics

- **Equations**: 2D incompressible Navier-Stokes + heat equation
- **Coupling**: Boussinesq buoyancy (temperature drives flow)
- **Solver**: Explicit advection/diffusion + iterative pressure Poisson (Jacobi)
- **Fields**: 5 arrays (u, v, p, T, div)
- **Grid Sizes**: 256², 512², 1024²
- **Precision**: FP32
- **Architectural Axis**: **Local vs Global**, **Multi-Field**, **BRAM Pressure**

### Expected Results

| Architecture | 256² | 512² | 1024² | Key Characteristic |
|--------------|------|------|-------|-------------------|
| **CPU** | OK | Good | Slow | Cache-limited |
| **GPU** | Fast | Fast | Fast | High DRAM bandwidth |
| **FPGA** | OK | Poor | **Very Poor** | **BRAM bottleneck!** |

**Critical Insight**: FPGAs struggle when working set exceeds on-chip memory capacity!

---

## The Physical Problem

### Navier-Stokes Equations

We solve the incompressible Navier-Stokes equations coupled with heat transport:

**Momentum (velocity):**
```
∂u/∂t + u·∇u = -∇p + ν∇²u
∂v/∂t + u·∇v = -∇p + ν∇²v + gβT  (buoyancy term)
```

**Incompressibility (via pressure Poisson):**
```
∇·u = 0  →  ∇²p = ∇·(u·∇u)
```

**Heat equation:**
```
∂T/∂t + u·∇T = α∇²T
```

**Where:**
- `u, v` = x and y velocity components
- `p` = pressure
- `T` = temperature
- `ν` = kinematic viscosity
- `α` = thermal diffusivity
- `g` = gravity
- `β` = thermal expansion coefficient (Boussinesq)

### Algorithm Steps (Per Timestep)

1. **Advection** (LOCAL): u·∇u
2. **Diffusion** (LOCAL): ν∇²u
3. **Buoyancy** (LOCAL): gβT
4. **Divergence** (GLOBAL): ∇·u
5. **Pressure Poisson** (GLOBAL): ∇²p = div (100 Jacobi iterations!)
6. **Projection** (LOCAL): u ← u - ∇p

**Time breakdown:**
- Local operations: ~10% of time
- Pressure Poisson: **~90% of time** (100 Jacobi iterations!)

---

## Directory Structure

```
coupled_pde_2d/
├── cpu/
│   ├── main.cpp           # CPU with Jacobi solver
│   └── CMakeLists.txt     # Build configuration
├── gpu/
│   ├── main.cu            # CUDA with 100 kernel launches/step
│   └── CMakeLists.txt     # CUDA build
├── fpga/
│   ├── pde_kernel.cpp     # HLS kernels (4 variants)
│   ├── pde_kernel.h       # Kernel header
│   ├── host.cpp           # Host code with BRAM analysis
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
./coupled_pde_cpu
```

**Expected Output:**
```
Grid: 512x512, Steps: 1000
Fields: u, v, p, T, div (5 arrays)
Memory: 5.000 MB

----- Performance Metrics -----
Time per step:   45.234 ms
Points/sec:      5.654 M pts/s

----- Operation Breakdown -----
LOCAL operations: Fast
GLOBAL operation (pressure Poisson): Expensive!
  - 100 Jacobi iterations
  - Dominates compute time (~90%)
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
./coupled_pde_gpu
```

**Expected Output:**
```
Grid: 512x512, Steps: 100
GPU Memory: 9.000 MB

----- Performance Metrics -----
Time per step:   20.345 ms
Points/sec:      12.834 M pts/s

----- GPU Analysis -----
LOCAL operations: ✓ Good parallelism
GLOBAL operation: ⚠️ 100 kernel launches per timestep!

Key Limitation: Jacobi iterations don't overlap
But GPU has bandwidth and capacity to handle it
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
Grid: 512x512

----- BRAM Capacity Analysis -----
Total memory: 5.000 MB

Configuration: HYBRID (partial BRAM, frequent DDR access)
⚠️  BRAM bottleneck becomes severe
✗ Cannot fit all fields on-chip
✗ Jacobi iterations particularly slow (100× DDR access)

Conclusion: GPU is likely FASTER for this workload!
```

---

## Performance Metrics

### Metrics Reported

1. **Time per Step (ms)**
   - How long each timestep takes
   - Dominated by Jacobi iterations (~90%)

2. **Points per Second (M pts/s)**
   - Number of grid points processed per second

3. **Throughput (GFLOP/s)**
   - Floating-point operations per second
   - Rough estimate: ~1000 ops per point (including Jacobi)

4. **Memory Bandwidth (GB/s)**
   - Memory traffic during execution
   - High due to Jacobi iterations

### Expected Performance

**CPU (Intel i9):**
| Grid | Memory | Time/Step | Notes |
|------|--------|-----------|-------|
| 256² | 1.25 MB | ~10 ms | Fits in L3 cache |
| 512² | 5.0 MB | ~45 ms | Fits in L3 cache |
| 1024² | 20 MB | ~200 ms | Exceeds L3, DRAM-bound |

**GPU (RTX 3090):**
| Grid | Memory | Time/Step | Notes |
|------|--------|-----------|-------|
| 256² | 2.25 MB | ~3 ms | Fast |
| 512² | 9.0 MB | ~12 ms | Fast |
| 1024² | 36 MB | ~50 ms | Good bandwidth |

**FPGA (Alveo U250):**
| Grid | Memory | Time/Step | Notes |
|------|--------|-----------|-------|
| 256² | 1.25 MB | ~8 ms | **Fits in BRAM** ✓ |
| 512² | 5.0 MB | ~40 ms | **Exceeds BRAM** ✗ |
| 1024² | 20 MB | ~180 ms | **Far exceeds BRAM** ✗ |

**Key Observation:**
- **256²**: FPGA competitive (fits in BRAM)
- **512²+**: GPU **2-4× faster** than FPGA!

---

## The BRAM Bottleneck Explained

### FPGA BRAM Capacity (Alveo U250)

**Total BRAM:** ~96 MB (2688 blocks × 36 KB)  
**Usable per kernel:** ~20-40 MB (rest used by system, other resources)

### Memory Requirements

**5 fields (u, v, p, T, div) × FP32:**

| Grid Size | Per Field | Total | BRAM Status |
|-----------|-----------|-------|-------------|
| 256² | 0.25 MB | **1.25 MB** | ✓ Feasible |
| 512² | 1.00 MB | **5.00 MB** | ⚠️ Tight |
| 1024² | 4.00 MB | **20.0 MB** | ❌ **Exceeds!** |

**Additional pressure:**
- Ping-pong buffers (2× for Jacobi): **40 MB** for 1024²
- Line buffers for streaming: +few KB
- Intermediate results: +few MB

**Conclusion:** 1024² workload **cannot fit** in FPGA BRAM!

### What Happens When BRAM is Exceeded?

**Must use DDR (off-chip memory):**
- DDR latency: ~100 ns vs 1 ns (BRAM)
- DDR bandwidth: ~20-40 GB/s vs ~1000 GB/s (BRAM)
- Random access: Kills burst efficiency

**Impact on Jacobi iterations:**
- 100 iterations × full grid traversal
- Each point: read 4 neighbors + div, write result
- Random access pattern (neighbors)
- **Result**: Performance collapses!

---

## Local vs Global Operations

### LOCAL Operations (Fast for FPGA)

**Advection, Diffusion, Buoyancy, Projection**

**Characteristics:**
- Stencil-based (5-point or 9-point)
- Can use line buffers (streaming)
- Neighbors in predictable pattern
- Can achieve II=1 (1 point/cycle)

**FPGA Implementation:**
```cpp
// Line buffers enable streaming
float line_buf[2][MAX_WIDTH];

for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
#pragma HLS PIPELINE II=1
        // Stream through data with line buffers
        result = stencil_compute(line_buf, ...);
    }
}
```

**Performance:** ~250-300 M pts/s (FPGA @ 300 MHz)

### GLOBAL Operations (Slow for FPGA)

**Pressure Poisson Solve (Jacobi Iteration)**

**Characteristics:**
- 100 iterations with inter-dependencies
- Each iteration needs previous result
- Random access to 4 neighbors
- Cannot stream (need full grid)
- Cannot pipeline across iterations

**FPGA Implementation:**
```cpp
// CANNOT pipeline across iterations!
for (int iter = 0; iter < 100; ++iter) {
#pragma HLS LOOP_TRIPCOUNT min=100 max=100
    // This loop must complete before next iteration
    
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
#pragma HLS PIPELINE II=1  // Only within iteration!
            p_new[i][j] = 0.25 * (p[i+1][j] + p[i-1][j] + 
                                  p[i][j+1] + p[i][j-1] - div[i][j]);
        }
    }
    swap(p, p_new);  // Can't start next until this completes
}
```

**Problems:**
- ❌ Cannot achieve II=1 **across** iterations
- ❌ Each iteration: full DDR read/write (if doesn't fit in BRAM)
- ❌ 100× slower than streaming operations
- ❌ Random access pattern defeats burst optimization

**Performance:** ~50-100 M pts/s (DDR-bound)

---

## Implementation Details

### CPU Implementation

**Strategy:**
- Standard nested loops
- Store all 5 fields in memory
- Jacobi with ping-pong buffers
- Rely on cache hierarchy

**Code structure:**
```cpp
void timestep(Grid2D& grid) {
    advect(...);      // LOCAL
    diffuse(...);     // LOCAL
    add_buoyancy(...);// LOCAL
    compute_divergence(...);  // GLOBAL
    solve_pressure(...);      // GLOBAL (100 Jacobi iters!)
    project_velocity(...);    // LOCAL
}
```

**Performance factors:**
- Working set fits in L3 cache: Good
- Exceeds L3 cache: DRAM-bound
- Jacobi iterations dominate (~90% of time)

### GPU Implementation

**Strategy:**
- Each kernel handles one operation
- Launch 100 kernels for Jacobi iterations
- Store all fields in GPU DRAM

**CUDA kernels:**
```cuda
__global__ void advect_kernel(...)        // LOCAL
__global__ void diffuse_kernel(...)       // LOCAL
__global__ void add_buoyancy_kernel(...)  // LOCAL
__global__ void compute_divergence_kernel(...)  // GLOBAL
__global__ void jacobi_iteration_kernel(...)    // GLOBAL (called 100×!)
__global__ void project_velocity_kernel(...)    // LOCAL
```

**Key observations:**
- ✓ GPU has plenty of DRAM (8-24 GB)
- ✓ High bandwidth (500+ GB/s)
- ⚠️ 100 kernel launches per timestep
- ⚠️ Kernel launch overhead (~1-5 μs each)
- ⚠️ Cannot overlap Jacobi iterations (dependency)

**But:** GPU bandwidth and parallelism overcome the overhead!

### FPGA Implementation (4 Variants)

#### 1. DDR-Based (Large Grids)

All fields in DDR memory:
```cpp
void pde_kernel_ddr(
    const float* u_in, const float* v_in, const float* p_in, ...
)
```

**Characteristics:**
- ❌ All operations access DDR
- ❌ ~100 ns latency per access
- ❌ Jacobi iterations: 100× full DDR traversal
- ❌ Random access pattern (no burst efficiency)

**When to use:** Grids > 512² (no choice!)

#### 2. BRAM-Based (Small Grids Only)

Entire grid fits in BRAM:
```cpp
void pde_kernel_bram(...)
{
    float u_buf[MAX_SIZE * MAX_SIZE];  // BRAM!
    float v_buf[MAX_SIZE * MAX_SIZE];
    // ...
}
```

**Characteristics:**
- ✓ 1 cycle BRAM access
- ✓ Can process multiple iterations on-chip
- ✓ No DDR round trips during Jacobi
- ❌ **Only works for ≤ 256²**

**When to use:** Small grids (≤ 256²)

#### 3. Streaming Advection

Line-buffer architecture:
```cpp
void pde_kernel_advect_streaming(...)
{
    float line_buf[2][MAX_WIDTH];  // Only 2 rows!
    // Stream through, compute on-the-fly
}
```

**Characteristics:**
- ✓ II=1 achievable
- ✓ Minimal BRAM (just line buffers)
- ✓ Sequential DDR access (burst efficient)
- ✓ **This works well!**

**When to use:** Advection/diffusion steps

#### 4. Multi-Iteration

Process multiple timesteps on-chip:
```cpp
void pde_kernel_multi_iter(...)
{
    // Load grid to BRAM
    // Process N iterations
    // Write final result
}
```

**Characteristics:**
- ✓ Eliminates DDR round trips between iterations
- ✓ Good for small grids
- ❌ Still has Jacobi iteration problem
- ❌ **Only works for ≤ 256²**

**When to use:** Small grids, batch processing

---

## Why GPU Wins for This Workload

### GPU Advantages

✅ **Large DRAM Capacity**
- 8-24 GB vs FPGA's ~96 MB BRAM
- Can handle 1024² easily (36 MB)
- No capacity constraints

✅ **High Memory Bandwidth**
- 500-900 GB/s (GDDR6/HBM)
- vs FPGA's 20-40 GB/s (DDR4)
- Jacobi iterations are bandwidth-bound

✅ **Massive Parallelism**
- Thousands of cores
- Hide Jacobi iteration overhead with parallelism
- Good utilization even with 100 kernel launches

✅ **Mature Ecosystem**
- CUDA well-optimized for iterative solvers
- Libraries available (cuSPARSE, etc.)

### FPGA Challenges

❌ **BRAM Capacity**
- 20-40 MB usable per kernel
- 5 fields × 1024² = 20 MB (exceeds!)
- Must use DDR → performance collapse

❌ **Lower DDR Bandwidth**
- 20-40 GB/s vs GPU's 500+ GB/s
- Jacobi iterations become bottleneck
- Random access kills burst efficiency

❌ **Iterative Solver Pipeline**
- Cannot pipeline across Jacobi iterations
- Must wait for each iteration to complete
- No streaming dataflow possible

❌ **Multi-Field Complexity**
- Need simultaneous access to 5 fields
- Line-buffer approach doesn't work (not streaming)
- Ping-pong buffers double memory requirement

---

## When FPGA is Competitive

**256² grid:**
- ✓ Fits in BRAM (1.25 MB)
- ✓ Can process multiple iterations on-chip
- ✓ Avoid DDR round trips
- **Result**: Competitive with GPU (lower power!)

**But:**
- Most real applications need larger grids
- 256² is too small for many CFD problems
- Real value: **understanding the limits**

---

## Profiling and Analysis

### CPU Profiling

```bash
# Cache performance
perf stat -e LLC-load-misses,LLC-loads ./coupled_pde_cpu

# Expected:
# 256²: ~5% LLC miss rate (fits in cache)
# 1024²: ~40% LLC miss rate (exceeds cache)
```

### GPU Profiling

```bash
# Nsight Compute
ncu --set full -o pde_profile ./coupled_pde_gpu

# Key metrics:
# - Kernel launch overhead: ~5 μs × 100 = 0.5 ms
# - Memory bandwidth: ~80-90% utilization
# - Compute utilization: ~60-70%
```

### FPGA Reports

After HLS synthesis:
```
pde_hls_project/solution1/syn/report/
```

**Expected findings:**
- **256²**: BRAM ~35 blocks (feasible)
- **512²**: BRAM ~140 blocks (tight)
- **1024²**: BRAM ~560 blocks (**EXCEEDS AVAILABLE!**)
- Synthesis **will fail** for 1024² BRAM version

---

## Use Cases

**This pattern appears in:**

1. **Computational Fluid Dynamics (CFD)**
   - Navier-Stokes solvers
   - Weather/climate models
   - Aerodynamics simulations

2. **Heat Transfer**
   - Thermal simulations
   - Conjugate heat transfer
   - Cooling system design

3. **Multi-Physics**
   - Fluid-structure interaction
   - Combustion
   - Magnetohydrodynamics

**Common characteristic:** Multiple coupled fields + iterative solver

**Why it matters:** Represents a large class of engineering simulations!

---

## Key Takeaways

### For FPGAs

❌ **BRAM capacity is the hard limit**
- 5 fields × N² × FP32 grows fast
- 1024² requires 20 MB (exceeds BRAM)
- Must use DDR → performance penalty

❌ **Iterative solvers don't pipeline well**
- Jacobi iterations have dependencies
- Cannot achieve streaming dataflow
- Random access pattern (neighbors)

❌ **Multi-field workloads challenging**
- Need simultaneous access to multiple arrays
- Line-buffer optimization doesn't apply
- Memory bandwidth becomes bottleneck

### For GPUs

✓ **Excels at this workload**
- Plenty of DRAM capacity
- High bandwidth handles Jacobi iterations
- Massive parallelism hides overhead

⚠️ **But not perfect**
- 100 kernel launches add overhead
- Cannot overlap Jacobi iterations
- Still memory bandwidth intensive

### The Bottom Line

**"FPGA is great... until BRAM/memory becomes the bottleneck."**

This benchmark demonstrates:
- ✅ Small grids (256²): FPGA competitive
- ⚠️ Medium grids (512²): FPGA struggles
- ❌ Large grids (1024²+): GPU clearly wins

**When to use FPGA:** Single-field, streamable workloads that fit in BRAM  
**When to use GPU:** Multi-field, large grids, iterative solvers

---

## Troubleshooting

### FPGA Build Issues

**Error:** `BRAM resource exhaustion`
```
ERROR: [HLS 200-1510] Allocation failed: cannot allocate more storage.
       Allocated: 2688 BRAM_18K  Available: 2688 BRAM_18K
```
**Solution:** Reduce grid size or use DDR-based kernel

**Error:** `Timing not met`
**Solution:** Lower clock frequency or accept longer latency

### GPU Build Issues

**Error:** `Out of memory`
**Solution:** Reduce grid size or number of simultaneous fields

**Error:** `Slow performance`
**Solution:** This is expected! Jacobi iterations dominate.

---

## Extensions

To extend this benchmark:

1. **Use Better Solver**
   - Replace Jacobi with Conjugate Gradient
   - Multigrid methods
   - Shows solver impact on FPGA viability

2. **Add More Fields**
   - Turbulence model (k-ε)
   - Chemical species
   - Shows linear scaling of BRAM pressure

3. **Test Different Precisions**
   - FP16 halves memory requirement
   - INT16 for some fields
   - May make FPGA more competitive

4. **Measure Power**
   - FPGA: ~20-40W
   - GPU: ~200-350W
   - Energy efficiency comparison

---

## References

1. **Navier-Stokes CFD:**
   - Ferziger & Peric. "Computational Methods for Fluid Dynamics." 2002.

2. **FPGA CFD:**
   - Skalicky et al. "FPGA-based Navier-Stokes Solver." FPL 2015.

3. **GPU CFD:**
   - Thibault & Senocak. "CUDA Implementation of Navier-Stokes." JCP 2009.

4. **FPGA BRAM Limits:**
   - Xilinx. "UltraScale+ FPGA Memory Resources." UG573.

---

## Summary

This benchmark is the **anchor workload** showing FPGA limitations:

**✅ What we learned:**
- FPGAs have strict BRAM capacity limits (~20-40 MB usable)
- Multi-field workloads quickly exceed on-chip memory
- Iterative global solvers don't pipeline well
- GPU wins when memory capacity and bandwidth matter

**❌ Where FPGA struggles:**
- Large working sets (> BRAM capacity)
- Multiple coupled fields
- Iterative solvers with dependencies
- Random access patterns

**✓ Where FPGA excels:**
- Small working sets (< 20 MB)
- Single-field streaming operations
- Local stencil operations
- Power-constrained environments

**The tagline:** *"FPGA is great... until BRAM/memory becomes the bottleneck."*

This completes the architectural story of GPU vs CPU vs FPGA!

---

## License

Same as parent repository (see `../../LICENSE`)
