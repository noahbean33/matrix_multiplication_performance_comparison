# Workload C: Sparse Matrix-Vector Multiplication (SpMV)

## Overview

This benchmark implements **Sparse Matrix-Vector Multiplication (SpMV)** using the **Compressed Sparse Row (CSR)** format across three architectures: CPU, GPU, and FPGA. The goal is to demonstrate how **irregular memory access patterns** affect performance differently on each platform.

### Key Characteristics

- **Operation**: `y = A * x` where `A` is a sparse matrix in CSR format
- **Sparsity Pattern**: 5-9 non-zeros per row (irregular, like real-world sparse matrices)
- **Problem Sizes**: 100K and 1M rows
- **Precision**: FP32
- **Memory Pattern**: Highly irregular (pointer-chasing through `col_indices[]`)

### Architectural Insights

| Architecture | Expected Performance | Key Factors |
|--------------|---------------------|-------------|
| **CPU** | Decent | Good caches, hardware prefetching handles irregular access |
| **GPU** | Poor (baseline) | **Uncoalesced memory access** on `x[col_indices[j]]` kills bandwidth |
| **FPGA** | Good (with tuning) | Custom dataflow, dedicated memory ports, pipelined irregular access |

---

## Directory Structure

```
spmv_structured/
├── cpu/
│   ├── main.cpp           # CPU implementation with CSR format
│   └── CMakeLists.txt     # Build configuration
├── gpu/
│   ├── main.cu            # CUDA implementation (shows coalescing issues)
│   └── CMakeLists.txt     # CUDA build configuration
├── fpga/
│   ├── spmv_kernel.cpp    # HLS kernel with 3 optimization variants
│   ├── spmv_kernel.h      # Kernel header
│   ├── host.cpp           # Host code (SW emulation + HW execution)
│   ├── Makefile           # Vitis build system
│   └── run_hls.tcl        # HLS synthesis script
└── README.md              # This file
```

---

## Building and Running

### CPU Version

**Requirements:**
- C++17 compiler (GCC, Clang, or MSVC)
- CMake 3.10+

**Build:**
```bash
cd cpu
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**Run:**
```bash
./spmv_cpu
```

**Expected Output:**
```
====================================
SpMV CPU Benchmark - CSR Format
Matrix Size: 100000 rows
====================================
Generating sparse matrix (nnz/row: 5-9)...
Total non-zeros: 700000
Avg nnz/row: 7.0

----- Performance Metrics -----
Avg Time per SpMV: 2.345 ms
Throughput:        0.597 GFLOP/s
Bandwidth:         3.456 GB/s
Arithmetic Intensity: 0.172 FLOP/Byte
```

### GPU Version

**Requirements:**
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+ (with nvcc)
- CMake 3.18+

**Build:**
```bash
cd gpu
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

**Run:**
```bash
./spmv_gpu
```

**Expected Output:**
```
====================================
SpMV GPU Benchmark - CSR Format
Using GPU: NVIDIA RTX 3090
Compute Capability: 8.6
====================================
Matrix Size: 100000 rows
Total non-zeros: 700000

----- Performance Metrics -----
Avg Time per SpMV: 0.523 ms
Throughput:        2.678 GFLOP/s
Bandwidth:         15.432 GB/s

----- Memory Access Pattern -----
WARNING: Irregular col_indices[] causes UNCOALESCED memory access!
x[col_indices[j]] reads are scattered across memory.
Expected: Poor cache hit rate and memory throughput.
```

**Note:** GPU performance is limited by uncoalesced memory access. Real-world optimizations include:
- Matrix reordering (e.g., reverse Cuthill-McKee)
- Vectorized formats (ELL, ELLPACK)
- Texture memory caching

### FPGA Version

**Requirements:**
- Xilinx Vitis HLS 2022.1+ (for synthesis)
- Xilinx XRT (for hardware execution)
- Target: Alveo U250 or similar

#### Software Emulation (No FPGA Required)

**Build:**
```bash
cd fpga
make sw_emu
```

**Run:**
```bash
make run_sw_emu
# or directly:
./build/host_sw
```

This runs a functional C++ simulation of the HLS kernel.

#### Hardware Emulation

**Build:**
```bash
cd fpga
make hw_emu
```

**Run:**
```bash
make run_hw_emu
```

**Time:** ~30-60 minutes (includes RTL simulation)

#### Hardware Synthesis

**Build:**
```bash
cd fpga
make hw
```

**Time:** 2-4 hours (full FPGA bitstream generation)

**Run:**
```bash
make run_hw
```

#### HLS Synthesis Report Only

**Quick synthesis report (no full build):**
```bash
cd fpga
make report
```

Check `spmv_hls_project/solution1/syn/report/` for:
- Latency/interval estimates
- Resource utilization (LUT, FF, BRAM, DSP)
- Pipeline II (initiation interval)

---

## Performance Metrics

### Metrics Reported

1. **Throughput (GFLOP/s)**
   - FLOPs = 2 × nnz (one multiply, one add per non-zero)
   
2. **Bandwidth (GB/s)**
   - Reads: `values[]`, `col_indices[]`, `row_ptr[]`, `x[]`
   - Writes: `y[]`
   - Total ≈ `(nnz × 8 + nnz × 4 + num_rows × 4 + nnz × 4 + num_rows × 4)` bytes

3. **Arithmetic Intensity (FLOP/Byte)**
   - SpMV is memory-bound: AI ≈ 0.15-0.20 FLOP/Byte
   - Compare to dense GEMM: AI ≈ 10-100 FLOP/Byte

4. **GPU-Specific**
   - Uncoalesced load warnings
   - Cache miss rates (via profiler)

5. **FPGA-Specific**
   - Pipeline II (initiation interval)
   - BRAM usage for buffering
   - Clock frequency achieved
   - Number of memory ports

---

## Implementation Variants

### CPU
- **Baseline CSR**: Simple row-wise loop
- **Optimizations**: Compiler auto-vectorization, prefetching

### GPU
- **Baseline**: One thread per row (shows poor coalescing)
- **Notes**: For better performance, consider:
  - Warp-level reduction
  - CSR-Adaptive (variable threads per row)
  - Format conversion to ELLPACK or BSR

### FPGA (3 Variants)

1. **`spmv_csr_kernel`** (Baseline)
   - Pipelined inner loop (II=1)
   - 5 independent AXI memory ports
   - Handles irregular access via pipelining

2. **`spmv_csr_kernel_optimized`**
   - BRAM buffer for frequently accessed `x[]` elements
   - Trade-off: BRAM vs DDR bandwidth
   - Suitable for matrices with locality

3. **`spmv_csr_kernel_dataflow`**
   - Task-level pipelining (producer-consumer)
   - Splits read/compute/write stages
   - Connected via FIFOs for streaming

---

## Expected Results

### Relative Performance (100K rows, 7 nnz/row)

| Platform | Time (ms) | GFLOP/s | GB/s | Notes |
|----------|-----------|---------|------|-------|
| CPU (Intel i9) | ~2-5 | 0.3-0.7 | 2-5 | Good cache performance |
| GPU (RTX 3090) | ~0.5-1 | 1-3 | 10-30 | Limited by uncoalesced access |
| FPGA (U250 @ 300MHz) | ~1-3 | 0.5-1.5 | 5-15 | Custom dataflow, full pipeline |

**Key Observation:** GPU advantage is **much smaller** than in dense workloads due to irregular memory pattern.

---

## Profiling and Analysis

### CPU Profiling

```bash
# Linux perf
perf stat -e cache-misses,cache-references ./spmv_cpu

# Valgrind cachegrind
valgrind --tool=cachegrind ./spmv_cpu
```

### GPU Profiling

```bash
# NVIDIA Nsight Compute
ncu --set full -o spmv_profile ./spmv_gpu

# Key metrics:
# - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum (global loads)
# - smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct (coalescing efficiency)
```

### FPGA Reports

After `make hw_emu` or `make hw`, check:
```
build/_x/logs/link/vivado.log          # Vivado synthesis log
build/_x/reports/                       # Resource/timing reports
spmv_hls_project/solution1/syn/report/ # HLS synthesis report
```

---

## Matrix Format Details

### CSR (Compressed Sparse Row)

**Storage:**
- `values[nnz]`: Non-zero values (FP32)
- `col_indices[nnz]`: Column index for each value (int32)
- `row_ptr[num_rows+1]`: Start index of each row in `values[]`

**Example:** 4×4 matrix with 7 non-zeros
```
A = [ 1.0  0    2.0  0   ]
    [ 0    3.0  0    0   ]
    [ 4.0  0    5.0  6.0 ]
    [ 0    7.0  0    0   ]

values      = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
col_indices = [0,   2,   1,   0,   2,   3,   1  ]
row_ptr     = [0,   2,   3,   6,   7              ]
```

**SpMV Algorithm:**
```cpp
for (int row = 0; row < num_rows; row++) {
    float sum = 0.0f;
    for (int j = row_ptr[row]; j < row_ptr[row+1]; j++) {
        sum += values[j] * x[col_indices[j]];  // Irregular!
    }
    y[row] = sum;
}
```

---

## Troubleshooting

### CPU Build Issues

**Error:** `CMake version too old`
- **Solution:** Upgrade CMake or use a newer compiler

### GPU Build Issues

**Error:** `nvcc not found`
- **Solution:** Install CUDA Toolkit and add to PATH
  ```bash
  export PATH=/usr/local/cuda/bin:$PATH
  ```

**Error:** `Unsupported GPU architecture`
- **Solution:** Edit `CMakeLists.txt` and adjust `CUDA_ARCHITECTURES`

### FPGA Build Issues

**Error:** `v++ not found`
- **Solution:** Source Vitis environment
  ```bash
  source /tools/Xilinx/Vitis/2022.1/settings64.sh
  ```

**Error:** `Platform not found`
- **Solution:** Install target platform
  ```bash
  # List available platforms
  platforminfo -l
  
  # Set in Makefile or environment
  export PLATFORM=xilinx_u250_gen3x16_xdma_4_1_202210_1
  ```

---

## References

1. **CSR Format:**
   - Davis, T. A. "Direct Methods for Sparse Linear Systems." SIAM, 2006.

2. **GPU SpMV Optimization:**
   - Bell, N., & Garland, M. "Efficient Sparse Matrix-Vector Multiplication on CUDA." NVIDIA Technical Report, 2008.

3. **FPGA SpMV:**
   - Zhuo, L., & Prasanna, V. K. "Sparse Matrix-Vector Multiplication on FPGAs." ACM FPGA, 2005.

4. **Benchmark Context:**
   - See `../../README.md` for overall benchmark suite design

---

## Contributing

To extend this benchmark:

1. **Add BSR (Block Sparse Row) variant** for better FPGA pipeline utilization
2. **Implement matrix reordering** (RCM, Nested Dissection) for locality
3. **Add vectorized formats** (ELLPACK, DIA) for GPU comparison
4. **Profile with real sparse matrices** from SuiteSparse collection

---

## License

Same as parent repository (see `../../LICENSE`)
