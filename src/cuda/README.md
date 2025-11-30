# CUDA Matrix Multiplication - GPU Acceleration

## Status

✅ **CORE IMPLEMENTATION COMPLETED**

**Implemented**:
- Naive CUDA kernel (global memory only)
- Tiled CUDA kernel with shared memory optimization
- CSV output with comprehensive timing metrics
- High-resolution timing for kernel and memory transfers
- GFLOPS calculation
- Automatic correctness verification
- Block size variations (16×16, 32×32)
- Matrix sizes: 512, 1024, 2048, 4096

**Pending**:
- cuBLAS comparison (Experiment 6)
- Mixed precision FP16/FP32 (Experiment 7)
- Pinned memory and async transfers (Experiment 8)
- Bank conflict avoidance optimization
- Additional block size variations (8×8, 16×32, etc.)

## Purpose

Quantify GPU acceleration performance and identify optimal CUDA kernel configurations for massive parallelism.

## Usage

```bash
# Compile
make

# Run benchmarks
./matrix_multiplication > cuda_results.csv

# Or use make targets
make benchmark
```

## Output Format

CSV with columns:
```
timestamp,implementation,matrix_size,total_time_ms,total_gflops,kernel_time_ms,h2d_time_ms,d2h_time_ms,block_size,node,verification
```

## Suggested Experiments

### Experiment 1: Basic GPU Speedup ⭐ CRITICAL
**Goal**: Measure raw GPU speedup vs CPU baseline

**Parameters**:
- Matrix sizes: 512, 1024, 2048, 4096, 8192
- Block sizes: 16×16, 32×32
- Grid configuration: Auto-calculated

**Metrics**:
- Kernel execution time
- Memory transfer time (H2D, D2H)
- Total time including transfers
- GFLOPS achieved
- GPU utilization %

**Expected Results**:
- Small matrices (<1024): Transfer overhead dominates
- Large matrices (>2048): 50-200x speedup over CPU
- Peak GFLOPS: 1000-5000 (depends on GPU)

**Key Performance Win**: 100-200x speedup for large matrices

### Experiment 2: Block/Grid Dimension Optimization ⭐
**Goal**: Find optimal thread block configuration

**Block sizes to test**:
- 8×8, 16×16, 32×32, 16×32, 32×16

**Matrix sizes**: 1024, 2048, 4096

**Expected Results**:
- 16×16 or 32×32 typically optimal
- Must be multiple of warp size (32)
- 20-50% performance variation

**Key Performance Win**: 30-50% improvement with optimal block size

### Experiment 3: Shared Memory Optimization ⭐⭐
**Goal**: Use shared memory for tile-based computation

**Implementations**:
1. Global memory only (naive)
2. Shared memory tiling (block size tiles)
3. Optimized shared memory with bank conflict avoidance

**Tile sizes**: 16, 32, 64

**Expected Results**:
- 5-10x improvement over global memory only
- Bank conflicts can reduce 20-30% performance

**Key Performance Win**: Shared memory gives 5-10x speedup

### Experiment 4: Memory Coalescing Analysis
**Goal**: Impact of memory access patterns

**Variations**:
- Row-major access (coalesced)
- Column-major access (non-coalesced)
- Strided access

**Expected Impact**: 3-5x difference between coalesced and non-coalesced

### Experiment 5: Occupancy Optimization
**Goal**: Maximize GPU occupancy

**Parameters to vary**:
- Threads per block: 64, 128, 256, 512, 1024
- Shared memory usage
- Register usage

**Tool**: Use `nvprof` or `nsys` to measure occupancy

**Expected Result**: Occupancy >50% for good performance

### Experiment 6: cuBLAS Comparison ⭐ CRITICAL
**Goal**: Compare custom kernel vs optimized library

**Implementations**:
- Custom naive kernel
- Custom tiled kernel
- cuBLAS SGEMM/DGEMM

**Expected Results**:
- cuBLAS: Near-peak performance (80-90% of theoretical)
- Custom tiled: 50-70% of cuBLAS
- Naive: 5-10% of cuBLAS

**Key Insight**: Quantify optimization ceiling

### Experiment 7: Mixed Precision Performance
**Goal**: FP32 vs FP16 performance

**Variations**:
- FP64 (double)
- FP32 (float)
- FP16 (half) with Tensor Cores if available

**Expected Results**:
- FP32: 2x faster than FP64
- FP16: 2-8x faster than FP32 (on Tensor Core GPUs)

### Experiment 8: Transfer Overhead Analysis
**Goal**: Quantify PCIe bottleneck

**Breakdown**:
```
Total time = H2D transfer + Kernel + D2H transfer
```

**Matrix sizes**: 512, 1024, 2048, 4096, 8192

**Optimization**: Pinned memory, async transfers, streams

**Expected Impact**: Pinned memory 2-3x faster than pageable

## Most Important Performance Wins (Ranked)

1. **GPU vs CPU baseline** - 100-200x speedup (⭐⭐⭐⭐⭐)
2. **Shared memory tiling** - 5-10x over naive GPU (⭐⭐⭐⭐⭐)
3. **Optimal block dimensions** - 30-50% improvement (⭐⭐⭐)
4. **Memory coalescing** - 3-5x improvement (⭐⭐⭐⭐)
5. **Mixed precision (FP16)** - 2-8x on Tensor Cores (⭐⭐⭐⭐)

**Total Potential**: 1000-5000 GFLOPS (vs 5-10 GFLOPS CPU)

## Implementation Notes

### Main Implementation
- **matrix_multiplication.cu**: Production benchmarking code implementing naive and tiled kernels

### Legacy Files (Reference Only)
- **01_naive.cu**: Original naive implementation (superseded)
- **02_step1.cu**: Original tiled implementation (superseded)
- **03_step.cu**: Original vectorized implementation (reference for future optimization)

The refactored `matrix_multiplication.cu` consolidates these implementations into a comprehensive benchmarking framework with:
- Professional documentation
- Consistent CSV output format matching the naive CPU implementation
- Comprehensive timing breakdown (kernel, H2D, D2H)
- Automatic verification
- Support for multiple configurations

### Key Design Decisions

1. **Template-based tiling**: Uses C++ templates to support different tile sizes at compile time for better performance
2. **Separate timing**: Distinguishes between kernel execution time and memory transfer overhead
3. **Fixed seed**: Uses `srand(42)` for reproducible benchmarks
4. **Error checking**: All CUDA calls wrapped with error checking macros
5. **Verification**: Each benchmark automatically verifies correctness against CPU reference

### Compilation Requirements

- CUDA Toolkit 10.0 or later
- GPU with compute capability 6.0 or higher (adjust `-arch` in Makefile)
- C++11 compatible compiler
