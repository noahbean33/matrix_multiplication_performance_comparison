# CUDA Matrix Multiplication - GPU Acceleration

## Purpose

Quantify GPU acceleration performance and identify optimal CUDA kernel configurations for massive parallelism.

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
