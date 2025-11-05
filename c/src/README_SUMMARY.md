# C/C++ Implementation Directories - Quick Reference

## Overview

Each subdirectory contains a detailed README with suggested experiments focused on quantifying HPC performance wins. All experiments are designed to generate CSV data for comparative analysis.

## Directory Structure

```
c/src/
├── naive/          - Baseline implementation (no optimization)
├── openmp/         - Shared-memory parallelization
├── mpi/            - Distributed-memory parallelization
├── cuda/           - GPU acceleration
├── cache_opt/      - Cache hierarchy optimizations
├── compiler_opt/   - Compiler flag experiments
└── algorithms/     - Advanced algorithms (Strassen, etc.)
```

## Quick Reference: Expected Performance Wins

### Performance Impact Summary

| Optimization | Expected Speedup | Effort | Priority | Best For |
|-------------|------------------|--------|----------|----------|
| **CUDA GPU** | 100-200x | Medium-High | ⭐⭐⭐⭐⭐ | N ≥ 2048 |
| **OpenMP + Cache** | 50-100x | Medium | ⭐⭐⭐⭐⭐ | N ≥ 1024 |
| **MPI Multi-Node** | 20-40x | High | ⭐⭐⭐⭐⭐ | Very large N |
| **Cache Blocking** | 5-15x | Low-Medium | ⭐⭐⭐⭐⭐ | All sizes |
| **Compiler -O3** | 5-10x | Trivial | ⭐⭐⭐⭐⭐ | All |
| **OpenMP Alone** | 6-20x | Low | ⭐⭐⭐⭐⭐ | N ≥ 512 |
| **Loop Reorder** | 2-5x | Trivial | ⭐⭐⭐⭐ | All |
| **-march=native** | 1.3-2x | Trivial | ⭐⭐⭐⭐ | All |
| **Thread Affinity** | 1.3-1.5x | Low | ⭐⭐⭐⭐ | NUMA systems |
| **Strassen** | 1.2-1.4x | High | ⭐⭐ | N > 4096 |

## Critical Experiments by Directory

### 📁 naive/
**Focus**: Baseline characterization
- Loop ordering impact (6 variations)
- Scaling behavior (O(N³) verification)
- Cache hierarchy boundaries
**Key Metric**: GFLOPS vs matrix size
**Expected**: 2-10 GFLOPS

### 📁 openmp/
**Focus**: Shared-memory scaling
- Strong scaling (fixed N, vary threads)
- Weak scaling (N grows with threads)
- Thread affinity impact
- Schedule strategy comparison
**Key Metric**: Parallel efficiency
**Expected**: 75-87% efficiency on 8 cores

### 📁 mpi/
**Focus**: Distributed-memory scaling
- Strong scaling (vary processes)
- Communication vs computation ratio
- Distribution strategies (1D vs 2D)
- Multi-node performance
**Key Metric**: Communication overhead %
**Expected**: 60-80% efficiency on 32 processes

### 📁 cuda/
**Focus**: GPU acceleration
- GPU vs CPU speedup
- Block/grid optimization
- Shared memory tiling
- cuBLAS comparison
**Key Metric**: GFLOPS (absolute)
**Expected**: 1000-5000 GFLOPS

### 📁 cache_opt/
**Focus**: Cache hierarchy exploitation
- Block size optimization
- Multi-level blocking (L1/L2/L3)
- Prefetching impact
- Loop interchange
**Key Metric**: Cache miss rate
**Expected**: 5-15x over naive

### 📁 compiler_opt/
**Focus**: Compiler-driven optimization
- Optimization levels (-O0 to -Ofast)
- Architecture-specific flags
- Vectorization analysis
- Link-time optimization
**Key Metric**: Auto-vectorization success
**Expected**: 5-10x for -O3 vs -O0

### 📁 algorithms/
**Focus**: Algorithmic complexity reduction
- Strassen's algorithm
- Cache-oblivious algorithm
- Crossover point identification
**Key Metric**: Operations count vs standard
**Expected**: 20-40% for N > 2048 (limited practical gain)

## Recommended Experiment Workflow

### Phase 1: Establish Baseline
1. Implement `naive/` - Standard triple loop
2. Measure with `-O2` (fair baseline)
3. Test all 6 loop orderings
4. Identify performance limiters

### Phase 2: Compiler Wins
1. Implement `compiler_opt/` - Same code, different flags
2. Compare -O0, -O1, -O2, -O3, -Ofast
3. Add -march=native
4. Establish "optimized baseline"

### Phase 3: Cache Optimization
1. Implement `cache_opt/` - Blocking algorithm
2. Test block sizes: 16, 32, 64, 128, 256
3. Multi-level blocking
4. Combined with best compiler flags

### Phase 4: Parallelization
1. Implement `openmp/` - Parallel for loops
2. Strong scaling study (1-32 threads)
3. Combine with cache blocking
4. Thread affinity experiments

### Phase 5: Distributed Computing
1. Implement `mpi/` - Row-wise distribution
2. Strong scaling (1-64 processes)
3. Communication analysis
4. Multi-node benchmarks

### Phase 6: GPU Acceleration
1. Implement `cuda/` - Basic kernel
2. Shared memory tiling
3. Block dimension optimization
4. Compare with cuBLAS

### Phase 7: Advanced (Optional)
1. Implement `algorithms/` - Strassen's
2. Compare with optimized O(N³)
3. Document crossover points

## CSV Output Format (Standardized)

All implementations should output:
```csv
timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node
2025-11-04T22:00:00,openmp,2048,125.3,109.5,8,1,node01
```

## Key Performance Metrics to Track

1. **Execution Time** (ms) - Primary metric
2. **GFLOPS** - Normalized performance
3. **Speedup** - Relative to naive baseline
4. **Efficiency** - Speedup / resources
5. **Scalability** - Strong and weak
6. **Communication Overhead** - For MPI
7. **Memory Bandwidth** - GB/s utilized

## Performance Targets (8192×8192 matrix)

| Implementation | Target GFLOPS | Target Time |
|---------------|---------------|-------------|
| Naive -O0 | 2 | ~500 sec |
| Naive -O3 | 10 | ~100 sec |
| Cache-Opt | 50 | ~20 sec |
| OpenMP-8 | 300 | ~3 sec |
| OpenMP-32 | 800 | ~1.2 sec |
| CUDA | 5000 | ~0.2 sec |
| MPI-64 | 4000 | ~0.25 sec |

## Most Important Takeaways

### For Single-Node Performance:
1. **Compiler optimizations are free** - Always use -O3 -march=native
2. **Cache blocking is essential** - 5-15x improvement alone
3. **OpenMP scales well** - Up to physical cores (not hyperthreads)
4. **Combine optimizations** - Synergistic effects

### For Multi-Node Performance:
1. **MPI communication is expensive** - Minimize communication
2. **Problem size matters** - Small matrices don't scale
3. **Distribution strategy critical** - 2D better than 1D for large P
4. **Network is the bottleneck** - Not computation

### For GPU Performance:
1. **GPUs dominate for large matrices** - 100-200x speedup
2. **Memory transfers are costly** - Keep data on GPU
3. **Shared memory is critical** - 5-10x over naive GPU kernel
4. **cuBLAS is highly optimized** - Hard to beat

## Ready-to-Run SLURM Scripts

After implementation, use:
- `scripts/slurm/benchmark_naive.sh`
- `scripts/slurm/benchmark_openmp.sh`
- `scripts/slurm/benchmark_mpi.sh`
- `scripts/slurm/benchmark_cuda.sh`

## Analysis Tools Ready

After benchmarking:
```bash
python python/analysis/compare_implementations.py
python python/visualization/plot_speedup.py
```

---

**Next Step**: Start with `naive/README.md` and implement the baseline. Each directory's README contains detailed experiment specifications.

See `OPTIMIZATION_PRIORITIES.md` for complete performance impact ranking and implementation roadmap.
