# OpenMP Matrix Multiplication - Shared Memory Parallelization

## Purpose

Quantify parallel performance gains using OpenMP on multi-core CPUs. Demonstrate strong and weak scaling characteristics and identify optimal thread counts.

## Implementation Requirements

### Core Features
- OpenMP parallel directives (`#pragma omp parallel for`)
- Thread-safe implementation
- Load balancing strategies
- Build with: `g++ -fopenmp -O2`

### Critical OpenMP Directives
```c
#pragma omp parallel for schedule(dynamic/static) collapse(2)
#pragma omp critical
#pragma omp atomic
```

## Suggested Experiments

### Experiment 1: Strong Scaling Study ⭐ CRITICAL
**Goal**: Measure speedup as thread count increases for fixed problem size

**Parameters**:
- Matrix size: **2048** (fixed)
- Thread counts: 1, 2, 4, 8, 16, 32
- Iterations: 10 per configuration
- Schedule: static

**Metrics**:
- Execution time per thread count
- Speedup = T₁ / Tₙ
- Parallel efficiency = Speedup / N_threads
- GFLOPS per thread count

**Expected Results**:
- Linear speedup up to ~8 threads (depends on cores)
- Efficiency degradation beyond physical cores
- Identify optimal thread count

**Key Performance Win**: Quantify actual parallel speedup (expected 6-8x on 8 cores)

### Experiment 2: Weak Scaling Study
**Goal**: Maintain constant work per processor as thread count increases

**Configuration**:
```
Threads=1:  N=1024  (work per thread = constant)
Threads=2:  N=1448  (√2 × 1024)
Threads=4:  N=2048  (2 × 1024)
Threads=8:  N=2896  (2√2 × 1024)
Threads=16: N=4096  (4 × 1024)
```

**Expected Insight**: Execution time should remain constant if weak scaling is perfect

**Key Performance Win**: Identify if memory bandwidth or cache becomes bottleneck

### Experiment 3: Schedule Strategy Comparison ⭐
**Goal**: Compare OpenMP scheduling strategies

**Strategies to Test**:
1. `schedule(static)` - Divide iterations evenly
2. `schedule(dynamic)` - Dynamic work assignment
3. `schedule(dynamic, chunk_size)` - Test chunks: 1, 16, 64, 256
4. `schedule(guided)` - Decreasing chunk sizes
5. `schedule(auto)` - Compiler decides

**Matrix sizes**: 1024, 2048
**Thread counts**: 8, 16

**Expected Results**:
- Static usually fastest for uniform work
- Dynamic helps with load imbalance (rare here)
- Chunk size impact on overhead

**Key Performance Win**: 10-20% improvement with optimal scheduling

### Experiment 4: Loop Parallelization Strategy
**Goal**: Compare which loop to parallelize

**Variations**:
```c
// Option 1: Parallelize outer loop (i)
#pragma omp parallel for
for (i) for (j) for (k)

// Option 2: Parallelize middle loop (j)
for (i)
  #pragma omp parallel for
  for (j) for (k)

// Option 3: Collapse nested loops
#pragma omp parallel for collapse(2)
for (i) for (j) for (k)
```

**Matrix sizes**: 512, 1024, 2048
**Thread count**: 8

**Expected Insights**:
- Outer loop parallelization usually best (coarse-grained)
- Collapse may help with load balance
- Inner loop parallelization has high overhead

**Key Performance Win**: Understanding granularity vs overhead tradeoff

### Experiment 5: Thread Affinity and Pinning ⭐ CRITICAL
**Goal**: Impact of thread placement on NUMA systems

**Environment Variables**:
```bash
export OMP_PROC_BIND=true/false
export OMP_PLACES=cores/threads/sockets
```

**Configurations**:
1. No binding (default)
2. Bind to cores
3. Bind to threads (with hyperthreading)
4. Spread across sockets

**Matrix size**: 2048
**Thread counts**: 8, 16, 32

**Expected Results**:
- Binding to cores: Better cache locality
- NUMA effects visible on multi-socket systems
- 20-50% improvement with proper binding

**Key Performance Win**: Proper thread affinity can give 30-50% improvement on NUMA systems

### Experiment 6: Cache-Aware Parallelization
**Goal**: Combine blocking with OpenMP

**Implementation**:
```c
#pragma omp parallel for
for (int i_block = 0; i_block < N; i_block += BLOCK_SIZE) {
  for (int j_block = 0; j_block < N; j_block += BLOCK_SIZE) {
    // Blocked multiplication
  }
}
```

**Block sizes**: 32, 64, 128
**Thread counts**: 4, 8, 16

**Expected Results**:
- Better cache utilization than naive OpenMP
- Synergistic effect: parallelism + cache optimization

**Key Performance Win**: 50-100% additional speedup over naive OpenMP

### Experiment 7: False Sharing Analysis
**Goal**: Measure impact of false sharing

**Test Cases**:
```c
// Potential false sharing
double* C = aligned_alloc(64, N*N*sizeof(double));

// With padding to avoid false sharing
struct alignas(64) PaddedDouble { double val; char padding[56]; };
```

**Matrix size**: 1024
**Thread counts**: 4, 8, 16

**Expected Impact**: 10-30% performance loss with false sharing

## Performance Expectations

### Target Performance
- **8 cores**: 6-7x speedup (75-87% efficiency)
- **16 cores**: 10-12x speedup (62-75% efficiency)
- **32 cores**: 15-20x speedup (47-62% efficiency, if available)

### Limitations
- Amdahl's Law: Serial fraction limits speedup
- Memory bandwidth saturation
- Cache contention
- NUMA effects on multi-socket systems

## Key Metrics to Report

1. **Strong Scaling Efficiency** - Most important
2. **Speedup vs Thread Count**
3. **GFLOPS per Thread**
4. **Parallel Overhead** - Time spent in synchronization
5. **Memory Bandwidth Utilization**
6. **Cache Behavior** - Miss rates per thread

## ORCA-Specific Considerations

```bash
# SLURM allocation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32

# Environment
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

## Validation

- Verify results match naive implementation
- Test thread safety (no race conditions)
- Check for deterministic results (may vary with dynamic schedule)

## Implementation Checklist

- [ ] Basic parallel for implementation
- [ ] All scheduling strategies
- [ ] Thread count parameterization
- [ ] Timing per thread count
- [ ] Strong scaling measurements
- [ ] Weak scaling measurements
- [ ] Thread affinity testing
- [ ] CSV output with thread count column

## Expected Timeline

- **Implementation**: 4-6 hours
- **Testing**: 2-3 hours
- **Benchmarking**: 2-3 hours on ORCA
- **Analysis**: 2-3 hours

## Most Important Performance Wins (Ranked)

1. **Basic OpenMP parallelization** - 6-8x speedup on 8 cores (⭐⭐⭐⭐⭐)
2. **Thread affinity/pinning** - Additional 30-50% on NUMA systems (⭐⭐⭐⭐)
3. **Optimal scheduling strategy** - 10-20% improvement (⭐⭐⭐)
4. **Combined with blocking** - 2x additional improvement (⭐⭐⭐⭐)

**Total Potential**: 50-100x speedup over naive (on 32 cores with cache optimization)
