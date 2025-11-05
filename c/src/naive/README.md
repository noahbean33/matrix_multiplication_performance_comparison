# Naive Matrix Multiplication - Baseline Implementation

## Purpose

Establish baseline performance metrics for matrix multiplication without any optimizations. This serves as the reference point for measuring speedup of all optimized implementations.

## Implementation Requirements

### Core Algorithm
- Triple nested loop (i, j, k)
- Standard row-major C array storage: `A[i][j]` or `A[i*N + j]`
- Double precision floating point (`double`)
- No compiler optimizations beyond `-O2` for fair baseline

### Output Format
CSV with columns:
```
timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node
```

## Suggested Experiments

### Experiment 1: Baseline Performance Characterization
**Goal**: Establish baseline GFLOPS across matrix sizes

**Parameters**:
- Matrix sizes: 64, 128, 256, 512, 1024, 2048, 4096
- Iterations: 10 per size
- Single thread execution

**Metrics to Track**:
- Execution time (ms)
- GFLOPS = (2 × N³) / (time_in_seconds × 10⁹)
- Memory footprint

**Expected Insights**:
- Identify where performance degrades (cache limits)
- Establish baseline GFLOPS for comparison
- Understand memory bandwidth limits

### Experiment 2: Loop Order Impact
**Goal**: Quantify performance impact of loop ordering

**Variations to Test**:
```c
// ijk order (standard)
for (i) for (j) for (k) C[i][j] += A[i][k] * B[k][j];

// ikj order (better cache locality for C)
for (i) for (k) for (j) C[i][j] += A[i][k] * B[k][j];

// jik order
for (j) for (i) for (k) C[i][j] += A[i][k] * B[k][j];

// jki order
for (j) for (k) for (i) C[i][j] += A[i][k] * B[k][j];

// kij order
for (k) for (i) for (j) C[i][j] += A[i][k] * B[k][j];

// kji order
for (k) for (j) for (i) C[i][j] += A[i][k] * B[k][j];
```

**Matrix sizes**: 512, 1024, 2048

**Expected Performance Ranking**:
1. ikj or jki (best cache reuse)
2. ijk (standard)
3. Others (poor cache behavior)

**Key Performance Win**: Understanding cache locality impact (potential 2-3x difference)

### Experiment 3: Data Type Impact
**Goal**: Compare single vs double precision performance

**Variations**:
- `float` (single precision)
- `double` (double precision)

**Matrix sizes**: 512, 1024, 2048, 4096

**Expected Insights**:
- Memory bandwidth vs compute balance
- Cache capacity effects
- ~2x GFLOPS improvement with float expected

**Performance Win**: Determine if precision reduction is viable

### Experiment 4: Scaling Behavior
**Goal**: Understand computational complexity scaling

**Matrix sizes**: 64, 128, 256, 512, 1024, 2048, 4096, 8192

**Analysis**:
- Plot execution time vs N³
- Verify O(N³) complexity
- Identify cache hierarchy transitions

**Expected Insights**:
- L1/L2/L3 cache size boundaries
- Memory bandwidth saturation point
- Polynomial fit verification

## Performance Expectations

### Typical Baseline Performance
- Small matrices (N < 256): 0.5-2 GFLOPS (L1 cache)
- Medium matrices (256 < N < 2048): 1-5 GFLOPS (L2/L3 cache)
- Large matrices (N > 2048): 2-8 GFLOPS (memory bound)

### Performance Limiters
1. **Memory bandwidth** - Primary bottleneck for large N
2. **Cache misses** - Dominant for poor loop orders
3. **No vectorization** - Missing SIMD opportunities
4. **No parallelism** - Single core utilization

## Key Metrics to Report

1. **Peak GFLOPS** - Best performance achieved
2. **Sustained GFLOPS** - Average across runs
3. **Efficiency** - Percentage of theoretical peak
4. **Memory Bandwidth Utilization** - GB/s achieved
5. **Cache Miss Rate** - If profiling tools available

## Validation

Before benchmarking, verify correctness:
```c
// Compare with known result
// Small matrix (easy to verify manually)
// Check numerical stability (relative error < 1e-10)
```

## Implementation Checklist

- [ ] Basic triple-loop implementation (ijk order)
- [ ] All 6 loop orderings for comparison
- [ ] CSV output in standard format
- [ ] High-resolution timing (microsecond precision)
- [ ] Correctness verification
- [ ] Memory allocation and initialization
- [ ] GFLOPS calculation
- [ ] Error handling

## Expected Timeline

- **Implementation**: 2-4 hours
- **Testing**: 1-2 hours
- **Benchmarking**: 30 minutes (local), 1-2 hours (ORCA)
- **Analysis**: 1 hour

## Most Important Performance Win

**Loop ordering optimization** - Can provide 2-5x speedup with zero algorithmic change, purely from better cache utilization. This is the foundation for understanding why cache optimizations matter.
