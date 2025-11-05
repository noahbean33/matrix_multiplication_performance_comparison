# Experiment Methodology

## Overview

This document outlines the experimental methodology for benchmarking matrix multiplication implementations on the ORCA cluster.

## Experimental Design

### Independent Variables

1. **Implementation Type**
   - Naive (baseline)
   - OpenMP
   - MPI
   - CUDA
   - Cache-optimized
   - Compiler-optimized
   - Advanced algorithms

2. **Matrix Size**
   - Small: 64, 128, 256
   - Medium: 512, 1024, 2048
   - Large: 4096, 8192

3. **Resource Configuration**
   - Thread count (OpenMP): 1, 2, 4, 8, 16, 32
   - Process count (MPI): 1, 2, 4, 8, 16, 32, 64
   - GPU configuration (CUDA): Block/grid dimensions

4. **Compiler Optimization Level**
   - O0, O1, O2, O3, Ofast

### Dependent Variables (Metrics)

1. **Execution Time** (milliseconds)
2. **GFLOPS** - Calculated as: `(2 * N^3) / (time_in_seconds * 10^9)`
3. **Speedup** - Relative to naive baseline
4. **Efficiency** - Speedup / number of processors
5. **Memory Usage** (MB)
6. **Cache Performance** (if tools available)

## Experimental Procedure

### Phase 1: Baseline Establishment

1. Run naive implementation for all matrix sizes
2. Verify correctness
3. Establish baseline performance metrics
4. Run multiple iterations (n=10) for statistical validity

### Phase 2: Single-Variable Testing

Test each optimization independently:

#### OpenMP Testing
- Fixed matrix size (e.g., 2048)
- Vary thread count: 1, 2, 4, 8, 16, 32
- Measure strong scaling

#### MPI Testing
- Fixed matrix size (e.g., 2048)
- Vary process count: 1, 2, 4, 8, 16, 32, 64
- Test on multiple nodes
- Measure communication overhead

#### CUDA Testing
- Test various matrix sizes
- Optimize block/grid dimensions
- Compare with CPU implementations

#### Cache Optimization Testing
- Test different block sizes: 16, 32, 64, 128
- Measure cache hit rates
- Compare loop orderings (ijk, ikj, jik, jki, kij, kji)

### Phase 3: Scalability Studies

#### Strong Scaling
- **Definition**: Fixed problem size, increase resources
- **Test**: Matrix size = 4096, vary processors
- **Expected**: Linear speedup in ideal case

#### Weak Scaling
- **Definition**: Problem size grows with resources
- **Test**: Matrix size per processor constant
- **Expected**: Constant execution time

### Phase 4: Cross-Variable Testing

Combine optimizations:
- OpenMP + Cache optimization
- MPI + Compiler optimization
- Compare combined effects

## Data Collection

### CSV Output Format

Each benchmark run outputs:
```csv
timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node_count,optimization_level,block_size,additional_notes
```

### Run Configuration

- **Iterations per configuration**: 10
- **Warmup runs**: 2 (discarded)
- **Time measurement**: High-resolution timer (microsecond precision)
- **Memory measurement**: Peak RSS

## Statistical Analysis

### Measures to Calculate

1. **Mean execution time**
2. **Standard deviation**
3. **95% Confidence intervals**
4. **Coefficient of variation**

### Outlier Detection

- Use Tukey's method (IQR)
- Remove runs with >3σ deviation (review for system issues)

### Significance Testing

- Use t-tests to compare implementations
- ANOVA for multi-way comparisons
- Report p-values and effect sizes

## Correctness Verification

Before benchmarking, verify:
1. Output matches naive implementation
2. Numerical stability (check relative error)
3. Edge cases (small matrices, non-square)

### Verification Script

```bash
./c/bin/naive_mm 64 > naive_out.txt
./c/bin/openmp_mm 64 > openmp_out.txt
diff naive_out.txt openmp_out.txt
```

## SLURM Job Configuration

### Resource Requests

```bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
```

Adjust based on:
- Expected runtime
- Memory requirements (≈3 × N² × 8 bytes for double precision)
- Node availability

## Reproducibility

### Document Everything

1. Hardware specifications
2. Software versions (gcc, CUDA, MPI)
3. SLURM job scripts
4. Random seeds (if applicable)
5. Date and time of runs
6. Node names

### Version Control

- Tag releases: `v1.0-experiments`
- Commit job scripts with results
- Include environment dump: `module list > environment.txt`

## Safety Checks

1. **Time limits**: Set conservative estimates
2. **Memory limits**: Calculate expected usage + 20% buffer
3. **Disk usage**: Monitor output file sizes
4. **Parallel I/O**: Avoid writing to same file

## Expected Results

### Hypotheses

1. OpenMP will show good scaling up to core count
2. MPI will have communication overhead
3. CUDA will excel for large matrices
4. Cache optimization will improve single-thread performance
5. Compiler optimization will provide 2-5x speedup

### Potential Issues

- **NUMA effects**: Pin threads/processes
- **Network contention**: Run during off-peak hours
- **Thermal throttling**: Use monitoring tools
- **Turbo boost**: Disable for consistency

## Timeline

1. **Week 1-2**: Implementation and verification
2. **Week 3-4**: Baseline and single-variable experiments
3. **Week 5**: Scalability studies
4. **Week 6**: Cross-variable and final experiments
5. **Week 7-8**: Analysis and visualization
