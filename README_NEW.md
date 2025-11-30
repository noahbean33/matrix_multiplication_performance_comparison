# Matrix Multiplication Performance Comparison

A focused HPC benchmarking project comparing matrix multiplication performance across different parallel computing paradigms.

## Overview

This project benchmarks matrix multiplication using:
- **Baseline C** - Simple, unoptimized reference implementation
- **CUDA** - GPU acceleration with NVIDIA CUDA
- **MPI** - Distributed computing across multiple nodes
- **OpenMP** - Multi-threaded shared memory parallelism
- **Compiler Optimizations** - Testing -O1, -O2, -O3, -Ofast flags

## Quick Start

### Prerequisites
- GCC/G++ compiler
- NVIDIA CUDA Toolkit (for CUDA)
- MPI implementation (OpenMPI/MPICH)
- Python 3.8+ with pandas, matplotlib, numpy

### Build All Implementations
```bash
./scripts/build.sh
```

### Run Benchmarks
```bash
./scripts/run_benchmarks.sh
```

### Plot Results
```bash
python scripts/plot_results.py
```

## Project Structure

```
src/
├── baseline/              # Basic C implementation
│   └── matrix_mult.c
├── cuda/                  # CUDA implementation
│   └── matrix_multiplication.cu
├── mpi/                   # MPI implementation
│   └── matrix_mult_mpi.c
├── openmp/               # OpenMP implementation
│   └── matrix_mult_omp.c
└── optimized/            # Compiler optimization tests
    └── matrix_mult.c

scripts/
├── build.sh              # Build all implementations
├── run_benchmarks.sh     # Run all benchmarks
└── plot_results.py       # Generate performance plots

results/
├── raw/                  # CSV output from benchmarks
└── plots/                # Generated performance graphs
```

## Implementation Details

### Baseline C
Simple triple-loop matrix multiplication with no optimizations. Used as reference for speedup calculations.

### CUDA
- Naive global memory kernel
- Tiled shared memory kernel (16x16, 32x32 blocks)
- Performance metrics: kernel time, transfer time, GFLOPS

### MPI
- Distributed row-based decomposition
- Each process handles a subset of rows
- Collective communication for result gathering

### OpenMP
- Parallel loops with different scheduling strategies
- Thread scaling tests (1, 2, 4, 8, 16 threads)
- Shared memory optimization

### Compiler Optimizations
Tests same baseline code with:
- `-O0` - No optimization
- `-O1` - Basic optimization
- `-O2` - Moderate optimization
- `-O3` - Aggressive optimization
- `-Ofast` - Maximum optimization (may affect accuracy)

## Benchmark Parameters

### Matrix Sizes
- Small: 512×512, 1024×1024
- Medium: 2048×2048, 4096×4096
- Large: 8192×8192 (if memory permits)

### Output Format
All implementations output CSV with:
```
timestamp,implementation,matrix_size,time_ms,gflops,threads/processes,node
```

## Analysis

The Python plotting script generates:
- Execution time comparison across implementations
- GFLOPS comparison
- Speedup relative to baseline
- Scaling efficiency (for MPI and OpenMP)

## Usage Examples

### Build specific implementation
```bash
# Baseline
gcc -O0 src/baseline/matrix_mult.c -o bin/baseline

# CUDA
cd src/cuda && make

# OpenMP
gcc -O3 -fopenmp src/openmp/matrix_mult_omp.c -o bin/openmp

# MPI
mpicc -O3 src/mpi/matrix_mult_mpi.c -o bin/mpi
```

### Run individual benchmarks
```bash
# Baseline
./bin/baseline 1024 > results/raw/baseline.csv

# CUDA
./src/cuda/matrix_multiplication > results/raw/cuda.csv

# OpenMP (8 threads)
OMP_NUM_THREADS=8 ./bin/openmp 1024 > results/raw/openmp_8.csv

# MPI (4 processes)
mpirun -np 4 ./bin/mpi 1024 > results/raw/mpi_4.csv
```

## Performance Metrics

- **Execution Time**: Wall-clock time in milliseconds
- **GFLOPS**: Billion floating-point operations per second
  - Formula: (2 × N³) / (time_seconds × 10⁹)
- **Speedup**: Time(baseline) / Time(optimized)
- **Efficiency**: Speedup / Number of processors

## Contributing

When adding new optimizations:
1. Follow the existing code structure
2. Output results in standard CSV format
3. Include verification against baseline
4. Update this README

## License

See LICENSE file for details.
