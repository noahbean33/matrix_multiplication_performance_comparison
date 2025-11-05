# C/C++ Implementation Directory

## Building the Project

### Prerequisites
- GCC/G++ compiler with C++11 support
- OpenMP support (usually included with GCC)
- MPI implementation (OpenMPI or MPICH)
- CUDA toolkit (for GPU implementations)

### Compilation

```bash
# Build all implementations
make all

# Build specific versions
make naive
make openmp
make mpi
make cuda
make cache_opt
```

## Implementation Details

### Naive Implementation (`src/naive/`)
Basic triple-nested loop matrix multiplication without optimization.

### OpenMP Implementation (`src/openmp/`)
Parallel implementation using OpenMP directives for multi-core CPUs.

### MPI Implementation (`src/mpi/`)
Distributed implementation using Message Passing Interface for cluster computing.

### CUDA Implementation (`src/cuda/`)
GPU-accelerated implementation using NVIDIA CUDA.

### Cache Optimizations (`src/cache_opt/`)
Various cache-aware optimizations:
- Loop tiling/blocking
- Cache-conscious data access patterns
- Prefetching techniques

### Compiler Optimizations (`src/compiler_opt/`)
Testing different compiler optimization flags:
- `-O1`, `-O2`, `-O3`, `-Ofast`
- Architecture-specific optimizations
- Link-time optimization (LTO)

### Advanced Algorithms (`src/algorithms/`)
Alternative multiplication algorithms:
- Strassen's algorithm
- Cache-oblivious algorithms
- Blocked matrix multiplication

## Benchmarking

All implementations output CSV data with the following format:
```
implementation,matrix_size,execution_time_ms,gflops,threads,processes
```

## Testing

Each implementation includes correctness verification against the naive implementation.
