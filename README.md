# Matrix Multiplication Performance Comparison - HPC Benchmarking

## Project Overview

This project uses the **ORCA cluster** to experiment with various High-Performance Computing (HPC) optimizations using matrix multiplication as a benchmark. The goal is to systematically evaluate and compare different optimization techniques to understand their impact on computational performance.

## Optimization Techniques

We will experiment with the following optimization approaches:

- **CUDA** - GPU acceleration using NVIDIA CUDA
- **MPI** - Message Passing Interface for distributed computing
- **OpenMP** - Shared-memory parallel programming
- **C++ Compiler Optimizations** - Various compiler flags and optimization levels (-O1, -O2, -O3, -Ofast)
- **Data Structures** - Different matrix storage formats (row-major, column-major, blocked)
- **Algorithms** - Naive, blocked/tiled, Strassen's algorithm, cache-oblivious
- **Cache Optimizations** - Loop tiling, cache blocking, prefetching

## Project Structure

```
matrix_multiplication_performance_comparison/
├── c/                          # C/C++ implementations
│   ├── src/                    # Source files
│   │   ├── naive/             # Naive implementation
│   │   ├── openmp/            # OpenMP parallelization
│   │   ├── mpi/               # MPI distributed implementation
│   │   ├── cuda/              # CUDA GPU implementation
│   │   ├── cache_opt/         # Cache optimization techniques
│   │   ├── compiler_opt/      # Compiler optimization experiments
│   │   └── algorithms/        # Advanced algorithms (Strassen, etc.)
│   ├── include/               # Header files
│   ├── benchmarks/            # Benchmark runner scripts
│   ├── Makefile               # Build configuration
│   └── README.md              # C++ specific documentation
│
├── python/                     # Python analysis and visualization
│   ├── analysis/              # Data analysis scripts
│   ├── visualization/         # Plotting and graphing
│   ├── data_processing/       # CSV processing utilities
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Python specific documentation
│
├── results/                    # Benchmark results
│   ├── raw/                   # Raw CSV data
│   ├── processed/             # Processed data
│   ├── plots/                 # Generated graphs and figures
│   └── reports/               # Summary reports
│
├── scripts/                    # Utility scripts
│   ├── slurm/                 # SLURM job submission scripts
│   ├── build/                 # Build automation
│   └── benchmark/             # Automated benchmark runners
│
├── docs/                       # Documentation
│   ├── setup.md               # ORCA cluster setup guide
│   ├── experiments.md         # Experiment methodology
│   └── results_summary.md     # Results and findings
│
├── .gitignore                  # Git ignore rules
├── LICENSE                     # Project license
└── README.md                   # This file
```

## Experimental Workflow

1. **Implementation Phase**
   - Develop matrix multiplication implementations for each optimization technique
   - Ensure correctness with unit tests
   - Document implementation details

2. **Benchmarking Phase**
   - Run experiments on ORCA cluster with varying matrix sizes
   - Test different thread counts, process counts, and GPU configurations
   - Collect timing data, memory usage, and other metrics
   - Export results to CSV format

3. **Analysis Phase**
   - Load and process CSV data using Python
   - Calculate speedup, efficiency, and scalability metrics
   - Generate comparative visualizations
   - Identify performance bottlenecks and optimization opportunities

4. **Documentation Phase**
   - Document findings and insights
   - Create performance comparison reports
   - Generate publication-ready figures

## Getting Started

### Prerequisites

- Access to ORCA cluster
- C/C++ compiler with C++11 support (gcc/g++)
- CUDA toolkit (for GPU implementations)
- MPI library (OpenMPI or MPICH)
- Python 3.8+ with pandas, matplotlib, seaborn, numpy
- SLURM workload manager knowledge

### Building the Project

```bash
cd c/
make all                    # Build all implementations
make cuda                   # Build CUDA version
make mpi                    # Build MPI version
make openmp                 # Build OpenMP version
```

### Running Benchmarks

```bash
# Local testing
./c/bin/naive_mm 1024

# Submit to ORCA cluster
sbatch scripts/slurm/benchmark_all.sh
```

### Analyzing Results

```bash
cd python/
pip install -r requirements.txt
python analysis/compare_implementations.py
python visualization/plot_speedup.py
```

## Benchmark Metrics

- **Execution Time** - Wall-clock time for matrix multiplication
- **GFLOPS** - Giga floating-point operations per second
- **Speedup** - Performance relative to baseline (naive implementation)
- **Efficiency** - Speedup / number of processors
- **Scalability** - Performance vs. problem size and resource count
- **Memory Usage** - Peak memory consumption
- **Cache Performance** - Cache hit/miss rates (where available)

## Matrix Sizes for Testing

We will test with the following square matrix dimensions:
- Small: 64×64, 128×128, 256×256
- Medium: 512×512, 1024×1024, 2048×2048
- Large: 4096×4096, 8192×8192

## Contributing

When adding new implementations or experiments:
1. Follow the existing code structure
2. Add appropriate documentation
3. Include benchmark scripts for SLURM
4. Update this README with new findings

## ORCA Cluster Specifics

- Use SLURM for job submission
- Request appropriate resources (CPUs, GPUs, memory)
- Be mindful of cluster usage policies
- Store large datasets in appropriate scratch space

## License

See LICENSE file for details.

## References

- ORCA Documentation: [Link to ORCA cluster documentation]
- Project Report: See `Matrix_Multiplication_Performance_Comparison_ECE_472_Project.pdf`

## Contact

For questions or collaboration, please open an issue or contact the project maintainer.