# Quick Start Guide

## Get Started in 3 Commands

```bash
# 1. Build all implementations
./scripts/build.sh

# 2. Run benchmarks
./scripts/run_benchmarks.sh

# 3. Generate plots
python scripts/plot_results.py
```

## What Each Implementation Does

| Implementation | Description | File |
|---------------|-------------|------|
| **Baseline** | Simple C, no optimizations | `src/baseline/matrix_mult.c` |
| **Optimized** | Same code with compiler flags | `src/optimized/matrix_mult.c` |
| **OpenMP** | Multi-threaded parallelism | `src/openmp/matrix_mult_omp.c` |
| **MPI** | Distributed across processes | `src/mpi/matrix_mult_mpi.c` |
| **CUDA** | GPU acceleration | `src/cuda/matrix_multiplication.cu` |

## Prerequisites

```bash
# On Linux/macOS
sudo apt-get install gcc g++ openmpi-bin libopenmpi-dev  # Ubuntu/Debian
# or
brew install gcc open-mpi  # macOS

# CUDA (for GPU benchmarks)
# Install from: https://developer.nvidia.com/cuda-downloads

# Python packages
pip install pandas matplotlib numpy seaborn
```

## Test Individual Implementations

### Baseline
```bash
gcc -O0 src/baseline/matrix_mult.c -o bin/baseline
./bin/baseline 1024
```

### OpenMP
```bash
gcc -O3 -fopenmp src/openmp/matrix_mult_omp.c -o bin/openmp
OMP_NUM_THREADS=4 ./bin/openmp 1024
```

### MPI
```bash
mpicc -O3 src/mpi/matrix_mult_mpi.c -o bin/mpi
mpirun -np 4 ./bin/mpi 1024
```

### CUDA
```bash
cd src/cuda
make
./matrix_multiplication
```

## Understanding the Output

All implementations output CSV format:
```
timestamp,implementation,matrix_size,time_ms,gflops,kernel_time_ms,h2d_time_ms,d2h_time_ms,block_size,node,verification
```

Example:
```
2025-11-29 20:30:45,baseline,1024,1234.56,1.75,1234.56,0.000,0.000,N/A,mycomputer,N/A
```

## Plotting Results

The plotting script automatically:
- Compares execution times
- Shows GFLOPS across implementations
- Calculates speedup vs baseline
- Shows scaling for OpenMP/MPI

Results saved to: `results/plots/<timestamp>/`

## Matrix Sizes

Default sizes tested:
- 512×512 (Small)
- 1024×1024 (Medium)
- 2048×2048 (Large)
- 4096×4096 (Very Large)

Adjust in `scripts/run_benchmarks.sh` by modifying:
```bash
SIZES=(512 1024 2048 4096)
```

## Common Issues

### Build fails with "command not found"
Install the missing compiler:
- `gcc` - Install build-essential or developer tools
- `nvcc` - Install CUDA toolkit
- `mpicc` - Install OpenMPI or MPICH

### OpenMP benchmark uses only 1 thread
Set threads explicitly:
```bash
export OMP_NUM_THREADS=8
./bin/openmp 1024
```

### MPI fails to run
Check MPI installation:
```bash
mpirun --version
```

### Plotting fails
Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Next Steps

1. **Run benchmarks** on your system
2. **Compare results** using the generated plots
3. **Modify code** to try new optimizations
4. **Update README** with your findings

## Project Structure

```
.
├── src/                    # Source code
│   ├── baseline/          # C reference
│   ├── cuda/              # GPU acceleration
│   ├── mpi/               # Distributed
│   ├── openmp/            # Multi-threaded
│   └── optimized/         # Compiler opts
├── scripts/               # Build & benchmark scripts
│   ├── build.sh
│   ├── run_benchmarks.sh
│   └── plot_results.py
├── results/               # Benchmark data
│   ├── raw/              # CSV files
│   └── plots/            # Graphs
└── README.md             # Full documentation
```

## Help

- Full documentation: `README.md`
- Migration guide: `MIGRATION_GUIDE.md`
- For issues: Check compiler/library versions first
