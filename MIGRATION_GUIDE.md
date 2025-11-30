# Migration Guide - Project Simplification

## What Changed

The project has been simplified to focus on core parallel computing technologies:
- **CUDA** - GPU acceleration
- **MPI** - Distributed computing
- **OpenMP** - Shared memory parallelism
- **Compiler Optimizations** - Testing different optimization levels
- **Baseline C** - Reference implementation

## Removed/Archived

The following directories are no longer part of the main workflow:
- `src/algorithms/` - Advanced algorithms (Strassen, etc.)
- `src/cache_opt/` - Cache-specific optimizations
- `docs/` - Extensive documentation (now in README)
- Old analysis scripts in `analysis/` directories
- Complex SLURM scripts in `scripts/slurm/`
- `orca-website/` - Website components

## New Structure

```
Simplified:
src/
├── baseline/          # NEW: Basic C reference
├── cuda/             # KEPT: Existing CUDA implementation
├── mpi/              # NEW: Simple MPI implementation
├── openmp/           # NEW: Simple OpenMP implementation
└── optimized/        # NEW: Compiler optimization tests

scripts/
├── build.sh          # NEW: Single build script for everything
├── run_benchmarks.sh # NEW: Single script to run all benchmarks
└── plot_results.py   # NEW: Unified plotting script

results/
├── raw/              # KEPT: Benchmark CSV outputs
└── plots/            # KEPT: Generated graphs
```

## What to Keep

### Working Implementations
- **CUDA**: `src/cuda/matrix_multiplication.cu` and `src/cuda/Makefile` are ready to use
- Everything else in `src/` directories can be ignored or removed

### Results
- Keep `results/raw/` and `results/plots/` for benchmark data

## Migration Steps

### 1. Use New Build System
Old:
```bash
cd src/cuda && make
cd src/openmp && make
cd src/mpi && make
```

New:
```bash
./scripts/build.sh  # Builds everything
```

### 2. Use New Benchmark Runner
Old:
```bash
sbatch scripts/slurm/benchmark_cuda.sh
sbatch scripts/slurm/benchmark_openmp.sh
sbatch scripts/slurm/benchmark_mpi.sh
```

New:
```bash
./scripts/run_benchmarks.sh  # Runs all benchmarks locally
# Results saved to results/raw/<timestamp>/
```

### 3. Use New Plotting Script
Old:
```bash
python analysis/compare_implementations.py
python analysis/visualization/plot_speedup.py
```

New:
```bash
python scripts/plot_results.py
# Generates all plots automatically
```

## Key Benefits

1. **Simpler Build**: One script builds everything
2. **Unified Output**: All implementations use the same CSV format
3. **Easier Analysis**: Single Python script generates all plots
4. **Less Code**: Focused on essential implementations
5. **Clear Structure**: Obvious where each implementation lives

## What If I Need Old Features?

The old code is still in the repository. You can:
1. Check out an older commit
2. Reference old files in `src/algorithms/`, `src/cache_opt/`, etc.
3. Look at `PROJECT_STATUS.md` for the old structure

## Quick Start After Migration

```bash
# Build everything
./scripts/build.sh

# Run benchmarks
./scripts/run_benchmarks.sh

# Generate plots
python scripts/plot_results.py

# View results
ls results/plots/<timestamp>/
```

## Testing the Migration

To verify everything works:

```bash
# 1. Build baseline
gcc -O0 src/baseline/matrix_mult.c -o bin/baseline

# 2. Test it
./bin/baseline 512

# 3. Should output CSV line like:
# 2025-11-29 20:30:45,baseline,512,234.567,0.234,234.567,0.000,0.000,N/A,hostname,N/A
```

## Python Dependencies

Install with:
```bash
pip install pandas matplotlib seaborn numpy
```

Or use the old `analysis/requirements.txt` if it exists.
