# Quick Start Guide

## Local Development

### 1. Build All Implementations
```bash
cd c/
make all
```

### 2. Test Locally
```bash
# Run naive implementation with 512x512 matrix
./bin/naive_mm 512

# Run OpenMP with 8 threads
export OMP_NUM_THREADS=8
./bin/openmp_mm 1024

# Run MPI with 4 processes
mpirun -np 4 ./bin/mpi_mm 1024
```

## ORCA Cluster Workflow

### 1. Connect and Setup
```bash
# SSH to ORCA
ssh your_username@orca.hpc.edu

# Clone repository
cd $HOME
git clone <repo_url> matrix_multiplication_performance_comparison
cd matrix_multiplication_performance_comparison

# Load modules
module load gcc/11.2.0
module load openmpi/4.1.1
module load cuda/11.7
module load python/3.9
```

### 2. Build on ORCA
```bash
cd c/
make all
```

### 3. Submit Benchmark Jobs
```bash
# Individual submissions
sbatch scripts/slurm/benchmark_naive.sh
sbatch scripts/slurm/benchmark_openmp.sh
sbatch scripts/slurm/benchmark_mpi.sh
sbatch scripts/slurm/benchmark_cuda.sh

# Or submit all at once
./scripts/benchmark/run_all_benchmarks.sh
```

### 4. Monitor Jobs
```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# View output (after job completes)
cat slurm-<job_id>-naive.out

# Check job efficiency
seff <job_id>
```

### 5. Collect Results
```bash
# Results are automatically saved to results/raw/
ls -lh results/raw/*.csv

# Transfer results to local machine
scp your_username@orca.hpc.edu:~/matrix_multiplication_performance_comparison/results/raw/*.csv ./results/raw/
```

## Analysis (Local Machine)

### 1. Setup Python Environment
```bash
cd python/
pip install -r requirements.txt
```

### 2. Run Analysis
```bash
# Generate comparison report
python analysis/compare_implementations.py

# Generate plots
python visualization/plot_speedup.py

# Check results
ls -lh ../results/plots/
ls -lh ../results/reports/
```

### 3. View Results
```bash
# View comparison report
cat ../results/reports/comparison_report.md

# Open plots
# PNG files in results/plots/
# - speedup_comparison.png
# - gflops_comparison.png
# - execution_time.png
```

## Common Commands

### Build Commands
```bash
make all           # Build everything
make naive         # Build only naive
make openmp        # Build only OpenMP
make mpi           # Build only MPI
make cuda          # Build only CUDA
make clean         # Clean build artifacts
make help          # Show all targets
```

### SLURM Commands
```bash
sbatch <script>    # Submit job
squeue -u $USER    # View your jobs
scancel <job_id>   # Cancel job
scontrol show job <job_id>  # Job details
sacct              # Job accounting
seff <job_id>      # Job efficiency
```

## Troubleshooting

### Build Issues
```bash
# Check compiler
gcc --version
g++ --version

# Check modules
module list

# Clean and rebuild
make clean && make all
```

### SLURM Issues
```bash
# Check partition availability
sinfo

# Check your limits
sshare -U

# View error logs
cat slurm-<job_id>-*.err
```

### CUDA Issues
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA
nvcc --version

# Check GPU in SLURM
sinfo -p gpu
```

## Tips

- **Start small**: Test with small matrix sizes (64, 128) first
- **Check correctness**: Verify outputs match before benchmarking
- **Monitor resources**: Use `seff` to check if you're using requested resources efficiently
- **Off-peak hours**: Run large jobs during off-peak times for faster turnaround
- **Save work**: Commit code regularly to git
- **Document**: Update docs/results_summary.md with findings

## File Naming Convention

- **Implementations**: `<method>_mm` (e.g., `naive_mm`, `openmp_mm`)
- **SLURM scripts**: `benchmark_<method>.sh`
- **Results**: `<method>_YYYYMMDD_HHMMSS.csv`
- **Plots**: `<metric>_comparison.png`

## Next Steps

1. ✅ Setup complete - folders and files created
2. ⏳ Implement matrix multiplication variants in `c/src/`
3. ⏳ Build and test locally
4. ⏳ Deploy to ORCA and run benchmarks
5. ⏳ Analyze results with Python scripts
6. ⏳ Document findings in `docs/results_summary.md`

---

For detailed information, see:
- **README.md** - Full project documentation
- **docs/setup.md** - Detailed ORCA setup
- **docs/experiments.md** - Experimental methodology
- **PROJECT_STATUS.md** - Current project status
