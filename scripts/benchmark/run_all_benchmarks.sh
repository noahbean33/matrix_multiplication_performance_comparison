#!/bin/bash
# Submit all benchmark jobs to SLURM

echo "=== Submitting Matrix Multiplication Benchmarks ==="
echo ""

SCRIPT_DIR="$(dirname "$0")/../slurm"

# Array to store job IDs
declare -a JOB_IDS

# Submit naive benchmark
echo "Submitting naive benchmark..."
JOB_ID=$(sbatch "$SCRIPT_DIR/benchmark_naive.sh" | awk '{print $4}')
JOB_IDS+=($JOB_ID)
echo "  Job ID: $JOB_ID"

# Submit OpenMP benchmark
echo "Submitting OpenMP benchmark..."
JOB_ID=$(sbatch "$SCRIPT_DIR/benchmark_openmp.sh" | awk '{print $4}')
JOB_IDS+=($JOB_ID)
echo "  Job ID: $JOB_ID"

# Submit MPI benchmark
echo "Submitting MPI benchmark..."
JOB_ID=$(sbatch "$SCRIPT_DIR/benchmark_mpi.sh" | awk '{print $4}')
JOB_IDS+=($JOB_ID)
echo "  Job ID: $JOB_ID"

# Submit CUDA benchmark (if GPU partition available)
echo "Submitting CUDA benchmark..."
JOB_ID=$(sbatch "$SCRIPT_DIR/benchmark_cuda.sh" | awk '{print $4}')
JOB_IDS+=($JOB_ID)
echo "  Job ID: $JOB_ID"

echo ""
echo "=== All Jobs Submitted ==="
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check status: squeue -j ${JOB_IDS[0]},${JOB_IDS[1]},${JOB_IDS[2]},${JOB_IDS[3]}"
