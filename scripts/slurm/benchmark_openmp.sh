#!/bin/bash
#SBATCH --job-name=mm_openmp
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --output=slurm-%j-openmp.out
#SBATCH --error=slurm-%j-openmp.err

# Matrix Multiplication OpenMP Benchmark
# This script runs the OpenMP implementation with varying thread counts

echo "=== Matrix Multiplication OpenMP Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo ""

# Load required modules
module load gcc/11.2.0
module list

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create results directory
RESULTS_DIR="results/raw"
mkdir -p $RESULTS_DIR

# Output file
OUTPUT_FILE="$RESULTS_DIR/openmp_$(date +%Y%m%d_%H%M%S).csv"

# Write CSV header
echo "timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node" > $OUTPUT_FILE

# Matrix sizes to test
SIZES=(512 1024 2048 4096)

# Thread counts to test
THREADS=(1 2 4 8 16 32)

# Number of iterations per configuration
ITERATIONS=10

echo "Running OpenMP benchmarks..."
echo ""

# Loop through matrix sizes
for SIZE in "${SIZES[@]}"; do
    echo "Testing matrix size: ${SIZE}x${SIZE}"
    
    # Loop through thread counts
    for NTHREADS in "${THREADS[@]}"; do
        echo "  Thread count: $NTHREADS"
        
        # Set OpenMP thread count
        export OMP_NUM_THREADS=$NTHREADS
        
        for i in $(seq 1 $ITERATIONS); do
            # Run the OpenMP implementation
            ./c/bin/openmp_mm $SIZE >> $OUTPUT_FILE
            
            if [ $? -ne 0 ]; then
                echo "Error running openmp_mm for size $SIZE with $NTHREADS threads"
                exit 1
            fi
        done
        
        echo "    Completed $NTHREADS threads"
    done
    
    echo "  Completed ${SIZE}x${SIZE}"
    echo ""
done

echo "=== Benchmark Complete ==="
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Print summary
echo "=== Summary ==="
TOTAL_RUNS=$(( ${#SIZES[@]} * ${#THREADS[@]} * $ITERATIONS ))
echo "Total runs: $TOTAL_RUNS"
echo "Output file size: $(du -h $OUTPUT_FILE | cut -f1)"
