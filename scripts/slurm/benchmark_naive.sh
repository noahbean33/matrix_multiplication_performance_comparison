#!/bin/bash
#SBATCH --job-name=mm_naive
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --output=slurm-%j-naive.out
#SBATCH --error=slurm-%j-naive.err

# Matrix Multiplication Naive Benchmark
# This script runs the naive implementation for various matrix sizes

echo "=== Matrix Multiplication Naive Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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
OUTPUT_FILE="$RESULTS_DIR/naive_$(date +%Y%m%d_%H%M%S).csv"

# Write CSV header
echo "timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node" > $OUTPUT_FILE

# Matrix sizes to test
SIZES=(64 128 256 512 1024 2048 4096)

# Number of iterations per size
ITERATIONS=10

echo "Running benchmarks..."
echo ""

# Loop through matrix sizes
for SIZE in "${SIZES[@]}"; do
    echo "Testing matrix size: ${SIZE}x${SIZE}"
    
    for i in $(seq 1 $ITERATIONS); do
        echo "  Iteration $i/$ITERATIONS"
        
        # Run the naive implementation
        ./c/bin/naive_mm $SIZE >> $OUTPUT_FILE
        
        if [ $? -ne 0 ]; then
            echo "Error running naive_mm for size $SIZE"
            exit 1
        fi
    done
    
    echo "  Completed ${SIZE}x${SIZE}"
    echo ""
done

echo "=== Benchmark Complete ==="
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Print summary statistics
echo "=== Summary ==="
echo "Total runs: $(( ${#SIZES[@]} * $ITERATIONS ))"
echo "Output file size: $(du -h $OUTPUT_FILE | cut -f1)"
