#!/bin/bash
#SBATCH --job-name=mm_cuda
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --output=slurm-%j-cuda.out
#SBATCH --error=slurm-%j-cuda.err

# Matrix Multiplication CUDA Benchmark
# This script runs the CUDA GPU implementation

echo "=== Matrix Multiplication CUDA Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Load required modules
module load gcc/11.2.0
module load cuda/11.7
module list

# Display GPU info
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create results directory
RESULTS_DIR="results/raw"
mkdir -p $RESULTS_DIR

# Output file
OUTPUT_FILE="$RESULTS_DIR/cuda_$(date +%Y%m%d_%H%M%S).csv"

# Write CSV header
echo "timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node,gpu" > $OUTPUT_FILE

# Matrix sizes to test
SIZES=(64 128 256 512 1024 2048 4096 8192)

# Number of iterations per size
ITERATIONS=10

echo "Running CUDA benchmarks..."
echo ""

# Loop through matrix sizes
for SIZE in "${SIZES[@]}"; do
    echo "Testing matrix size: ${SIZE}x${SIZE}"
    
    for i in $(seq 1 $ITERATIONS); do
        echo "  Iteration $i/$ITERATIONS"
        
        # Run the CUDA implementation
        ./c/bin/cuda_mm $SIZE >> $OUTPUT_FILE
        
        if [ $? -ne 0 ]; then
            echo "Error running cuda_mm for size $SIZE"
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

# Print GPU final state
echo "=== GPU Final State ==="
nvidia-smi

# Print summary
echo "=== Summary ==="
TOTAL_RUNS=$(( ${#SIZES[@]} * $ITERATIONS ))
echo "Total runs: $TOTAL_RUNS"
echo "Output file size: $(du -h $OUTPUT_FILE | cut -f1)"
