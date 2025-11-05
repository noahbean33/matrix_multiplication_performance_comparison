#!/bin/bash
#SBATCH --job-name=mm_mpi
#SBATCH --partition=short
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --output=slurm-%j-mpi.out
#SBATCH --error=slurm-%j-mpi.err

# Matrix Multiplication MPI Benchmark
# This script runs the MPI implementation with varying process counts

echo "=== Matrix Multiplication MPI Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo ""

# Load required modules
module load gcc/11.2.0
module load openmpi/4.1.1
module list

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create results directory
RESULTS_DIR="results/raw"
mkdir -p $RESULTS_DIR

# Output file
OUTPUT_FILE="$RESULTS_DIR/mpi_$(date +%Y%m%d_%H%M%S).csv"

# Write CSV header
echo "timestamp,implementation,matrix_size,execution_time_ms,gflops,threads,processes,node" > $OUTPUT_FILE

# Matrix sizes to test
SIZES=(512 1024 2048 4096)

# Process counts to test
PROCS=(1 2 4 8 16 32)

# Number of iterations per configuration
ITERATIONS=10

echo "Running MPI benchmarks..."
echo ""

# Loop through matrix sizes
for SIZE in "${SIZES[@]}"; do
    echo "Testing matrix size: ${SIZE}x${SIZE}"
    
    # Loop through process counts
    for NPROCS in "${PROCS[@]}"; do
        # Skip if requesting more processes than allocated
        if [ $NPROCS -gt $SLURM_NTASKS ]; then
            echo "  Skipping $NPROCS processes (exceeds allocation)"
            continue
        fi
        
        echo "  Process count: $NPROCS"
        
        for i in $(seq 1 $ITERATIONS); do
            # Run the MPI implementation
            mpirun -np $NPROCS ./c/bin/mpi_mm $SIZE >> $OUTPUT_FILE
            
            if [ $? -ne 0 ]; then
                echo "Error running mpi_mm for size $SIZE with $NPROCS processes"
                exit 1
            fi
        done
        
        echo "    Completed $NPROCS processes"
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
echo "Output file size: $(du -h $OUTPUT_FILE | cut -f1)"
