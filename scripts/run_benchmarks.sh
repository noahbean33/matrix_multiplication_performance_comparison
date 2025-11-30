#!/bin/bash
# Run all matrix multiplication benchmarks
# Outputs results to results/raw/ directory

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Matrix Multiplication Benchmark Suite ===${NC}"

# Create results directories
mkdir -p results/raw
mkdir -p results/plots

# Timestamp for this benchmark run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/raw/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

echo -e "${BLUE}Results will be saved to: ${RESULTS_DIR}${NC}\n"

# Matrix sizes to test
SIZES=(512 1024 2048 4096)

# Function to check if binary exists
binary_exists() {
    [ -f "$1" ]
}

# ==================== BASELINE C ====================
echo -e "${BLUE}Running baseline C benchmarks...${NC}"
if binary_exists "bin/baseline"; then
    for size in "${SIZES[@]}"; do
        echo "  Testing size ${size}x${size}..."
        ./bin/baseline ${size} >> "${RESULTS_DIR}/baseline.csv"
    done
    echo -e "${GREEN}✓ Baseline complete${NC}\n"
else
    echo -e "${YELLOW}⚠ bin/baseline not found, skipping${NC}\n"
fi

# ==================== COMPILER OPTIMIZATIONS ====================
echo -e "${BLUE}Running compiler optimization benchmarks...${NC}"
for opt in O1 O2 O3 Ofast; do
    if binary_exists "bin/optimized_${opt}"; then
        echo "  Testing -${opt}..."
        for size in "${SIZES[@]}"; do
            echo "    Size ${size}x${size}..."
            ./bin/optimized_${opt} ${size} >> "${RESULTS_DIR}/optimized_${opt}.csv"
        done
        echo -e "${GREEN}✓ -${opt} complete${NC}"
    else
        echo -e "${YELLOW}⚠ bin/optimized_${opt} not found, skipping${NC}"
    fi
done
echo ""

# ==================== OPENMP ====================
echo -e "${BLUE}Running OpenMP benchmarks...${NC}"
if binary_exists "bin/openmp"; then
    THREAD_COUNTS=(1 2 4 8 16)
    for threads in "${THREAD_COUNTS[@]}"; do
        echo "  Testing with ${threads} threads..."
        for size in "${SIZES[@]}"; do
            echo "    Size ${size}x${size}..."
            OMP_NUM_THREADS=${threads} ./bin/openmp ${size} >> "${RESULTS_DIR}/openmp_${threads}t.csv"
        done
    done
    echo -e "${GREEN}✓ OpenMP complete${NC}\n"
else
    echo -e "${YELLOW}⚠ bin/openmp not found, skipping${NC}\n"
fi

# ==================== MPI ====================
echo -e "${BLUE}Running MPI benchmarks...${NC}"
if binary_exists "bin/mpi"; then
    PROCESS_COUNTS=(1 2 4 8)
    for procs in "${PROCESS_COUNTS[@]}"; do
        echo "  Testing with ${procs} processes..."
        for size in "${SIZES[@]}"; do
            echo "    Size ${size}x${size}..."
            mpirun -np ${procs} ./bin/mpi ${size} >> "${RESULTS_DIR}/mpi_${procs}p.csv"
        done
    done
    echo -e "${GREEN}✓ MPI complete${NC}\n"
else
    echo -e "${YELLOW}⚠ bin/mpi not found, skipping${NC}\n"
fi

# ==================== CUDA ====================
echo -e "${BLUE}Running CUDA benchmarks...${NC}"
if [ -f "src/cuda/matrix_multiplication" ]; then
    echo "  Running CUDA benchmark suite..."
    ./src/cuda/matrix_multiplication > "${RESULTS_DIR}/cuda.csv"
    echo -e "${GREEN}✓ CUDA complete${NC}\n"
else
    echo -e "${YELLOW}⚠ src/cuda/matrix_multiplication not found, skipping${NC}\n"
fi

# ==================== SUMMARY ====================
echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo -e "Results saved to: ${RESULTS_DIR}/"
echo -e "\nGenerated files:"
ls -lh "${RESULTS_DIR}"/*.csv 2>/dev/null || echo "No CSV files generated"

echo -e "\n${BLUE}To visualize results, run:${NC}"
echo -e "  python scripts/plot_results.py ${RESULTS_DIR}"

# Combine all results into a single file
echo -e "\n${BLUE}Combining results...${NC}"
cat "${RESULTS_DIR}"/*.csv 2>/dev/null > "${RESULTS_DIR}/combined_results.csv" || true
echo -e "${GREEN}✓ Combined results saved to ${RESULTS_DIR}/combined_results.csv${NC}"
