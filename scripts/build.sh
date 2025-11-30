#!/bin/bash
# Build script for matrix multiplication performance comparison
# Builds: baseline C, CUDA, MPI, OpenMP, and optimized versions

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Matrix Multiplication Build Script ===${NC}"

# Create bin directory if it doesn't exist
mkdir -p bin

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Build baseline C implementation
echo -e "\n${BLUE}Building baseline C implementation...${NC}"
if [ -f "src/baseline/matrix_mult.c" ]; then
    gcc -O0 src/baseline/matrix_mult.c -o bin/baseline -lm
    echo -e "${GREEN}✓ Baseline built successfully${NC}"
else
    echo -e "${RED}✗ src/baseline/matrix_mult.c not found${NC}"
fi

# Build optimized versions (different compiler flags)
echo -e "\n${BLUE}Building optimized versions...${NC}"
if [ -f "src/optimized/matrix_mult.c" ]; then
    gcc -O1 src/optimized/matrix_mult.c -o bin/optimized_O1 -lm
    gcc -O2 src/optimized/matrix_mult.c -o bin/optimized_O2 -lm
    gcc -O3 src/optimized/matrix_mult.c -o bin/optimized_O3 -lm
    gcc -Ofast src/optimized/matrix_mult.c -o bin/optimized_Ofast -lm
    echo -e "${GREEN}✓ Optimized versions built (O1, O2, O3, Ofast)${NC}"
else
    echo -e "${RED}✗ src/optimized/matrix_mult.c not found${NC}"
fi

# Build OpenMP version
echo -e "\n${BLUE}Building OpenMP implementation...${NC}"
if command_exists gcc; then
    if [ -f "src/openmp/matrix_mult_omp.c" ]; then
        gcc -O3 -fopenmp src/openmp/matrix_mult_omp.c -o bin/openmp -lm
        echo -e "${GREEN}✓ OpenMP built successfully${NC}"
    else
        echo -e "${RED}✗ src/openmp/matrix_mult_omp.c not found${NC}"
    fi
else
    echo -e "${RED}✗ GCC not found, skipping OpenMP${NC}"
fi

# Build MPI version
echo -e "\n${BLUE}Building MPI implementation...${NC}"
if command_exists mpicc; then
    if [ -f "src/mpi/matrix_mult_mpi.c" ]; then
        mpicc -O3 src/mpi/matrix_mult_mpi.c -o bin/mpi -lm
        echo -e "${GREEN}✓ MPI built successfully${NC}"
    else
        echo -e "${RED}✗ src/mpi/matrix_mult_mpi.c not found${NC}"
    fi
else
    echo -e "${RED}✗ mpicc not found, skipping MPI${NC}"
fi

# Build CUDA version
echo -e "\n${BLUE}Building CUDA implementation...${NC}"
if command_exists nvcc; then
    if [ -f "src/cuda/matrix_multiplication.cu" ]; then
        cd src/cuda
        make clean && make
        cd ../..
        echo -e "${GREEN}✓ CUDA built successfully${NC}"
    else
        echo -e "${RED}✗ src/cuda/matrix_multiplication.cu not found${NC}"
    fi
else
    echo -e "${RED}✗ nvcc not found, skipping CUDA${NC}"
fi

echo -e "\n${GREEN}=== Build complete ===${NC}"
echo "Executables are in the bin/ directory"
echo "Run ./scripts/run_benchmarks.sh to start benchmarking"
