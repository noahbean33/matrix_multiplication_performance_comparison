#!/bin/bash
# Convenience script to build all SpMV implementations

set -e  # Exit on error

echo "========================================"
echo "Building SpMV Benchmark Suite"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# CPU Build
echo -e "\n${YELLOW}[1/3] Building CPU version...${NC}"
cd cpu
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cd ../..
echo -e "${GREEN}✓ CPU build complete${NC}"

# GPU Build (optional, check for CUDA)
echo -e "\n${YELLOW}[2/3] Building GPU version...${NC}"
if command -v nvcc &> /dev/null; then
    cd gpu
    if [ -d "build" ]; then
        rm -rf build
    fi
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build .
    cd ../..
    echo -e "${GREEN}✓ GPU build complete${NC}"
else
    echo -e "${RED}✗ CUDA not found, skipping GPU build${NC}"
    echo "  Install CUDA Toolkit to build GPU version"
fi

# FPGA Build (software emulation only)
echo -e "\n${YELLOW}[3/3] Building FPGA software emulation...${NC}"
cd fpga
if [ -d "build" ]; then
    rm -rf build
fi
make sw_emu
cd ..
echo -e "${GREEN}✓ FPGA software emulation build complete${NC}"

echo ""
echo "========================================"
echo -e "${GREEN}Build Summary${NC}"
echo "========================================"
echo "CPU:  ./cpu/build/spmv_cpu"
if command -v nvcc &> /dev/null; then
    echo "GPU:  ./gpu/build/spmv_gpu"
else
    echo "GPU:  [not built - CUDA not available]"
fi
echo "FPGA: ./fpga/build/host_sw (software emulation)"
echo ""
echo "Run individual benchmarks:"
echo "  cd cpu/build && ./spmv_cpu"
echo "  cd gpu/build && ./spmv_gpu"
echo "  cd fpga && make run_sw_emu"
echo ""
