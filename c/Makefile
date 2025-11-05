# Makefile for Matrix Multiplication Performance Comparison
# Targets: naive, openmp, mpi, cuda, cache_opt, compiler_opt, algorithms

CC = gcc
CXX = g++
MPICC = mpicc
MPICXX = mpicxx
NVCC = nvcc

# Directories
SRC_DIR = src
INC_DIR = include
BIN_DIR = bin
BUILD_DIR = build

# Compiler flags
CFLAGS = -Wall -I$(INC_DIR)
CXXFLAGS = -Wall -std=c++11 -I$(INC_DIR)
OMPFLAGS = -fopenmp
MPIFLAGS = -I$(INC_DIR)
NVCCFLAGS = -I$(INC_DIR) -arch=sm_60

# Optimization levels for testing
OPT_NONE = -O0
OPT_1 = -O1
OPT_2 = -O2
OPT_3 = -O3
OPT_FAST = -Ofast -march=native

# Create directories
$(shell mkdir -p $(BIN_DIR) $(BUILD_DIR))

# Targets
.PHONY: all clean naive openmp mpi cuda cache_opt compiler_opt algorithms

all: naive openmp mpi cuda cache_opt

# Naive implementation
naive: $(BIN_DIR)/naive_mm

$(BIN_DIR)/naive_mm: $(SRC_DIR)/naive/matrix_mult.cpp
	$(CXX) $(CXXFLAGS) $(OPT_2) $< -o $@

# OpenMP implementation
openmp: $(BIN_DIR)/openmp_mm

$(BIN_DIR)/openmp_mm: $(SRC_DIR)/openmp/matrix_mult.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(OPT_2) $< -o $@

# MPI implementation
mpi: $(BIN_DIR)/mpi_mm

$(BIN_DIR)/mpi_mm: $(SRC_DIR)/mpi/matrix_mult.cpp
	$(MPICXX) $(CXXFLAGS) $(MPIFLAGS) $(OPT_2) $< -o $@

# CUDA implementation
cuda: $(BIN_DIR)/cuda_mm

$(BIN_DIR)/cuda_mm: $(SRC_DIR)/cuda/matrix_mult.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Cache optimizations
cache_opt: $(BIN_DIR)/cache_opt_mm

$(BIN_DIR)/cache_opt_mm: $(SRC_DIR)/cache_opt/matrix_mult.cpp
	$(CXX) $(CXXFLAGS) $(OPT_3) $< -o $@

# Compiler optimization experiments
compiler_opt: $(BIN_DIR)/comp_opt_O0 $(BIN_DIR)/comp_opt_O1 $(BIN_DIR)/comp_opt_O2 $(BIN_DIR)/comp_opt_O3 $(BIN_DIR)/comp_opt_Ofast

$(BIN_DIR)/comp_opt_O0: $(SRC_DIR)/compiler_opt/matrix_mult.cpp
	$(CXX) $(CXXFLAGS) $(OPT_NONE) $< -o $@

$(BIN_DIR)/comp_opt_O1: $(SRC_DIR)/compiler_opt/matrix_mult.cpp
	$(CXX) $(CXXFLAGS) $(OPT_1) $< -o $@

$(BIN_DIR)/comp_opt_O2: $(SRC_DIR)/compiler_opt/matrix_mult.cpp
	$(CXX) $(CXXFLAGS) $(OPT_2) $< -o $@

$(BIN_DIR)/comp_opt_O3: $(SRC_DIR)/compiler_opt/matrix_mult.cpp
	$(CXX) $(CXXFLAGS) $(OPT_3) $< -o $@

$(BIN_DIR)/comp_opt_Ofast: $(SRC_DIR)/compiler_opt/matrix_mult.cpp
	$(CXX) $(CXXFLAGS) $(OPT_FAST) $< -o $@

# Advanced algorithms
algorithms: $(BIN_DIR)/strassen_mm $(BIN_DIR)/blocked_mm

$(BIN_DIR)/strassen_mm: $(SRC_DIR)/algorithms/strassen.cpp
	$(CXX) $(CXXFLAGS) $(OPT_3) $< -o $@

$(BIN_DIR)/blocked_mm: $(SRC_DIR)/algorithms/blocked.cpp
	$(CXX) $(CXXFLAGS) $(OPT_3) $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)/* $(BUILD_DIR)/*

# Help target
help:
	@echo "Matrix Multiplication Build System"
	@echo "Available targets:"
	@echo "  all          - Build all implementations"
	@echo "  naive        - Build naive implementation"
	@echo "  openmp       - Build OpenMP implementation"
	@echo "  mpi          - Build MPI implementation"
	@echo "  cuda         - Build CUDA implementation"
	@echo "  cache_opt    - Build cache-optimized version"
	@echo "  compiler_opt - Build all compiler optimization variants"
	@echo "  algorithms   - Build advanced algorithm implementations"
	@echo "  clean        - Remove all built files"
