# Makefile for CUDA Matrix Multiplication Benchmarks
# Usage:
#   make          - Build the main benchmarking program
#   make clean    - Remove compiled binaries
#   make run      - Build and run benchmarks

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -std=c++11 -O3 -arch=sm_60
# Adjust -arch flag based on your GPU:
#   sm_60: Pascal (GTX 10xx, P100)
#   sm_70: Volta (V100)
#   sm_75: Turing (RTX 20xx, T4)
#   sm_80: Ampere (A100, RTX 30xx)
#   sm_86: Ampere (RTX 3090)
#   sm_89: Ada (RTX 40xx)

# Target executable
TARGET = matrix_multiplication
SOURCE = matrix_multiplication.cu

# Default target
all: $(TARGET)

# Build the main program
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Run with output to CSV file
benchmark: $(TARGET)
	./$(TARGET) > cuda_results.csv
	@echo "Results saved to cuda_results.csv"

# Clean up
clean:
	rm -f $(TARGET) cuda_results.csv

# Phony targets
.PHONY: all run benchmark clean
