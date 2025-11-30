#!/bin/bash
#
# run_benchmarks.sh - Automated CUDA matrix multiplication benchmarking
#
# This script compiles and runs the CUDA benchmarks, saving results to CSV
#

set -e  # Exit on error

echo "========================================="
echo "CUDA Matrix Multiplication Benchmarking"
echo "========================================="
echo ""

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed."
    exit 1
fi

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# Compile
echo "Compiling CUDA code..."
make clean
make
echo ""

# Run benchmarks
echo "Running benchmarks..."
OUTPUT_FILE="cuda_results_$(date +%Y%m%d_%H%M%S).csv"
./matrix_multiplication | tee "$OUTPUT_FILE"
echo ""

# Summary
echo "========================================="
echo "Benchmarking Complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "========================================="

# Optional: Quick analysis
if command -v python3 &> /dev/null; then
    echo ""
    echo "Quick Analysis:"
    python3 - <<EOF
import csv
with open('$OUTPUT_FILE', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)
    
if data:
    print(f"Total runs: {len(data)}")
    print(f"Matrix sizes tested: {sorted(set(row['matrix_size'] for row in data))}")
    
    # Find best GFLOPS
    best = max(data, key=lambda x: float(x.get('total_gflops', 0)))
    print(f"\nBest performance:")
    print(f"  Implementation: {best['implementation']}")
    print(f"  Matrix size: {best['matrix_size']}")
    print(f"  GFLOPS: {best['total_gflops']}")
    print(f"  Block size: {best['block_size']}")
    
    # Check verification
    failed = [row for row in data if row.get('verification') != 'PASS']
    if failed:
        print(f"\nWarning: {len(failed)} benchmarks failed verification!")
    else:
        print(f"\n✓ All benchmarks passed verification")
EOF
fi
