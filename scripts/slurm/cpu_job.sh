#!/bin/bash
#SBATCH --job-name=mm_cpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=0
#SBATCH --time=30
#SBATCH --output=out_cpu.txt
#SBATCH --error=err_cpu.txt

module purge
module load gcc/13.2.0

# Build C benchmark (example)
gcc -O3 -march=native -ffp-contract=fast -funroll-loops -o matrix_multiplication c/matrix_multiplication.c

# Run and capture CSV
./matrix_multiplication > data/results/c_optimized_result.csv
