#!/bin/bash
#SBATCH --job-name=mm_gpu
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=30
#SBATCH --output=out_gpu.txt
#SBATCH --error=err_gpu.txt

module purge
module load cuda gcc/13.2.0

# Example CUDA build (replace src/cuda/mm.cu with your file)
# nvcc -O3 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -o mm_cuda src/cuda/mm.cu

# Example run
# ./mm_cuda
