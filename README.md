# **Matrix Multiplication Performance Comparison**

This repository contains the implementation and analysis of matrix multiplication using Python, C, and GPU acceleration. It explores the performance, scalability, and trade-offs between these methods for various matrix sizes.

---

## Project Update (2025-08-14)

This repository now focuses on a deep-dive HPC performance comparison of matrix multiplication on the Orca cluster. SystemVerilog/FPGA content has been dropped. The scope is:

- CPU: C baselines, cache-blocked micro-kernel with SIMD, OpenMP tiling.
- GPU: CUDA kernels (naïve → tiled → vectorized → WMMA) with cuBLAS baseline.
- MPI: Multi-node SUMMA using OpenMPI and Slurm `srun`.

### New Directory Structure

- `c/` — C implementation (prints CSV)
- `python/` — Python baselines (pure/NumPy)
- `src/cuda/` — CUDA kernels (to be added)
- `src/mpi/` — MPI SUMMA (to be added)
- `benchmarks/` — Harness/configs for cpu/cuda/mpi
- `scripts/slurm/` — Slurm job scripts (`cpu_job.sh`, `gpu_job.sh`, `mpi_job.sh`)
- `data/results/` — Standardized CSV outputs
- `plots/` — Generated figures
- `orca-website-main/` — Orca hardware/software docs snapshot

### How to Run on Orca (Slurm)

1) Load modules

```bash
module load gcc/13.2.0
module load python
module load cuda              # for GPU
module load openmpi/4.1.4-gcc-13.2.0  # for MPI
```

2) C baseline (single node)

```bash
gcc -O3 -march=native -ffp-contract=fast -funroll-loops -o matrix_multiplication c/matrix_multiplication.c
./matrix_multiplication > data/results/c_optimized_result.csv
```

3) Submit Slurm jobs

```bash
sbatch scripts/slurm/cpu_job.sh           # CPU (64 cores)
sbatch scripts/slurm/gpu_job.sh           # GPU (L40S by default; change to a30 if needed)
sbatch scripts/slurm/mpi_job.sh           # MPI multi-node CPU
```

4) CUDA build flags (A30 + L40S)

```bash
nvcc -O3 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_89,code=sm_89 -o mm_cuda src/cuda/mm.cu
```

Notes:
- SystemVerilog has been removed from scope.
- Prefer SGEMM (FP32) for GPU comparisons across A30/L40S.
- Use `OMP_PROC_BIND=close` and `OMP_PLACES=cores` for OpenMP runs.

---

## **Overview**
Matrix multiplication is a fundamental algorithm widely used in scientific computing, signal pprocessing, and machine learning. This project evaluates different implementations of matrix multiplication by comparing their execution times in order to assess the quantitive and qualitative trade-offs involved.

---

## **Features**
1. **Implementations:**
   - **Python:** Basic triple-loop and optimized NumPy implementations.
   - **C:** Baseline and compiler-optimized (`-O3`) implementations; OpenMP tiling planned.
   - **CUDA (planned):** Naïve → tiled shared-memory → vectorized → WMMA; cuBLAS baseline.
   - **MPI (planned):** Multi-node SUMMA using OpenMPI and Slurm `srun`.

2. **Performance Analysis:**
   - Comparison across varying matrix sizes (2×2 to 1002×1002).
   - Execution time measurements and scalability evaluation in units of seconds.

3. **Trade-offs:**
   - Ease of implementation versus computational efficiency.
   - Hardware complexity versus software complexity.

4. **Challenges Addressed:**
   - Achieving high utilization on EPYC CPUs via blocking, SIMD, and OpenMP.
   - Tuning CUDA kernels for A30/L40S GPUs (occupancy, memory efficiency).
   - Orchestrating multi-node runs with Slurm and MPI.

---

## **Files and Directory Structure**

### **Code & Benchmarks**
- `python/`: Python baseline scripts (pure and NumPy).
- `c/`: C implementation (baseline CSV output).
- `src/cuda/`: CUDA kernels (to be added).
- `src/mpi/`: MPI SUMMA code (to be added).
- `benchmarks/cpu/`, `benchmarks/cuda/`, `benchmarks/mpi/`: Benchmark harnesses and configs.

### **Jobs & Data**
- `scripts/slurm/`: Slurm job scripts for Orca (`cpu_job.sh`, `gpu_job.sh`, `mpi_job.sh`).
- `data/results/`: Standardized CSV outputs and aggregated results.
- `plots/`: Generated figures (time, GFLOP/s, roofline, scaling).

### **Reports**
- `Matrix_Multiplication_Performance_Comparison_ECE_472_Project.pdf`: Prior write-up.
- `orca-website-main/`: Orca cluster docs snapshot (hardware/software usage).

---

## **How to Run**

See the section above “How to Run on Orca (Slurm)” for module loads, compilation, job submission, and CUDA flags. For local testing, you can still run the Python and C baselines directly.

# Notes

- SystemVerilog/FPGA content has been removed from scope for this repository.

## Results

The performance evaluation highlighted the following:

- **Python (Basic):** Easy to implement but slow for large matrices.
- **NumPy:** Leveraged optimized C libraries for the fastest performance.
- **C (Optimized):** Balanced speed and control, suitable for most use cases.

For detailed graphs and analysis, see `Matrix_Multiplication_Performance_Comparison_ECE_472_Project.pdf` and generated plots in `plots/`.

---

## Lessons Learned

- Hardware-level parallelism is critical for scaling computationally intensive tasks.
- Compiler optimizations significantly improve software performance with minimal effort.

From HPC experiments on Orca:
- CPU performance hinges on cache-aware blocking, SIMD, and thread affinity.
- GPU performance depends on tiling, memory coalescing, and occupancy; cuBLAS is an upper bound.
- MPI scaling is constrained by communication; SUMMA with overlap is a strong baseline.

---

## Future Work

- Implement OpenMP micro-kernel and compare vs OpenBLAS/BLIS.
- Add CUDA tiled/vectorized/WMMA kernels and compare vs cuBLAS.
- Implement MPI SUMMA and collect strong/weak scaling on Orca.

---

## Contributions

Contributions to enhance the implementations or add new features are welcome!

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments

Thanks to the authors of the referenced books, tools, and tutorials that supported this project.
