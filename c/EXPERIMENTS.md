# C Experiments: Algorithms, Data Layouts, and Compiler Optimizations

This guide tracks experiments for SGEMM/DGEMM on Orca CPU nodes (AMD EPYC Genoa 9534, 64 cores).

## Targets
- Single-core baselines and micro-kernel (SIMD)
- Multicore with OpenMP tiling
- Data layout choices and B-transpose
- Compiler flags sensitivity

## Variants
- naive: triple-loop reference
- blocked: 3-level cache tiling (Mc, Nc, Kc); Mr×Nr micro-kernel
- microkernel_avx: AVX2/AVX-512 intrinsics with FMA

## Data Layout
- Row-major, column-major (if needed)
- Store B as transposed to make inner loop unit-stride for A and B^T
- Align allocations to 64 bytes; use `restrict` and `__builtin_assume_aligned`

## Compiler Flags
- Baseline: `-O3 -march=native -ffp-contract=fast -funroll-loops`
- Optional: `-Ofast -fno-math-errno -ffast-math` (document accuracy trade-offs)
- Reports: `-fopt-info-vec-optimized` (GCC) for vectorization diagnostics

## OpenMP
- Outer tile parallelism: `#pragma omp parallel for collapse(2) schedule(static)`
- Inner loop: `#pragma omp simd`
- Env: `OMP_NUM_THREADS`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`

## Measurement
- Warm-up then median of N runs
- Record: n, time (s), GFLOP/s (=2*n^3/time), variant, threads, commit
- Tools: `perf stat` (IPC, cache), `likwid` (if available)

## Build/Run Skeleton
- See `c/scripts/` for build/run stubs
- Baseline build (single-file): `gcc -O3 -march=native -o mm c/matrix_multiplication.c`
- Multi-file (when variants ready): link `c/src/*.c` and include `c/include/mm.h`

## TODO
- [ ] Implement blocked kernel with B^T and (Mc,Nc,Kc) tuning
- [ ] Implement AVX2/AVX-512 micro-kernel (Mr×Nr)
- [ ] Add OpenMP tiling and affinity controls
- [ ] Add CSV writer with unified schema to `data/results/`
