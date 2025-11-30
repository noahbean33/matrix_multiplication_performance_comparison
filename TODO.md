# TODO - Next Steps

## Immediate Tasks

### Testing & Validation
- [ ] Test `./scripts/build.sh` on your system
- [ ] Verify all compilers are installed (gcc, nvcc, mpicc)
- [ ] Run a small test with `./bin/baseline 512`
- [ ] Test CUDA implementation if GPU available
- [ ] Test MPI with `mpirun -np 2 ./bin/mpi 512`
- [ ] Test OpenMP with `OMP_NUM_THREADS=2 ./bin/openmp 512`

### Initial Benchmarks
- [ ] Run `./scripts/run_benchmarks.sh` (may take time)
- [ ] Check results in `results/raw/<timestamp>/`
- [ ] Generate plots with `python scripts/plot_results.py`
- [ ] Review plots in `results/plots/<timestamp>/`

### Code Review
- [ ] Review baseline implementation in `src/baseline/`
- [ ] Check OpenMP implementation in `src/openmp/`
- [ ] Check MPI implementation in `src/mpi/`
- [ ] Verify CUDA code still works in `src/cuda/`

## Optional Cleanup

### Archive Old Files
- [ ] Review old code in `src/algorithms/`
- [ ] Review old code in `src/cache_opt/`
- [ ] Decide whether to keep or archive old docs
- [ ] Consider archiving `orca-website/` directory
- [ ] Run `scripts/clean.sh` to remove build artifacts

### Git Cleanup
- [ ] Commit new simplified structure
- [ ] Tag this as a milestone version
- [ ] Update `.gitignore` if needed
- [ ] Consider creating a `legacy` branch for old code

## Enhancements

### Code Improvements
- [ ] Add verification to baseline/optimized implementations
- [ ] Add more block sizes to CUDA benchmarks
- [ ] Implement cache-blocked version (if desired)
- [ ] Add error handling for memory allocation failures
- [ ] Add timing breakdown for MPI communication vs computation

### Documentation
- [ ] Add performance results to README
- [ ] Document hardware specifications used for testing
- [ ] Add troubleshooting section based on issues encountered
- [ ] Create example plots for README
- [ ] Document expected speedups for each implementation

### Analysis
- [ ] Add strong vs weak scaling analysis
- [ ] Add efficiency plots (speedup / num_processors)
- [ ] Compare against theoretical peak performance
- [ ] Add memory bandwidth analysis
- [ ] Create summary table of best results

## Advanced Features (Optional)

### Additional Implementations
- [ ] Implement BLAS baseline for comparison
- [ ] Add cuBLAS version for CUDA
- [ ] Implement hybrid MPI+OpenMP version
- [ ] Add vectorization (AVX/SSE) version
- [ ] Implement blocked/tiled CPU version

### Benchmarking
- [ ] Add support for non-square matrices
- [ ] Test with different data types (double, float16)
- [ ] Add memory usage tracking
- [ ] Add cache miss analysis (if tools available)
- [ ] Test on different hardware architectures

### Automation
- [ ] Create SLURM submission script for clusters
- [ ] Add CI/CD for automatic testing
- [ ] Create Docker container for reproducibility
- [ ] Add automatic result comparison between runs
- [ ] Create web dashboard for results

### Plotting
- [ ] Add error bars for multiple runs
- [ ] Create combined comparison plots
- [ ] Add hardware utilization plots
- [ ] Generate LaTeX tables for papers
- [ ] Add interactive plots (plotly)

## Documentation Tasks

- [ ] Write methodology section
- [ ] Document hardware specifications
- [ ] Add bibliography/references
- [ ] Create performance analysis report
- [ ] Write conclusions section

## Before Publication/Submission

- [ ] Run comprehensive benchmarks on target hardware
- [ ] Verify all results are reproducible
- [ ] Check all plots are publication quality
- [ ] Spell-check all documentation
- [ ] Add acknowledgments section
- [ ] Add contact information
- [ ] Test on clean system to verify prerequisites
- [ ] Create release archive with results

## Known Issues to Address

- [ ] Windows compatibility for bash scripts (consider .bat versions)
- [ ] CSV header not printed by default (add --header flag?)
- [ ] Plot script assumes specific directory structure
- [ ] Matrix sizes hardcoded in run_benchmarks.sh
- [ ] No automatic verification of correctness

## Performance Goals

Set targets based on your hardware:
- [ ] OpenMP: Target ___x speedup with ___ threads
- [ ] MPI: Target ___x speedup with ___ processes
- [ ] CUDA: Target ___x speedup vs CPU
- [ ] Compiler opts: Target ___x speedup vs -O0

## Questions to Answer

Document in your results:
- [ ] What is the optimal OpenMP thread count?
- [ ] How does MPI scale with process count?
- [ ] What CUDA block size performs best?
- [ ] Which compiler flag gives best performance?
- [ ] What is the break-even matrix size for GPU?
- [ ] How does performance scale with matrix size?

---

**Priority**: Focus on testing and validation first, then enhancements

Mark items complete as you finish them!
