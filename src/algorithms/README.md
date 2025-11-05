# Advanced Algorithms - Algorithmic Improvements

## Purpose

Evaluate algorithmic alternatives to the standard O(N³) matrix multiplication.

## Suggested Experiments

### Experiment 1: Strassen's Algorithm ⭐
**Goal**: Reduce computational complexity from O(N³) to O(N^2.807)

**Implementation**:
- Recursive divide-and-conquer
- 7 multiplications instead of 8
- Base case: Switch to standard algorithm for small matrices

**Base case sizes to test**: 64, 128, 256

**Matrix sizes**: 512, 1024, 2048, 4096

**Expected Results**:
- Crossover point around N=512-1024
- 20-40% speedup for large matrices (N>2048)
- More memory operations (can be slower for small N)

**Key Performance Win**: 20-40% for very large matrices

### Experiment 2: Winograd's Variant
**Goal**: Reduce additions in Strassen's algorithm

**Expected Impact**: 5-10% improvement over basic Strassen

### Experiment 3: Cache-Oblivious Algorithm
**Goal**: Optimal cache performance without knowing cache size

**Implementation**: Recursive blocking

**Expected Result**: Similar to optimal manual blocking

### Experiment 4: Fast Rectangular Multiplication
**Goal**: Optimize for non-square matrices

**Applications**: When M ≠ N ≠ K

### Experiment 5: Coppersmith-Winograd Variants
**Goal**: Theoretical optimal (O(N^2.376))

**Note**: Impractical for realistic sizes (crossover > 10^10)

## Most Important Performance Wins

1. **Strassen's algorithm** - 20-40% for N>2048 (⭐⭐⭐)
2. **Cache-oblivious** - Automatic optimization (⭐⭐⭐)
3. **Winograd variant** - 5-10% improvement (⭐⭐)

**Reality Check**: For most practical sizes, well-optimized O(N³) algorithms outperform asymptotically faster algorithms due to better cache behavior and lower overhead.

**Recommendation**: Focus on cache optimization + OpenMP/CUDA rather than Strassen for HPC benchmarking.
