# Compiler Optimizations - Compiler Flag Impact

## Purpose

Quantify performance impact of different compiler optimization levels and flags.

## Suggested Experiments

### Experiment 1: Optimization Level Comparison ⭐ CRITICAL
**Goal**: Measure performance across optimization levels

**Flags to test**:
- `-O0` - No optimization (baseline)
- `-O1` - Basic optimizations
- `-O2` - Recommended optimizations
- `-O3` - Aggressive optimizations
- `-Ofast` - Aggressive + non-standards compliant
- `-Os` - Optimize for size

**Matrix sizes**: 512, 1024, 2048

**Expected Results**:
- O0: Baseline (slowest)
- O1: 2-3x faster than O0
- O2: 3-5x faster than O0
- O3: 4-8x faster than O0
- Ofast: 5-10x faster than O0 (slight numerical differences possible)

**Key Performance Win**: -O3 gives 5-10x improvement over -O0

### Experiment 2: Architecture-Specific Optimizations ⭐
**Goal**: Impact of CPU-specific optimizations

**Flags**:
- `-march=native` - Use all CPU instructions
- `-mtune=native` - Tune for current CPU
- `-mavx2` - AVX2 vectorization
- `-mfma` - Fused multiply-add

**Expected Results**:
- march=native: 20-50% improvement
- Enables SIMD vectorization
- Must test on target architecture

**Key Performance Win**: 30-50% additional speedup

### Experiment 3: Vectorization Analysis
**Goal**: Measure auto-vectorization impact

**Flags**:
- `-ftree-vectorize` (enabled at -O3)
- `-fopt-info-vec` - Vectorization report
- `-ffast-math` - Relaxed math for vectorization

**Matrix sizes**: 1024, 2048

**Expected Result**: 2-4x from vectorization alone

### Experiment 4: Link-Time Optimization (LTO)
**Goal**: Cross-file optimization impact

**Flags**: `-flto`

**Expected Impact**: 5-15% improvement

### Experiment 5: Profile-Guided Optimization (PGO)
**Goal**: Optimize using runtime profile

**Steps**:
1. Compile with `-fprofile-generate`
2. Run with representative data
3. Recompile with `-fprofile-use`

**Expected Impact**: 10-30% improvement

## Most Important Performance Wins

1. **-O3 vs -O0** - 5-10x speedup (⭐⭐⭐⭐⭐)
2. **-march=native** - 30-50% additional (⭐⭐⭐⭐)
3. **Vectorization** - 2-4x speedup (⭐⭐⭐⭐⭐)
4. **-Ofast** - 10-20% over -O3 (⭐⭐⭐)

**Total Potential**: 10-20x vs unoptimized code
