# MPI Matrix Multiplication - Distributed Memory Parallelization

## Purpose

Evaluate distributed memory parallelization across multiple nodes using Message Passing Interface (MPI). Quantify communication overhead, network performance impact, and multi-node scaling characteristics.

## Implementation Requirements

### Core Features
- MPI initialization and finalization
- Matrix distribution strategies
- Point-to-point and collective communication
- Build with: `mpicc -O2` or `mpicxx -O2`

### Essential MPI Calls
```c
MPI_Init(), MPI_Finalize()
MPI_Comm_size(), MPI_Comm_rank()
MPI_Send(), MPI_Recv()
MPI_Bcast(), MPI_Scatter(), MPI_Gather()
MPI_Barrier(), MPI_Wtime()
```

## Suggested Experiments

### Experiment 1: Strong Scaling Study ⭐ CRITICAL
**Goal**: Measure speedup as process count increases for fixed problem size

**Parameters**:
- Matrix size: **4096** (fixed, large enough to see scaling)
- Process counts: 1, 2, 4, 8, 16, 32, 64
- Node configuration: Vary nodes and processes per node
- Iterations: 10 per configuration

**Metrics**:
- Total execution time
- Communication time vs computation time
- Speedup = T₁ / Tₚ
- Efficiency = Speedup / P
- Network bandwidth utilization

**Expected Results**:
- Good scaling up to 16-32 processes
- Communication overhead increases with process count
- Efficiency degrades beyond certain point

**Key Performance Win**: Quantify multi-node speedup (expected 10-20x on 32 processes)

### Experiment 2: Communication Overhead Analysis ⭐ CRITICAL
**Goal**: Quantify time spent in communication vs computation

**Implementation**:
```c
double t_comp_start = MPI_Wtime();
// Computation
double t_comp = MPI_Wtime() - t_comp_start;

double t_comm_start = MPI_Wtime();
// MPI communication
double t_comm = MPI_Wtime() - t_comm_start;

// Report: computation_time, communication_time, ratio
```

**Matrix sizes**: 1024, 2048, 4096
**Process counts**: 4, 8, 16, 32

**Expected Insights**:
- Small matrices: Communication dominates (poor scaling)
- Large matrices: Computation dominates (better scaling)
- Identify optimal problem size per process

**Key Performance Win**: Understanding communication/computation ratio for efficiency

### Experiment 3: Matrix Distribution Strategies
**Goal**: Compare different data distribution approaches

**Strategies to Test**:

1. **Row-wise Block Distribution** (Simplest)
```
Process 0: Rows 0 to N/P-1
Process 1: Rows N/P to 2N/P-1
...
```

2. **Column-wise Block Distribution**
```
Process 0: Columns 0 to N/P-1
...
```

3. **2D Block-Cyclic Distribution** (ScaLAPACK style)
```
Processes arranged in P = Pr × Pc grid
Better load balance
```

4. **Checkerboard Distribution**
```
Each process gets a square block
Minimal communication for Cannon's algorithm
```

**Matrix size**: 2048
**Process counts**: 4, 9, 16 (for 2D grids)

**Expected Results**:
- 1D distribution: Simple but limited scalability
- 2D distribution: Better scaling, more complex
- 2D reduces communication volume

**Key Performance Win**: 2D distribution can improve scaling by 30-50%

### Experiment 4: MPI Algorithm Comparison ⭐
**Goal**: Compare different parallel matrix multiplication algorithms

**Algorithms**:

1. **Naive Distribution** (Broadcast B)
```c
// Each process gets A rows
// Broadcast entire B matrix
// Compute local C rows
```

2. **Cannon's Algorithm** (Optimal for square processor grids)
```c
// Initial alignment
// Iterative shifting and multiplication
// Minimal communication
```

3. **DNS Algorithm** (Divide and conquer)
```c
// Recursive decomposition
// Better cache behavior
```

**Matrix size**: 2048, 4096
**Process counts**: 4, 9, 16, 25

**Expected Results**:
- Naive: Simplest, high communication
- Cannon's: Best for large P, complex implementation
- DNS: Good cache behavior

**Key Performance Win**: Cannon's can reduce communication by 50-80%

### Experiment 5: Network Performance Impact
**Goal**: Measure impact of network topology and bandwidth

**Configurations**:
1. **Single node, multiple processes** (shared memory)
2. **Multiple nodes, 1 process per node**
3. **Multiple nodes, multiple processes per node**

**Test Matrix**:
```
Config 1: 1 node × 16 processes
Config 2: 4 nodes × 4 processes each
Config 3: 8 nodes × 2 processes each
Config 4: 16 nodes × 1 process each
```

**Matrix size**: 2048

**Expected Insights**:
- Single node: Fast (memory bandwidth)
- Multiple nodes: Network becomes bottleneck
- Optimal process/node ratio

**Key Performance Win**: Understanding network bottleneck impact

### Experiment 6: Message Size Optimization
**Goal**: Impact of message size on performance

**Variations**:
- Send entire rows/columns
- Send in chunks (size: 128, 256, 512, 1024 doubles)
- Use MPI derived datatypes

**Matrix size**: 2048
**Process count**: 8

**Expected Results**:
- Small messages: High latency overhead
- Large messages: Better bandwidth utilization
- Optimal message size depends on network

**Key Performance Win**: Message aggregation can improve performance 20-40%

### Experiment 7: Collective Communication Strategies ⭐
**Goal**: Compare collective communication patterns

**Approaches**:
1. **MPI_Bcast** - Broadcast entire matrix
2. **MPI_Scatter/Gather** - Distribute and collect
3. **Point-to-point** - Manual send/recv
4. **Non-blocking** - MPI_Isend/Irecv with overlap

**Matrix sizes**: 1024, 2048
**Process counts**: 8, 16

**Expected Results**:
- Collective operations: Optimized by MPI implementation
- Non-blocking: Overlap communication and computation
- 20-50% improvement possible

**Key Performance Win**: Non-blocking communication for computation/communication overlap

### Experiment 8: Weak Scaling Study
**Goal**: Maintain constant work per process as process count increases

**Configuration**:
```
P=1:  N=2048   (work per process = 2048²)
P=4:  N=4096   (4× processes, 4× total work)
P=16: N=8192   (16× processes, 16× total work)
P=64: N=16384  (64× processes, 64× total work)
```

**Expected Result**: Time should remain constant with perfect weak scaling

**Key Performance Win**: Demonstrates scalability to larger problems

### Experiment 9: Load Balancing Analysis
**Goal**: Measure load imbalance across processes

**Metrics per Process**:
```c
double local_time = computation_time;
double max_time, min_time, avg_time;
MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

// Load imbalance = (max_time - min_time) / avg_time
```

**Expected Insight**: Identify stragglers and imbalanced decomposition

## Performance Expectations

### Target Speedup
- **4 processes**: 3-3.5x speedup (75-87% efficiency)
- **16 processes**: 10-14x speedup (62-87% efficiency)
- **32 processes**: 18-25x speedup (56-78% efficiency)
- **64 processes**: 30-45x speedup (47-70% efficiency)

### Limitations
- Network bandwidth saturation
- Communication overhead (α + β × message_size)
- Load imbalance
- Synchronization overhead

## Key Metrics to Report

1. **Strong Scaling Efficiency** - Critical
2. **Communication vs Computation Ratio** - Critical
3. **Network Bandwidth Utilization** (GB/s)
4. **Message Count and Sizes**
5. **Load Imbalance Factor**
6. **Speedup per Node**

## ORCA-Specific Considerations

```bash
# SLURM allocation
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8

# Running
mpirun -np 32 ./mpi_mm 4096

# Or with SLURM
srun --mpi=pmix ./mpi_mm 4096
```

## Validation

- Verify results match sequential version
- Test with different process counts
- Check correctness with non-square process grids
- Validate all-to-all communication patterns

## Implementation Checklist

- [ ] Basic row-wise distribution
- [ ] MPI initialization/finalization
- [ ] Matrix scattering and gathering
- [ ] Timing communication separately
- [ ] Multiple distribution strategies
- [ ] Cannon's algorithm (advanced)
- [ ] Non-blocking communication
- [ ] Load balance measurement
- [ ] CSV output with process count and node count

## Expected Timeline

- **Implementation**: 8-12 hours (complex)
- **Testing**: 3-4 hours
- **Benchmarking**: 3-4 hours on ORCA
- **Analysis**: 3-4 hours

## Most Important Performance Wins (Ranked)

1. **Basic MPI parallelization** - 10-20x on 32 processes (⭐⭐⭐⭐⭐)
2. **2D block-cyclic distribution** - 30-50% better scaling (⭐⭐⭐⭐)
3. **Non-blocking communication** - 20-30% improvement via overlap (⭐⭐⭐⭐)
4. **Optimal message aggregation** - 20-40% reduction in communication time (⭐⭐⭐)
5. **Cannon's algorithm** - 50-80% less communication for large P (⭐⭐⭐⭐)

**Total Potential**: 100-200x speedup on 64+ processes with large matrices

**Critical Insight**: MPI excels for very large problems where data doesn't fit on one node, but communication overhead makes it less efficient than OpenMP for small problems.
