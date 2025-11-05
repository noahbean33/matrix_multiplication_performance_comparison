# ORCA Cluster Setup Guide

## Initial Setup

### 1. Connect to ORCA

```bash
ssh your_username@orca.hpc.edu
```

### 2. Load Required Modules

```bash
# Load compiler and MPI
module load gcc/11.2.0
module load openmpi/4.1.1

# Load CUDA (if using GPU nodes)
module load cuda/11.7

# Load Python
module load python/3.9
```

### 3. Clone the Repository

```bash
cd $HOME
git clone <repository_url> matrix_multiplication_performance_comparison
cd matrix_multiplication_performance_comparison
```

## Environment Setup

### Set Up Python Virtual Environment

```bash
cd python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Build C/C++ Implementations

```bash
cd ../c
make all
```

## SLURM Configuration

### Understanding SLURM Partitions

Check available partitions:
```bash
sinfo
```

Common partitions on ORCA:
- `short` - Short jobs (<4 hours)
- `long` - Long jobs (<24 hours)
- `gpu` - GPU nodes
- `bigmem` - High memory nodes

### Resource Allocation

Check your account and QOS:
```bash
sacctmgr show user $USER
```

Check current cluster usage:
```bash
squeue -u $USER
```

## Running Your First Benchmark

### Interactive Session

Request an interactive session for testing:
```bash
srun --partition=short --nodes=1 --ntasks=8 --time=01:00:00 --pty bash
```

### Submit a Batch Job

```bash
sbatch scripts/slurm/benchmark_naive.sh
```

Check job status:
```bash
squeue -u $USER
```

View job output:
```bash
cat slurm-<jobid>.out
```

## Storage Locations

- **Home Directory**: `$HOME` - Limited space, backed up
- **Scratch Space**: `$SCRATCH` or `/scratch/$USER` - Large temporary storage
- **Project Space**: If applicable

Store large result files in scratch space:
```bash
export RESULTS_DIR=$SCRATCH/matrix_mult_results
mkdir -p $RESULTS_DIR
```

## Best Practices

1. **Always test locally first** with small matrix sizes
2. **Use job arrays** for parameter sweeps
3. **Monitor your resource usage** with `seff <jobid>`
4. **Clean up scratch space** regularly
5. **Set appropriate time limits** to avoid wasting resources
6. **Use dependencies** for multi-stage pipelines

## Troubleshooting

### Module Not Found
```bash
module avail  # List all available modules
```

### Out of Memory
- Reduce matrix size
- Request more memory: `#SBATCH --mem=32GB`

### Job Timeout
- Increase time limit: `#SBATCH --time=04:00:00`

### GPU Issues
- Check GPU availability: `nvidia-smi`
- Verify CUDA module is loaded

## Useful Commands

```bash
# Check job efficiency
seff <jobid>

# Cancel a job
scancel <jobid>

# Check account balance/usage
sshare -U

# View detailed job info
scontrol show job <jobid>

# Monitor running job
watch -n 5 squeue -u $USER
```

## Contact

For ORCA-specific issues, contact your cluster support team.
