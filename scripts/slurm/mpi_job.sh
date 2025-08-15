#!/bin/bash
#SBATCH --job-name=mm_mpi
#SBATCH --partition=short
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --time=60
#SBATCH --output=out_mpi.txt
#SBATCH --error=err_mpi.txt

module purge
module load gcc/13.2.0
module load openmpi/4.1.4-gcc-13.2.0

# Build MPI program (replace with your source path)
# mpicc -O3 -o mm_mpi src/mpi/summa.c

# Launch with srun (Orca uses srun, not mpirun)
srun -n 64 ./mm_mpi
