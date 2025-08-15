#!/usr/bin/env bash
#SBATCH -J c-experiments
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
module load gcc/13.2.0

ROOT_DIR="$SLURM_SUBMIT_DIR/c"
cd "$ROOT_DIR"

# Build
bash scripts/build.sh

# Sweep (single-thread baseline for now)
bash scripts/run_sweep.sh
