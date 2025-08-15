#!/usr/bin/env bash
# Source this on Orca to load standard modules
# usage: source scripts/env/load_modules.sh [cpu|gpu|mpi]
set -euo pipefail
FLAVOR=${1:-cpu}
module load gcc/13.2.0
module load python
if [[ "$FLAVOR" == "gpu" ]]; then
  module load cuda
fi
if [[ "$FLAVOR" == "mpi" ]]; then
  module load openmpi/4.1.4-gcc-13.2.0
fi
module list
