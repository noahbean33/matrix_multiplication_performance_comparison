#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-results/$(date +%F)}
mkdir -p "$OUT_DIR"

HOSTNAME=$(hostname || echo unknown)
SLURM_JOB_ID_VAL=${SLURM_JOB_ID:-}

# Basic commands (best-effort)
CPU_JSON=$(lscpu -J 2>/dev/null || true)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ' || echo 0)
GPU_DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)
MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
OS_NAME=$(source /etc/os-release 2>/dev/null; echo ${NAME:-unknown})
OS_VER=$(source /etc/os-release 2>/dev/null; echo ${VERSION_ID:-unknown})

GCC_VER=$(gcc --version 2>/dev/null | head -n1 | awk '{print $3}')
NVCC_VER=$(nvcc --version 2>/dev/null | tail -n1 | awk '{print $2}' || true)
MPIRUN_VER=$(mpirun --version 2>/dev/null | head -n1 || true)
BLAS_VENDOR=${BLAS_VENDOR:-OpenBLAS}

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "dirty")

# Minimal CPU model from lscpu JSON
CPU_MODEL=$(echo "$CPU_JSON" | awk -F '"' '/"Model name"/ {getline; print $4}' 2>/dev/null | head -n1)

cat > "$OUT_DIR/env.json" <<EOF
{
  "datetime_iso": "$(date --iso-8601=seconds)",
  "hostname": "$HOSTNAME",
  "slurm_job_id": "$SLURM_JOB_ID_VAL",
  "cpu": {"model":"${CPU_MODEL:-unknown}"},
  "gpu": {"model":"${GPU_NAME:-}", "count": ${GPU_COUNT:-0}, "driver":"${GPU_DRV:-}"},
  "memory_gb": ${MEM_GB:-0},
  "os": {"name":"$OS_NAME", "version":"$OS_VER"},
  "compiler": {"name":"gcc", "version":"${GCC_VER:-}", "flags":"-O3 -march=native -ffp-contract=fast"},
  "blas": {"vendor":"$BLAS_VENDOR", "version":""},
  "cuda": {"version":"${NVCC_VER:-}"},
  "mpi": {"vendor":"OpenMPI", "version":"${MPIRUN_VER:-}"},
  "git_commit": "$GIT_COMMIT"
}
EOF

echo "[probe_env] wrote $OUT_DIR/env.json"
