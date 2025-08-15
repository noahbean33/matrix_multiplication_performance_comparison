#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
BIN="$ROOT_DIR/bin/mm_experiments"
OUT_DIR="$(cd "$ROOT_DIR/.." && pwd)/data/results"
mkdir -p "$OUT_DIR"

if [[ ! -x "$BIN" ]]; then
  echo "Binary not found, building..." >&2
  "$ROOT_DIR/scripts/build.sh"
fi

SIZES=(256 512 1024 1536 2048)
VARIANTS=(naive blocked microkernel_avx)
OUT_CSV="$OUT_DIR/c_experiments_$(date +%Y%m%d_%H%M%S).csv"

echo "n,time_seconds,variant,threads" > "$OUT_CSV"
for n in "${SIZES[@]}"; do
  for v in "${VARIANTS[@]}"; do
    echo "[run] n=$n variant=$v"
    "$BIN" "$n" "$v" >> "$OUT_CSV"
  done
done

echo "[done] results -> $OUT_CSV"
