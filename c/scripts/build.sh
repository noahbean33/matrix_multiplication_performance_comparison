#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
BIN_DIR="$ROOT_DIR/bin"
SRC_DIR="$ROOT_DIR/src"
INC_DIR="$ROOT_DIR/include"
mkdir -p "$BIN_DIR"

CFLAGS="-O3 -march=native -ffp-contract=fast -funroll-loops -Wall -Wextra -std=c11"

# Link all sources
cmd=(gcc $CFLAGS -I"$INC_DIR" -o "$BIN_DIR/mm_experiments" \
    "$SRC_DIR/main.c" "$SRC_DIR/common.c" "$SRC_DIR/naive.c" \
    "$SRC_DIR/blocked.c" "$SRC_DIR/microkernel_avx.c")

echo "[build] ${cmd[*]}"
"${cmd[@]}"
echo "[build] output -> $BIN_DIR/mm_experiments"
