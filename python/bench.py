#!/usr/bin/env python3
import argparse
import os
import time
import csv
import numpy as np


def mm_py(A, B):
    n = len(A)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = A[i][k]
            for j in range(n):
                C[i][j] += aik * B[k][j]
    return C


def bench_once(n: int, impl: str) -> float:
    if impl == "python":
        rng = np.random.default_rng(1)
        A = rng.random((n, n)).tolist()
        B = rng.random((n, n)).tolist()
        t0 = time.perf_counter()
        _ = mm_py(A, B)
        t1 = time.perf_counter()
        return t1 - t0
    elif impl == "numpy":
        rng = np.random.default_rng(1)
        A = rng.random((n, n), dtype=np.float64)
        B = rng.random((n, n), dtype=np.float64)
        # warm-up
        _ = A @ B
        t0 = time.perf_counter()
        _ = A @ B
        t1 = time.perf_counter()
        return t1 - t0
    else:
        raise ValueError(f"unknown impl: {impl}")


def main():
    ap = argparse.ArgumentParser(description="Python/NumPy matrix multiply benchmark")
    ap.add_argument("--impl", choices=["python", "numpy"], default="numpy")
    ap.add_argument("--min-n", type=int, default=128)
    ap.add_argument("--max-n", type=int, default=1024)
    ap.add_argument("--step", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outfile", type=str, default=os.path.join("data", "results", "py_bench.csv"))
    ap.add_argument("--variant", type=str, default="naive")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    header = ["n", "time_seconds", "variant", "threads", "impl", "gflops"]
    out_exists = os.path.exists(args.outfile)
    with open(args.outfile, "a", newline="") as f:
        w = csv.writer(f)
        if not out_exists:
            w.writerow(header)
        for n in range(args.min_n, args.max_n + 1, args.step):
            times = []
            for _ in range(args.repeats):
                t = bench_once(n, args.impl)
                times.append(t)
            tmed = sorted(times)[len(times)//2]
            gflops = (2.0 * (n ** 3)) / tmed / 1e9
            # threads is not controlled here; leave as 1 or unknown
            w.writerow([n, f"{tmed:.6f}", args.variant, 1, args.impl, f"{gflops:.3f}"])
            print(f"n={n} impl={args.impl} t={tmed:.6f}s GF/s={gflops:.3f}")


if __name__ == "__main__":
    main()
