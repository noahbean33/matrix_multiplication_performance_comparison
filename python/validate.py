#!/usr/bin/env python3
import argparse
import numpy as np


def validate(n: int, tol: float = 1e-8) -> bool:
    rng = np.random.default_rng(0)
    A = rng.random((n, n), dtype=np.float64)
    B = rng.random((n, n), dtype=np.float64)

    # Reference via NumPy
    C_ref = A @ B

    # Placeholder for external results: here we just recompute; later, load from C/CSV or binary
    C_test = A @ B

    diff = np.abs(C_ref - C_test)
    max_abs = float(diff.max())
    rel = diff / (np.abs(C_ref) + 1e-12)
    max_rel = float(rel.max())

    ok = (max_abs <= tol) or (max_rel <= tol)
    print(f"n={n} max_abs={max_abs:.3e} max_rel={max_rel:.3e} ok={ok}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Validate matmul vs NumPy reference")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--tol", type=float, default=1e-8)
    args = ap.parse_args()
    _ = validate(args.n, args.tol)


if __name__ == "__main__":
    main()
