#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd

# Minimal required columns across CPU/GPU/MPI runs
REQUIRED_COLS = [
    'run_id','datetime_iso','hostname',
    'algo','impl','precision','n',
    'trial','time_s','gflops'
]

# Recommended numeric sanity ranges
RANGES = {
    'n': (1, 10_000_000),
    'time_s': (0.0, 86_400.0),  # up to a day
    'gflops': (0.0, 10_000_000.0),
}

def approx_equal(a, b, rel=0.01):
    if b == 0:
        return abs(a) < 1e-12
    return abs(a - b) / abs(b) <= rel


def validate(df: pd.DataFrame) -> int:
    errors = 0

    # Required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"ERROR missing columns: {missing}")
        errors += 1

    # Types and ranges (best-effort coercion)
    for c in ['n','trial']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-1).astype(int)
    for c in ['time_s','gflops']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    for c, (lo, hi) in RANGES.items():
        if c in df.columns:
            bad = df[(df[c] < lo) | (df[c] > hi)]
            if len(bad) > 0:
                print(f"ERROR out-of-range {c}: {len(bad)} rows")
                errors += 1

    # GFLOPS sanity if possible
    if {'n','time_s','gflops'}.issubset(df.columns):
        n = pd.to_numeric(df['n'], errors='coerce')
        t = pd.to_numeric(df['time_s'], errors='coerce')
        g = pd.to_numeric(df['gflops'], errors='coerce')
        g_expected = (2.0 * (n ** 3)) / (t * 1e9)
        mask = ~(g_expected.replace([pd.NA, pd.NaT], pd.NA).isna() | (t <= 0))
        mism = (~(abs(g[mask] - g_expected[mask]) <= 0.01 * g_expected[mask])).sum()
        if mism > 0:
            print(f"WARN gflops mismatch beyond 1% for {int(mism)} rows")

    # Trial continuity (per run_id/impl/n)
    if {'run_id','impl','n','trial'}.issubset(df.columns):
        groups = df.groupby(['run_id','impl','n'])
        for key, gdf in groups:
            trials = sorted(set(int(x) for x in gdf['trial'] if pd.notna(x)))
            if trials and trials[0] != 1:
                print(f"WARN trials start at {trials[0]} for {key}")
            # monotonic increasing increments of 1 (not strictly enforced; warning only)

    return errors


def main():
    ap = argparse.ArgumentParser(description='Validate raw CSV according to data plan')
    ap.add_argument('--raw', required=True, help='Path to raw.csv')
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        print(f"no such file: {args.raw}")
        return 2

    df = pd.read_csv(args.raw)
    errs = validate(df)
    if errs:
        print(f"Validation completed with {errs} error group(s)")
        return 1
    print("Validation OK")
    return 0


if __name__ == '__main__':
    sys.exit(main())
