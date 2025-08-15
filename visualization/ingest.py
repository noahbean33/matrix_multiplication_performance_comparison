#!/usr/bin/env python3
import argparse
import json
import os
import sys
import pandas as pd

from validate_raw import validate


def load_day(day_dir: str):
    raw_csv = os.path.join(day_dir, 'raw.csv')
    env_json = os.path.join(day_dir, 'env.json')
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f'missing {raw_csv}')
    df = pd.read_csv(raw_csv)

    env = {}
    if os.path.exists(env_json):
        with open(env_json, 'r') as f:
            env = json.load(f)
    return df, env, raw_csv, env_json


def enrich_with_env(df: pd.DataFrame, env: dict) -> pd.DataFrame:
    if not env:
        return df
    # Add a few top-level env fields if absent; avoid overwriting existing columns
    add = {}
    add['env_datetime_iso'] = env.get('datetime_iso')
    add['env_hostname'] = env.get('hostname')
    add['env_cpu_model'] = (env.get('cpu') or {}).get('model')
    add['env_gpu_model'] = (env.get('gpu') or {}).get('model')
    add['env_cuda_version'] = (env.get('cuda') or {}).get('version')
    add['env_mpi_vendor'] = (env.get('mpi') or {}).get('vendor')
    add['env_mpi_version'] = (env.get('mpi') or {}).get('version')
    for k, v in add.items():
        if k not in df.columns:
            df[k] = v
    return df


def main():
    ap = argparse.ArgumentParser(description='Ingest a day folder: validate, enrich, write Parquet')
    ap.add_argument('--day', required=True, help='Path to results/YYYY-MM-DD folder containing raw.csv and env.json')
    ap.add_argument('--out', required=False, help='Output Parquet path (default: <day>/raw.parquet)')
    args = ap.parse_args()

    df, env, raw_csv, env_json = load_day(args.day)

    # Validate
    errs = validate(df.copy())
    if errs:
        print(f'[ingest] validation reported {errs} error(s); continuing to write for inspection', file=sys.stderr)

    df = enrich_with_env(df, env)

    out_parquet = args.out or os.path.join(args.day, 'raw.parquet')
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    try:
        df.to_parquet(out_parquet, index=False)
    except Exception as e:
        print('Parquet write failed (pyarrow/fastparquet missing?). Falling back to CSV copy.', file=sys.stderr)
        fallback = os.path.join(args.day, 'raw_validated.csv')
        df.to_csv(fallback, index=False)
        print(f'[ingest] wrote fallback CSV -> {fallback}')
        return 0

    print(f'[ingest] wrote Parquet -> {out_parquet}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
