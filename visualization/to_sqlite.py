#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
from typing import List
import pandas as pd

from validate_raw import validate


def load_day_df(day_dir: str) -> pd.DataFrame:
    parq = os.path.join(day_dir, 'raw.parquet')
    csvp = os.path.join(day_dir, 'raw.csv')
    if os.path.exists(parq):
        df = pd.read_parquet(parq)
    elif os.path.exists(csvp):
        df = pd.read_csv(csvp)
    else:
        raise FileNotFoundError(f'No raw.csv or raw.parquet in {day_dir}')
    return df


def load_env(day_dir: str) -> pd.DataFrame:
    envp = os.path.join(day_dir, 'env.json')
    if not os.path.exists(envp):
        return pd.DataFrame()
    with open(envp, 'r') as f:
        env = json.load(f)
    # Flatten a few fields and keep raw json for reference
    flat = {
        'day': os.path.basename(day_dir.rstrip("/\\")),
        'datetime_iso': env.get('datetime_iso'),
        'hostname': env.get('hostname'),
        'slurm_job_id': env.get('slurm_job_id'),
        'cpu_model': (env.get('cpu') or {}).get('model'),
        'gpu_model': (env.get('gpu') or {}).get('model'),
        'gpu_count': (env.get('gpu') or {}).get('count'),
        'memory_gb': env.get('memory_gb'),
        'compiler_name': (env.get('compiler') or {}).get('name'),
        'compiler_version': (env.get('compiler') or {}).get('version'),
        'blas_vendor': (env.get('blas') or {}).get('vendor'),
        'blas_version': (env.get('blas') or {}).get('version'),
        'cuda_version': (env.get('cuda') or {}).get('version'),
        'mpi_vendor': (env.get('mpi') or {}).get('vendor'),
        'mpi_version': (env.get('mpi') or {}).get('version'),
        'os_name': (env.get('os') or {}).get('name'),
        'os_version': (env.get('os') or {}).get('version'),
        'git_commit': env.get('git_commit'),
        'env_json': json.dumps(env),
    }
    return pd.DataFrame([flat])


def upsert(df: pd.DataFrame, con: sqlite3.Connection, table: str):
    # Simple append for now; dedupe can be added later based on run_id
    df.to_sql(table, con, if_exists='append', index=False)


def main():
    ap = argparse.ArgumentParser(description='Aggregate days into a SQLite database')
    ap.add_argument('--days', nargs='+', required=True, help='List of results/YYYY-MM-DD directories')
    ap.add_argument('--db', required=True, help='Path to SQLite DB to create/append')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    con = sqlite3.connect(args.db)

    try:
        for day in args.days:
            print(f'[sqlite] processing {day}')
            df = load_day_df(day)
            errs = validate(df.copy())
            if errs:
                print(f'[sqlite] validation had {errs} error(s); ingesting anyway for audit')
            envdf = load_env(day)
            df['day'] = os.path.basename(day.rstrip('/\\'))
            upsert(df, con, 'runs')
            if not envdf.empty:
                upsert(envdf, con, 'env')
        print(f'[sqlite] wrote -> {args.db}')
    finally:
        con.close()


if __name__ == '__main__':
    main()
