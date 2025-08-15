#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_meta(out_path: str, source: str, query: str):
    meta = {
        'source': source,
        'query': query,
        'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
    }
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_df(path: str) -> pd.DataFrame:
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def add_gflops(df: pd.DataFrame) -> pd.DataFrame:
    if {'n','time_s'}.issubset(df.columns) and 'gflops' not in df.columns:
        df = df.copy()
        df['gflops'] = (2.0 * (df['n']**3)) / df['time_s'] / 1e9
    return df


def fig_time_vs_n(df: pd.DataFrame, outdir: str, src: str, query: str):
    df = df.sort_values('n')
    plt.figure(figsize=(7,4))
    sns.lineplot(data=df, x='n', y='time_s', hue='impl', style='variant', markers=True)
    plt.title('Time vs n')
    plt.grid(True, alpha=0.3)
    ensure_dir(outdir)
    out = os.path.join(outdir, 'time_vs_n.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    save_meta(os.path.join(outdir, 'time_vs_n.meta.json'), src, query)
    print(f'wrote {out}')


def fig_gflops_vs_n(df: pd.DataFrame, outdir: str, src: str, query: str):
    df = add_gflops(df).sort_values('n')
    plt.figure(figsize=(7,4))
    sns.lineplot(data=df, x='n', y='gflops', hue='impl', style='variant', markers=True)
    plt.title('GFLOP/s vs n')
    plt.grid(True, alpha=0.3)
    ensure_dir(outdir)
    out = os.path.join(outdir, 'gflops_vs_n.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    save_meta(os.path.join(outdir, 'gflops_vs_n.meta.json'), src, query)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description='Generate standard figures for a raw dataset')
    ap.add_argument('--raw', required=True, help='Path to raw.csv or raw.parquet')
    ap.add_argument('--outdir', required=True, help='Directory to write figures to')
    ap.add_argument('--query', default="")
    args = ap.parse_args()

    df = load_df(args.raw)
    if args.query:
        df = df.query(args.query)
    fig_time_vs_n(df, args.outdir, args.raw, args.query)
    fig_gflops_vs_n(df, args.outdir, args.raw, args.query)


if __name__ == '__main__':
    main()
