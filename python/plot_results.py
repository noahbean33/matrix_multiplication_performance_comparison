#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_csvs(paths):
    frames = []
    for p in paths:
        for f in glob.glob(p):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"skip {f}: {e}")
    if not frames:
        raise SystemExit("no input CSVs found")
    df = pd.concat(frames, ignore_index=True)
    # best-effort typing
    for col in ["n", "threads"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_time(df, outdir):
    plt.figure(figsize=(7,4))
    for key, g in df.groupby([df.get("impl", "impl"), df.get("variant", "variant")]):
        label = "-".join([str(k) for k in (key if isinstance(key, tuple) else (key,))])
        g2 = g.sort_values("n")
        plt.plot(g2["n"], g2["time_seconds"], marker="o", label=label)
    plt.xlabel("n")
    plt.ylabel("Time (s)")
    plt.title("Matrix Multiply Time vs n")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "time_vs_n.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


def plot_gflops(df, outdir):
    if "gflops" not in df.columns:
        # compute if missing
        if {"n", "time_seconds"}.issubset(df.columns):
            df = df.copy()
            df["gflops"] = (2.0 * (df["n"]**3)) / df["time_seconds"] / 1e9
        else:
            return
    plt.figure(figsize=(7,4))
    for key, g in df.groupby([df.get("impl", "impl"), df.get("variant", "variant")]):
        label = "-".join([str(k) for k in (key if isinstance(key, tuple) else (key,))])
        g2 = g.sort_values("n")
        plt.plot(g2["n"], g2["gflops"], marker="o", label=label)
    plt.xlabel("n")
    plt.ylabel("GFLOP/s")
    plt.title("Matrix Multiply GFLOP/s vs n")
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "gflops_vs_n.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot results from CSVs")
    ap.add_argument("--inputs", nargs="+", default=["data/results/*.csv"]) 
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()
    df = load_csvs(args.inputs)
    plot_time(df, args.outdir)
    plot_gflops(df, args.outdir)


if __name__ == "__main__":
    main()
