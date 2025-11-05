#!/usr/bin/env python3
"""
Generate speedup comparison plots for matrix multiplication implementations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_results(results_dir='../../results/raw'):
    """Load all CSV results"""
    results_path = Path(results_dir)
    csv_files = list(results_path.glob('*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files found in {results_path}")
        sys.exit(1)
    
    dfs = [pd.read_csv(f) for f in csv_files]
    return pd.concat(dfs, ignore_index=True)

def calculate_speedup(df, baseline='naive'):
    """Calculate speedup relative to baseline"""
    baseline_df = df[df['implementation'] == baseline].copy()
    baseline_grouped = baseline_df.groupby('matrix_size')['execution_time_ms'].mean()
    
    speedup_data = []
    for impl in df['implementation'].unique():
        impl_df = df[df['implementation'] == impl].copy()
        impl_grouped = impl_df.groupby('matrix_size')['execution_time_ms'].mean()
        
        for size in impl_grouped.index:
            if size in baseline_grouped.index:
                speedup = baseline_grouped[size] / impl_grouped[size]
                speedup_data.append({
                    'implementation': impl,
                    'matrix_size': size,
                    'speedup': speedup
                })
    
    return pd.DataFrame(speedup_data)

def plot_speedup_comparison(df, output_dir='../../results/plots'):
    """Generate speedup comparison plot"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    speedup_df = calculate_speedup(df)
    
    plt.figure(figsize=(14, 8))
    
    # Plot speedup for each implementation
    for impl in speedup_df['implementation'].unique():
        if impl == 'naive':
            continue  # Skip baseline
        
        impl_data = speedup_df[speedup_df['implementation'] == impl]
        impl_data = impl_data.sort_values('matrix_size')
        
        plt.plot(
            impl_data['matrix_size'],
            impl_data['speedup'],
            marker='o',
            linewidth=2,
            markersize=8,
            label=impl
        )
    
    plt.xlabel('Matrix Size (N×N)', fontsize=14, fontweight='bold')
    plt.ylabel('Speedup vs Naive', fontsize=14, fontweight='bold')
    plt.title('Matrix Multiplication Speedup Comparison', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    
    # Add horizontal line at speedup=1
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    plt.tight_layout()
    
    output_file = output_path / 'speedup_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # Also save as PDF for publication
    plt.savefig(output_path / 'speedup_comparison.pdf', bbox_inches='tight')
    
    plt.close()

def plot_gflops_comparison(df, output_dir='../../results/plots'):
    """Generate GFLOPS comparison plot"""
    output_path = Path(output_dir)
    
    plt.figure(figsize=(14, 8))
    
    # Group by implementation and matrix size
    grouped = df.groupby(['implementation', 'matrix_size'])['gflops'].mean().reset_index()
    
    for impl in grouped['implementation'].unique():
        impl_data = grouped[grouped['implementation'] == impl]
        impl_data = impl_data.sort_values('matrix_size')
        
        plt.plot(
            impl_data['matrix_size'],
            impl_data['gflops'],
            marker='s',
            linewidth=2,
            markersize=8,
            label=impl
        )
    
    plt.xlabel('Matrix Size (N×N)', fontsize=14, fontweight='bold')
    plt.ylabel('Performance (GFLOPS)', fontsize=14, fontweight='bold')
    plt.title('Matrix Multiplication Performance (GFLOPS)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.tight_layout()
    
    output_file = output_path / 'gflops_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.savefig(output_path / 'gflops_comparison.pdf', bbox_inches='tight')
    
    plt.close()

def plot_execution_time(df, output_dir='../../results/plots'):
    """Generate execution time comparison plot"""
    output_path = Path(output_dir)
    
    plt.figure(figsize=(14, 8))
    
    grouped = df.groupby(['implementation', 'matrix_size'])['execution_time_ms'].mean().reset_index()
    
    for impl in grouped['implementation'].unique():
        impl_data = grouped[grouped['implementation'] == impl]
        impl_data = impl_data.sort_values('matrix_size')
        
        plt.plot(
            impl_data['matrix_size'],
            impl_data['execution_time_ms'],
            marker='^',
            linewidth=2,
            markersize=8,
            label=impl
        )
    
    plt.xlabel('Matrix Size (N×N)', fontsize=14, fontweight='bold')
    plt.ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    plt.title('Matrix Multiplication Execution Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.yscale('log')
    
    plt.tight_layout()
    
    output_file = output_path / 'execution_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.savefig(output_path / 'execution_time.pdf', bbox_inches='tight')
    
    plt.close()

def main():
    print("=== Generating Speedup Visualizations ===\n")
    
    # Load data
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} measurements\n")
    
    # Generate plots
    print("Generating speedup comparison plot...")
    plot_speedup_comparison(df)
    
    print("Generating GFLOPS comparison plot...")
    plot_gflops_comparison(df)
    
    print("Generating execution time plot...")
    plot_execution_time(df)
    
    print("\n=== Visualization Complete ===")

if __name__ == '__main__':
    main()
