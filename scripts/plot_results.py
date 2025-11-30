#!/usr/bin/env python3
"""
Plot matrix multiplication benchmark results
Usage: python plot_results.py [results_directory]
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_results(results_dir):
    """Load all CSV files from results directory"""
    csv_files = list(Path(results_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        return None
    
    print(f"Loading {len(csv_files)} CSV files...")
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    if not dfs:
        return None
    
    # Combine all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} benchmark results")
    return combined

def plot_execution_time(df, output_dir):
    """Plot execution time comparison"""
    plt.figure(figsize=(14, 8))
    
    # Group by implementation and matrix size
    pivot_data = df.pivot_table(
        values='total_time_ms',
        index='matrix_size',
        columns='implementation',
        aggfunc='mean'
    )
    
    pivot_data.plot(kind='bar', width=0.8)
    plt.title('Execution Time Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Matrix Size', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'execution_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_gflops(df, output_dir):
    """Plot GFLOPS comparison"""
    plt.figure(figsize=(14, 8))
    
    # Group by implementation and matrix size
    pivot_data = df.pivot_table(
        values='total_gflops',
        index='matrix_size',
        columns='implementation',
        aggfunc='mean'
    )
    
    pivot_data.plot(kind='bar', width=0.8)
    plt.title('Performance Comparison (GFLOPS)', fontsize=16, fontweight='bold')
    plt.xlabel('Matrix Size', fontsize=12)
    plt.ylabel('GFLOPS', fontsize=12)
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'gflops_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_speedup(df, output_dir, baseline='baseline'):
    """Plot speedup relative to baseline"""
    plt.figure(figsize=(14, 8))
    
    # Calculate speedup for each implementation
    baseline_df = df[df['implementation'] == baseline].copy()
    
    if baseline_df.empty:
        print(f"Warning: No baseline '{baseline}' found, skipping speedup plot")
        return
    
    speedup_data = []
    for impl in df['implementation'].unique():
        if impl == baseline:
            continue
        
        impl_df = df[df['implementation'] == impl].copy()
        
        for size in df['matrix_size'].unique():
            baseline_time = baseline_df[baseline_df['matrix_size'] == size]['total_time_ms'].mean()
            impl_time = impl_df[impl_df['matrix_size'] == size]['total_time_ms'].mean()
            
            if pd.notna(baseline_time) and pd.notna(impl_time) and impl_time > 0:
                speedup = baseline_time / impl_time
                speedup_data.append({
                    'implementation': impl,
                    'matrix_size': size,
                    'speedup': speedup
                })
    
    if not speedup_data:
        print("Warning: Could not calculate speedup")
        return
    
    speedup_df = pd.DataFrame(speedup_data)
    pivot_speedup = speedup_df.pivot_table(
        values='speedup',
        index='matrix_size',
        columns='implementation',
        aggfunc='mean'
    )
    
    pivot_speedup.plot(kind='bar', width=0.8)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
    plt.title(f'Speedup Relative to {baseline.upper()}', fontsize=16, fontweight='bold')
    plt.xlabel('Matrix Size', fontsize=12)
    plt.ylabel('Speedup (x)', fontsize=12)
    plt.legend(title='Implementation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'speedup_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_scaling(df, output_dir):
    """Plot scaling for OpenMP and MPI"""
    # OpenMP thread scaling
    openmp_data = df[df['implementation'].str.contains('openmp', case=False, na=False)]
    if not openmp_data.empty and 'threads' in openmp_data.columns:
        plt.figure(figsize=(12, 6))
        
        for size in sorted(openmp_data['matrix_size'].unique()):
            size_data = openmp_data[openmp_data['matrix_size'] == size]
            plt.plot(size_data['threads'], size_data['total_gflops'], 
                    marker='o', label=f'{size}x{size}')
        
        plt.title('OpenMP Thread Scaling', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Threads', fontsize=12)
        plt.ylabel('GFLOPS', fontsize=12)
        plt.legend(title='Matrix Size')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'openmp_scaling.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    # MPI process scaling
    mpi_data = df[df['implementation'].str.contains('mpi', case=False, na=False)]
    if not mpi_data.empty and 'processes' in mpi_data.columns:
        plt.figure(figsize=(12, 6))
        
        for size in sorted(mpi_data['matrix_size'].unique()):
            size_data = mpi_data[mpi_data['matrix_size'] == size]
            plt.plot(size_data['processes'], size_data['total_gflops'], 
                    marker='s', label=f'{size}x{size}')
        
        plt.title('MPI Process Scaling', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Processes', fontsize=12)
        plt.ylabel('GFLOPS', fontsize=12)
        plt.legend(title='Matrix Size')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'mpi_scaling.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    summary = df.groupby('implementation').agg({
        'total_gflops': ['mean', 'max'],
        'total_time_ms': 'mean'
    }).round(2)
    
    print(summary)
    print("="*60)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find most recent results directory
        results_base = Path("results/raw")
        if not results_base.exists():
            print("Error: results/raw directory not found")
            print("Usage: python plot_results.py [results_directory]")
            sys.exit(1)
        
        # Get most recent timestamp directory
        subdirs = [d for d in results_base.iterdir() if d.is_dir()]
        if not subdirs:
            print("Error: No results found in results/raw/")
            sys.exit(1)
        
        results_dir = max(subdirs, key=os.path.getmtime)
        print(f"Using most recent results: {results_dir}")
    
    # Create output directory for plots
    output_dir = Path("results/plots") / Path(results_dir).name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    df = load_results(results_dir)
    if df is None or df.empty:
        print("Error: No data loaded")
        sys.exit(1)
    
    print(f"\nGenerating plots in {output_dir}...")
    
    # Generate plots
    try:
        plot_execution_time(df, output_dir)
        plot_gflops(df, output_dir)
        plot_speedup(df, output_dir)
        plot_scaling(df, output_dir)
        
        # Print summary
        print_summary(df)
        
        print(f"\n✓ All plots saved to: {output_dir}/")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
