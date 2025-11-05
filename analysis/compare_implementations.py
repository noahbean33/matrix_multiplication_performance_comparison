#!/usr/bin/env python3
"""
Compare different matrix multiplication implementations
Generates comparison reports and statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_results(results_dir='../../results/raw'):
    """Load all CSV results from the raw directory"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        sys.exit(1)
    
    csv_files = list(results_path.glob('*.csv'))
    
    if not csv_files:
        print(f"Error: No CSV files found in {results_path}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load and combine all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"  Loaded: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
    
    if not dfs:
        print("Error: No data loaded")
        sys.exit(1)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(combined_df)}")
    
    return combined_df

def calculate_statistics(df):
    """Calculate statistics for each implementation and matrix size"""
    grouped = df.groupby(['implementation', 'matrix_size'])
    
    stats = grouped.agg({
        'execution_time_ms': ['mean', 'std', 'min', 'max'],
        'gflops': ['mean', 'std', 'max']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    return stats

def calculate_speedup(df, baseline='naive'):
    """Calculate speedup relative to baseline implementation"""
    # Get baseline performance
    baseline_df = df[df['implementation'] == baseline].copy()
    baseline_grouped = baseline_df.groupby('matrix_size')['execution_time_ms'].mean()
    
    # Calculate speedup for each implementation
    results = []
    for impl in df['implementation'].unique():
        impl_df = df[df['implementation'] == impl].copy()
        impl_grouped = impl_df.groupby('matrix_size')['execution_time_ms'].mean()
        
        for size in impl_grouped.index:
            if size in baseline_grouped.index:
                speedup = baseline_grouped[size] / impl_grouped[size]
                results.append({
                    'implementation': impl,
                    'matrix_size': size,
                    'speedup': speedup
                })
    
    return pd.DataFrame(results)

def generate_report(df, output_dir='../../results/reports'):
    """Generate comparison report"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report_file = output_path / 'comparison_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Matrix Multiplication Performance Comparison Report\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        stats = calculate_statistics(df)
        f.write(stats.to_markdown(index=False))
        f.write("\n\n")
        
        # Speedup analysis
        f.write("## Speedup Analysis (vs Naive)\n\n")
        speedup_df = calculate_speedup(df)
        speedup_pivot = speedup_df.pivot(
            index='matrix_size',
            columns='implementation',
            values='speedup'
        )
        f.write(speedup_pivot.to_markdown())
        f.write("\n\n")
        
        # Best performers
        f.write("## Best Performers by Matrix Size\n\n")
        best = df.loc[df.groupby('matrix_size')['gflops'].idxmax()]
        f.write(best[['matrix_size', 'implementation', 'gflops']].to_markdown(index=False))
        f.write("\n")
    
    print(f"\nReport saved to: {report_file}")
    
    # Also save as CSV
    stats.to_csv(output_path / 'statistics.csv', index=False)
    speedup_df.to_csv(output_path / 'speedup.csv', index=False)

def main():
    print("=== Matrix Multiplication Implementation Comparison ===\n")
    
    # Load data
    df = load_results()
    
    # Display basic info
    print("\nImplementations found:")
    for impl in df['implementation'].unique():
        count = len(df[df['implementation'] == impl])
        print(f"  - {impl}: {count} measurements")
    
    print("\nMatrix sizes tested:")
    for size in sorted(df['matrix_size'].unique()):
        print(f"  - {size}x{size}")
    
    # Generate reports
    print("\nGenerating comparison report...")
    generate_report(df)
    
    print("\n=== Analysis Complete ===")

if __name__ == '__main__':
    main()
