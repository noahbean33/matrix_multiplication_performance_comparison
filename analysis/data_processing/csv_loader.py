#!/usr/bin/env python3
"""
Utility functions for loading and processing CSV benchmark data
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional

def load_benchmark_data(
    results_dir: str = '../../results/raw',
    implementation: Optional[str] = None,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load benchmark CSV data with optional filtering
    
    Args:
        results_dir: Directory containing CSV files
        implementation: Filter by implementation name (e.g., 'naive', 'openmp')
        min_size: Minimum matrix size to include
        max_size: Maximum matrix size to include
    
    Returns:
        Combined pandas DataFrame with all benchmark data
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    # Find all CSV files
    csv_files = list(results_path.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_path}")
    
    # Load all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file.name}: {e}")
    
    if not dfs:
        raise ValueError("No data could be loaded from CSV files")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Apply filters
    if implementation:
        combined_df = combined_df[combined_df['implementation'] == implementation]
    
    if min_size:
        combined_df = combined_df[combined_df['matrix_size'] >= min_size]
    
    if max_size:
        combined_df = combined_df[combined_df['matrix_size'] <= max_size]
    
    # Convert timestamp if present
    if 'timestamp' in combined_df.columns:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
    
    return combined_df

def get_implementations(df: pd.DataFrame) -> List[str]:
    """Get list of unique implementations in the dataset"""
    return sorted(df['implementation'].unique())

def get_matrix_sizes(df: pd.DataFrame) -> List[int]:
    """Get list of unique matrix sizes in the dataset"""
    return sorted(df['matrix_size'].unique())

def filter_outliers(df: pd.DataFrame, column: str = 'execution_time_ms', threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method
    
    Args:
        df: Input dataframe
        column: Column to check for outliers
        threshold: Z-score threshold (default: 3.0 standard deviations)
    
    Returns:
        Filtered dataframe with outliers removed
    """
    from scipy import stats
    
    z_scores = stats.zscore(df[column])
    abs_z_scores = abs(z_scores)
    filtered_df = df[abs_z_scores < threshold]
    
    removed = len(df) - len(filtered_df)
    if removed > 0:
        print(f"Removed {removed} outliers from {len(df)} measurements ({100*removed/len(df):.1f}%)")
    
    return filtered_df

def aggregate_runs(df: pd.DataFrame, group_by: List[str] = None) -> pd.DataFrame:
    """
    Aggregate multiple runs with statistics
    
    Args:
        df: Input dataframe
        group_by: Columns to group by (default: implementation and matrix_size)
    
    Returns:
        Aggregated dataframe with mean, std, min, max
    """
    if group_by is None:
        group_by = ['implementation', 'matrix_size']
    
    agg_dict = {
        'execution_time_ms': ['mean', 'std', 'min', 'max', 'count'],
        'gflops': ['mean', 'std', 'max']
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
    
    grouped = df.groupby(group_by).agg(agg_dict).reset_index()
    
    # Flatten multi-level column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    return grouped

if __name__ == '__main__':
    # Test the loader
    print("Testing CSV loader...")
    
    try:
        df = load_benchmark_data()
        print(f"\nLoaded {len(df)} measurements")
        print(f"Implementations: {get_implementations(df)}")
        print(f"Matrix sizes: {get_matrix_sizes(df)}")
        print("\nFirst few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
