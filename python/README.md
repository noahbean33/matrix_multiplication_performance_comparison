# Python Analysis and Visualization

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure

### `analysis/`
Scripts for processing and analyzing benchmark data:
- `compare_implementations.py` - Compare different implementation approaches
- `scalability_analysis.py` - Analyze strong and weak scaling
- `statistical_analysis.py` - Statistical tests and confidence intervals

### `visualization/`
Scripts for generating plots and figures:
- `plot_speedup.py` - Generate speedup comparison plots
- `plot_efficiency.py` - Plot parallel efficiency
- `plot_heatmaps.py` - Create performance heatmaps
- `plot_scalability.py` - Scalability curves

### `data_processing/`
Utilities for data handling:
- `csv_loader.py` - Load and parse CSV benchmark data
- `data_cleaner.py` - Clean and preprocess data
- `metrics_calculator.py` - Calculate derived metrics (speedup, efficiency, etc.)

## Usage Examples

### Load and Analyze Data

```python
from data_processing.csv_loader import load_benchmark_data
from analysis.compare_implementations import compare_all

# Load data
data = load_benchmark_data('../results/raw/')

# Generate comparison report
compare_all(data, output_dir='../results/reports/')
```

### Generate Plots

```python
from visualization.plot_speedup import plot_speedup_comparison

# Create speedup plot
plot_speedup_comparison(
    data,
    baseline='naive',
    output_file='../results/plots/speedup_comparison.png'
)
```

## Output Formats

- **Plots**: PNG, PDF, SVG (publication-ready)
- **Reports**: Markdown, CSV, LaTeX tables
- **Statistics**: JSON, CSV

## Dependencies

- pandas - Data manipulation
- numpy - Numerical operations
- matplotlib - Basic plotting
- seaborn - Statistical visualization
- scipy - Statistical analysis
