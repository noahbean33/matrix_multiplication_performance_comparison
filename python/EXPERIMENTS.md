# Python Experiments: Baselines, CSV, and Plots

Use this folder for quick baselines, validation, and analysis.

## Goals
- Provide pure-Python and NumPy baselines
- Emit unified CSV to `data/results/`
- Simple validation vs NumPy reference
- Produce quick plots for reports

## Scripts
- `bench.py` — sweep sizes and variants (python|numpy), write CSV
- `validate.py` — compare results vs NumPy reference, report errors
- `plot_results.py` — read CSVs and generate figures into `plots/`
- `requirements.txt` — minimal deps for analysis

## CSV Schema
`n,time_seconds,variant,threads,impl`
- `variant` — algorithm label (e.g., naive)
- `impl` — python|numpy

## Threading Notes (Orca)
Set BLAS threads to 1 for stable baselines:
```
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Examples
- Bench NumPy single-thread on a sweep:
```
OPENBLAS_NUM_THREADS=1 python python/bench.py --impl numpy \
  --min-n 256 --max-n 2048 --step 256 --repeats 5 \
  --outfile data/results/py_numpy_sweep.csv
```
- Validate vs NumPy:
```
python python/validate.py --n 512
```
- Plot results:
```
python python/plot_results.py --inputs data/results/*.csv --outdir plots/
```
