# Visualization & Data Management

Implements the plan in `visualization/plan.txt`:
- Append-only daily CSVs under `results/YYYY-MM-DD/raw.csv`
- Daily `env.json` snapshot
- Validation, Parquet/SQLite promotion, and figure generation with meta

## Scripts
- `validate_raw.py` — schema/type checks and sanity validations on a raw CSV
- `ingest.py` — load a day folder, validate, enrich, and write Parquet
- `to_sqlite.py` — load multiple days into a single SQLite `results.db`
- `make_figures.py` — generate standard figures and `*.meta.json`

## Quick start
```
# Validate a day's raw.csv
python visualization/validate_raw.py --day results/2025-08-14

# Convert to Parquet
python visualization/ingest.py --day results/2025-08-14 --out results/2025-08-14/raw.parquet

# Build/append SQLite from multiple days
python visualization/to_sqlite.py --days results/2025-08-14 results/2025-08-15 --db results/results.db

# Make figures
python visualization/make_figures.py --raw results/2025-08-14/raw.csv --outdir results/2025-08-14/figures
```

## Requirements
See `visualization/requirements.txt` (pandas, pyarrow, matplotlib, seaborn, sqlalchemy).
