# Code for Paper Reproduction

This repository contains code to reproduce the experiments in the accompanying paper. No author or institutional information is included for double-blind review.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and the dependencies listed in `requirements.txt` (NumPy, matplotlib, pygame, NetworkX, OSMnx, pandas, etc.).

## Experiments

### 1. Four-way 2-in-1-out intersection

Default: 24 agents, 30% left turns, 35% straight, 35% right.

```bash
# Main comparison: IDEAL, SP, PIBT, G-PIBT, GLC, Circular
python experiments_summary.py

# 4-way only
python -c "from experiments_summary import run_4way_only; run_4way_only()"

# PIBT vs Circular w.r.t. % left-turn
python run_4way_left_turn_sweep.py

# Overhead figure (PIBT vs Circular)
python plot_4way_circular_vs_pibt.py

# Interactive visualization (methods: gsp, pibt, circular, sp)
python visualize_intersection_fourway.py --agents 24 --method pibt

# Save animation to GIF (e.g. GLC, 24 agents)
python visualize_intersection_fourway.py --agents 24 --method glc --save-gif 4way_glc_24agents.gif
```

### 2. 3×3 small road network

```bash
# IDEAL, SP, PIBT, G-PIBT, GLC w.r.t. #agents
python -c "from experiments_summary import run_3x3_only; run_3x3_only()"
```

### 3. City-scale Manhattan

**Included in this repo:** `sample_od.parquet` (small OD sample) and `.manhattan_cache/` (Manhattan road graph + zone centroids). You can run Manhattan experiments without downloading the full NYC TLC dataset.

```bash
# Using the included sample (run from repo root)
# 100 agents: IDEAL, SP, TAP, PIBT, G-PIBT, GLC
python run_manhattan_large_scale.py sample_od.parquet --agents 100 --methods ideal,sp,tap,pibt,gpibt,glc --output-csv manhattan_100_progress.csv --progress-interval 1

# 1000 agents: IDEAL, SP, PIBT, GLC
python run_manhattan_large_scale.py sample_od.parquet --agents 1000 --methods ideal,sp,pibt,glc --output-csv manhattan_1000_progress.csv --progress-interval 1

# 10000 agents: IDEAL, SP, PIBT, GLC
python run_manhattan_large_scale.py sample_od.parquet --agents 10000 --methods ideal,sp,pibt,glc --output-csv manhattan_10000_progress.csv --progress-interval 1

# Plot progress (travel time and runtime vs step)
python plot_manhattan_progress.py manhattan_1000_progress.csv --out-delay delay_1000.png --out-runtime runtime_1000.png
```

For full-scale reproduction with the complete NYC TLC Yellow taxi file (e.g. `yellow_tripdata_2024-01.parquet`), download it from [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and pass its path instead of `sample_od.parquet`. The script uses `--cache-dir .manhattan_cache` by default (graph and OD cache).

### 4. Proof-of-concept experiments

See `experiments/README_realistic_poc.md` and `experiments/README_mixed_autonomy_poc.md` for the realistic traffic and mixed-autonomy PoCs.

## Method names

- **GLC:** Our method (graph shortest paths + priority-based collision resolution).
- **PIBT:** Paper-PIBT (time-based priority) is used wherever “PIBT” is reported.

## Replicating paper tables and figures

See **PAPER_REPLICATION_CHECKLIST.md** for a mapping from each table and figure in the paper to the script and command that produces it. Experiments use 5 random seeds; the four-way table uses SP-Mod for the "SP" row; the 3×3 table uses agent counts 8, 16, 24, 32, 40, 60.

## License

MIT License.
