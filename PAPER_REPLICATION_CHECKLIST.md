# Paper Replication Checklist

This checklist maps each **table and figure** in the Experiments section to the code and notes any gaps.

---

## Setup and metrics

| Paper element | Code / note |
|---------------|-------------|
| **5 random seeds** | Paper states "mean ± standard deviation over 5 random seeds." Current code uses **3 seeds** `[42, 123, 456]` in `experiments_summary.py` and `run_4way_left_turn_sweep.py`. **Gap:** Either (a) change code to use 5 seeds for exact replication, or (b) state in the paper that 3 seeds were used. |
| **Total travel time** \(\sum_i T_i\) | Implemented in all runners; reported as `total_delay` or `total_agent_time`. ✓ |
| **IDEAL** \(\sum_i d_i^*\) | `compute_ideal_lane` (4-way), `compute_ideal_3x3` (3×3), `compute_ideal_graph` (Manhattan). ✓ |
| **Overhead** \(\frac{\sum_i T_i}{\sum_i d_i^*} - 1\) | Computed in table-printing code. ✓ |

---

## Four-way 2-in-1-out (Table 1, Fig. fourway, Fig. circular vs PIBT)

| Paper element | Code | Status |
|----------------|------|--------|
| **Table 1** (IDEAL 1419, SP 1879 (+32%), Circular 1721 (+21%), PIBT 1483, G-PIBT 1485, GLC 1486) | `experiments_summary.py` → `run_4way_experiments()`. **Note:** Paper LaTeX comment says "%SP-Mod is used here" for the SP row (1879). So the table reports **SP-Mod**, not raw SP. Code currently runs **SP** only. To replicate: run **SP-Mod** for that row (e.g. add SP-Mod to 4-way and report it as SP, or expose both). | **Gap:** Use SP-Mod for the "SP" row to get 1879 (+32%). |
| **Fig. fourway_inbound / at_intersection / outbound** | Illustrative snapshots under "PIBT-style conflict resolution." Can be produced by running `visualize_intersection_fourway.py` with `--method pibt` and capturing frames (or adding a "save snapshot" option). | **Gap:** No script currently exports PNG snapshots; use screenshots or add export. |
| **Fig. 4way_circular_vs_pibt_overhead.png** | `plot_4way_circular_vs_pibt.py` → outputs `4way_circular_vs_pibt_overhead.png`. ✓ | ✓ |

---

## 3×3 directed grid (Table 2, Fig. 3by3scenario)

| Paper element | Code | Status |
|----------------|------|--------|
| **Table 2** (Agents 8, 16, 24, 32, 40, 60; IDEAL, SP, PIBT, G-PIBT, GLC) | `experiments_summary.py` → `run_3x3_limited()`. Default `agent_counts = [8, 12, 16, 20, 24]`. Paper uses **8, 16, 24, 32, 40, 60**. | **Gap:** Set agent counts to `[8, 16, 24, 32, 40, 60]` for exact replication. |
| **Fig. 3by3scenario.png** (origins/destinations) | Not in paper-release. Main repo has `visualize_3x3_grid.py` (not copied to paper-release). A small script that plots the 3×3 grid and sampled OD would reproduce this. | **Gap:** Add a script to plot 3×3 scenario (or include `visualize_3x3_grid.py` and document). |

---

## City-scale Manhattan (Table 3, Fig. manhattan_map_od, Fig. manhattan_runtime / accumulated_delay)

| Paper element | Code | Status |
|----------------|------|--------|
| **Table 3** (100 / 1000 / 10000 agents; IDEAL, SP, TAP, PIBT, G-PIBT, GLC at 100; IDEAL, SP, PIBT, GLC at 1k and 10k) | `run_manhattan_large_scale.py` with `--agents 100/1000/10000`, `--methods ideal,sp,tap,pibt,gpibt,glc` (100) or `ideal,sp,pibt,glc` (1k, 10k). ✓ | ✓ |
| **Fig. manhattan_map_od.png** | Map with OD sample. `visualize_manhattan.py` in main repo can produce such maps; it is **not** in paper-release. | **Gap:** Either add a minimal map-plot script or document "figure produced from visualize_manhattan.py (not included)." |
| **Fig. manhattan_runtime.png, manhattan_accumulated_delay.png** | `plot_manhattan_progress.py` with CSVs from 10k run: `--out-delay manhattan_accumulated_delay.png` and `--out-runtime manhattan_runtime.png`. Expects CSVs with columns `method`, `step`, `accumulated_delay`, `runtime_s`. ✓ | ✓ |

---

## Proof-of-concept

| Paper element | Code | Status |
|----------------|------|--------|
| **Table: Realistic POC** (IDEAL, SP, PIBT, SP-PIBT; 100 agents) | `experiments/manhattan_realistic_poc.py`. ✓ | ✓ |
| **Table: Mixed AV/human** (sweep \(p\); total travel time, completion, violations) | `experiments/manhattan_mixed_autonomy_poc.py`. ✓ | ✓ |

---

## Summary of gaps to fix for exact replication

1. **Seeds:** Use **5 seeds** in experiments if the paper states "5 random seeds"; otherwise align the paper text with 3 seeds.
2. **Four-way table:** The "SP" row (1879, +32%) is **SP-Mod** in the paper. Add SP-Mod to the 4-way experiment and report it so the table matches (or label the row "SP-Mod" in the paper).
3. **3×3 table:** Use agent counts **8, 16, 24, 32, 40, 60** (not 8, 12, 16, 20, 24).
4. **Figures:**  
   - Four-way snapshots: add PNG export to the 4-way visualizer or document screenshot workflow.  
   - 3×3 scenario: add a small plot script or include `visualize_3x3_grid.py`.  
   - Manhattan map: add a map-plot script or document source.

All other experiments (4-way left-turn figure, Manhattan tables and progress figures, both POC tables) are covered by the current paper-release code.
