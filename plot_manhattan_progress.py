#!/usr/bin/env python3
"""Plot Manhattan progress: accumulated_delay and runtime_s vs step.
Uses three CSVs: GLC (SP-PIBT 10k), PIBT (Paper-PIBT 10k), SP (1k agents).
Extends each curve to max step (flat plateau = simulation ended)."""
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Default CSV paths: GLC, PIBT, SP
DEFAULT_CSV_GLC = "manhattan_sppibt_10000_progress.csv"
DEFAULT_CSV_PIBT = "manhattan_paperpibt_progress.csv"
DEFAULT_CSV_SP = "manhattan_progress_10000agents.csv"


def load_three_curves(csv_glc, csv_pibt, csv_sp):
    """Load three CSVs and return a combined dataframe with method labels GLC, PIBT, SP."""
    dfs = []
    if csv_glc and os.path.isfile(csv_glc):
        df = pd.read_csv(csv_glc)
        df["method"] = "GLC"
        dfs.append(df)
    if csv_pibt and os.path.isfile(csv_pibt):
        df = pd.read_csv(csv_pibt)
        df["method"] = "PIBT"
        dfs.append(df)
    if csv_sp and os.path.isfile(csv_sp):
        df = pd.read_csv(csv_sp)
        # File may contain multiple methods (SP, PIBT, SP-PIBT); keep only SP rows
        df = df[df["method"] == "SP"].copy()
        df["method"] = "SP"
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("None of the CSV files found. Need at least one of: GLC, PIBT, SP.")
    return pd.concat(dfs, ignore_index=True)


def extend_to_max_step(df, methods):
    """Extend each method's data to the global max step with flat values (simulation ended)."""
    max_step = df["step"].max()
    extended = []
    completion_steps = {}
    for m in methods:
        sub = df[df["method"] == m].sort_values("step")
        if len(sub) == 0:
            continue
        last = sub.iloc[-1]
        last_step = last["step"]
        completion_steps[m] = last_step
        if last_step < max_step:
            ext_row = last.copy()
            ext_row["step"] = max_step
            extended.append(pd.concat([sub, pd.DataFrame([ext_row])], ignore_index=True))
        else:
            extended.append(sub.copy())
    if not extended:
        return df, completion_steps
    return pd.concat(extended, ignore_index=True), completion_steps


LINESTYLES = ["-", "--", "-."]
# Order: SP, PIBT, GLC. GLC = red.
COLORS = ["#1f77b4", "#2ca02c", "#d62728"]  # blue, green, red


def plot_with_completion_markers(ax, df_ext, methods, ycol, ylabel, completion_steps):
    """Plot with flat extension and completion markers (★ = all agents completed)."""
    from matplotlib.lines import Line2D
    for i, m in enumerate(methods):
        sub = df_ext[df_ext["method"] == m].sort_values("step")
        if len(sub) == 0:
            continue
        ls = LINESTYLES[i % len(LINESTYLES)]
        col = COLORS[i % len(COLORS)]
        ax.plot(sub["step"], sub[ycol], linewidth=2, color=col, linestyle=ls)
        comp_step = completion_steps.get(m)
        if comp_step is not None:
            row = sub[sub["step"] == comp_step]
            if len(row) > 0:
                yval = row[ycol].iloc[0]
                ax.scatter([comp_step], [yval], marker="*", s=120, color=col,
                           zorder=5, edgecolors="black", linewidths=0.5)
    handles = [
        Line2D([0], [0], color=COLORS[i % len(COLORS)], linestyle=LINESTYLES[i % len(LINESTYLES)],
               linewidth=2, label=m) for i, m in enumerate(methods) if m in df_ext["method"].values
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description="Plot Manhattan progress: GLC, PIBT, SP from separate CSVs")
    ap.add_argument("--csv-glc", default=DEFAULT_CSV_GLC, help="CSV for GLC curve (SP-PIBT 10k)")
    ap.add_argument("--csv-pibt", default=DEFAULT_CSV_PIBT, help="CSV for PIBT curve (Paper-PIBT 10k)")
    ap.add_argument("--csv-sp", default=DEFAULT_CSV_SP, help="CSV for SP curve (10k agents)")
    ap.add_argument("--out-delay", default="manhattan_accumulated_delay.png", help="Output figure for accumulated delay")
    ap.add_argument("--out-runtime", default="manhattan_runtime.png", help="Output figure for runtime")
    args = ap.parse_args()

    df = load_three_curves(args.csv_glc, args.csv_pibt, args.csv_sp)
    methods = [m for m in ["SP", "PIBT", "GLC"] if m in df["method"].values]
    df_ext, completion_steps = extend_to_max_step(df, methods)

    # Figure 1: step vs accumulated_delay
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    plot_with_completion_markers(ax1, df_ext, methods, "accumulated_delay", "Total Travel Time", completion_steps)
    plt.tight_layout()
    plt.savefig(args.out_delay, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out_delay}")

    # Figure 2: step vs runtime_s (SP CSV may not have runtime_segment_s; runtime_s is present in all)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    plot_with_completion_markers(ax2, df_ext, methods, "runtime_s", "Runtime (s)", completion_steps)
    plt.tight_layout()
    plt.savefig(args.out_runtime, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out_runtime}")


if __name__ == "__main__":
    main()
