#!/usr/bin/env python3
"""
Analyze LHS sweep results: metric patterns, design-metric relationships, Pareto front.

Generates:
  1. Pairwise metric correlations (4x4 scatter matrix)
  2. Design parameter → metric relationships (4x4 grid)
  3. Parallel coordinates plot (all objectives, colored by dominance)
  4. Baseline comparison table

Usage:
  python analyze_lhs_metrics.py --csv results/results_summary_mobo.csv --out_dir results/analysis_plots
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations


# Baseline config
BASELINE = {
    "aperture_diam_mm": 0.4,
    "n_apertures": 180,
    "n_det_ring1": 480,
    "n_det_ring2": 720,
}

METRIC_COLS = ["fwhm_mean", "asci_pct", "sensitivity_mean", "mpxi_mean"]
METRIC_LABELS = ["FWHM (mm)", "ASCI (%)", "Sensitivity", "MPXI"]
# Direction: -1 = minimize (lower is better), +1 = maximize (higher is better)
METRIC_DIRS = [-1, 1, 1, -1]

DESIGN_COLS = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2"]
DESIGN_LABELS = ["Aperture Diam (mm)", "N Apertures", "N Det Ring 1", "N Det Ring 2"]


def is_pareto_dominant(objectives):
    """
    Find Pareto-optimal points.
    objectives: (n, m) array where all objectives are to be MAXIMIZED.
    Returns boolean mask of non-dominated points.
    """
    n = len(objectives)
    is_optimal = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j >= i on all objectives and j > i on at least one
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_optimal[i] = False
                break
    return is_optimal


def plot_metric_correlations(df, out_dir):
    """Plot 1: Pairwise metric scatter matrix with Pareto coloring."""
    n = len(METRIC_COLS)
    fig, axes = plt.subplots(n, n, figsize=(14, 12))
    fig.suptitle("Pairwise Metric Correlations (LHS Sweep)", fontsize=14, y=0.98)

    # Compute Pareto front
    obj_matrix = df[METRIC_COLS].values.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj_matrix[:, i] *= d  # flip so all are "maximize"
    pareto_mask = is_pareto_dominant(obj_matrix)

    colors = np.where(pareto_mask, "#e74c3c", "#3498db")
    sizes = np.where(pareto_mask, 60, 30)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Histogram on diagonal
                ax.hist(df[METRIC_COLS[i]], bins=15, color="#3498db", alpha=0.7, edgecolor="white")
                ax.set_xlabel(METRIC_LABELS[i], fontsize=8)
            else:
                for k in range(len(df)):
                    ax.scatter(df[METRIC_COLS[j]].iloc[k], df[METRIC_COLS[i]].iloc[k],
                               c=colors[k], s=sizes[k], alpha=0.7, edgecolors="white", linewidth=0.5)

                # Correlation coefficient
                corr = df[METRIC_COLS[j]].corr(df[METRIC_COLS[i]])
                ax.annotate(f"r={corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                            fontsize=8, color="black",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if j == 0:
                ax.set_ylabel(METRIC_LABELS[i], fontsize=8)
            if i == n - 1:
                ax.set_xlabel(METRIC_LABELS[j], fontsize=8)
            ax.tick_params(labelsize=7)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Pareto optimal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Dominated'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    path = os.path.join(out_dir, "metric_correlations.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_design_vs_metrics(df, out_dir):
    """Plot 2: How each design parameter affects each metric."""
    n_design = len(DESIGN_COLS)
    n_metric = len(METRIC_COLS)

    fig, axes = plt.subplots(n_metric, n_design, figsize=(16, 12))
    fig.suptitle("Design Parameters → Metrics (LHS Sweep)", fontsize=14, y=0.98)

    for i, (mcol, mlabel) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
        for j, (dcol, dlabel) in enumerate(zip(DESIGN_COLS, DESIGN_LABELS)):
            ax = axes[i, j]
            ax.scatter(df[dcol], df[mcol], c="#2ecc71", s=35, alpha=0.7, edgecolors="white", linewidth=0.5)

            # Add baseline marker
            if dcol in BASELINE:
                baseline_metric = None  # we don't know baseline metric values
                ax.axvline(BASELINE[dcol], color="red", linestyle="--", alpha=0.5, linewidth=1)

            # Trend line
            if len(df) >= 5:
                z = np.polyfit(df[dcol], df[mcol], 1)
                x_line = np.linspace(df[dcol].min(), df[dcol].max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), color="gray", linestyle="--", alpha=0.6, linewidth=1)

                corr = df[dcol].corr(df[mcol])
                ax.annotate(f"r={corr:.2f}", xy=(0.05, 0.92), xycoords="axes fraction",
                            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            if j == 0:
                ax.set_ylabel(mlabel, fontsize=9)
            if i == n_metric - 1:
                ax.set_xlabel(dlabel, fontsize=9)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "design_vs_metrics.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def plot_parallel_coordinates(df, out_dir):
    """Plot 3: Parallel coordinates with Pareto coloring."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Parallel Coordinates — All Objectives (LHS Sweep)", fontsize=13)

    # Normalize each metric to [0, 1] for visualization
    # Flip sign for minimization objectives so "up" is always "better"
    obj_vals = df[METRIC_COLS].values.copy()
    display_vals = obj_vals.copy()
    for i, d in enumerate(METRIC_DIRS):
        if d == -1:
            display_vals[:, i] = -display_vals[:, i]

    # Normalize to [0,1]
    mins = display_vals.min(axis=0)
    maxs = display_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (display_vals - mins) / ranges

    # Pareto front
    obj_maximized = obj_vals.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj_maximized[:, i] *= d
    pareto_mask = is_pareto_dominant(obj_maximized)

    x_positions = np.arange(len(METRIC_COLS))

    # Draw dominated first (behind)
    for idx in range(len(df)):
        if not pareto_mask[idx]:
            ax.plot(x_positions, normed[idx], color="#3498db", alpha=0.25, linewidth=1)

    # Draw Pareto on top
    for idx in range(len(df)):
        if pareto_mask[idx]:
            ax.plot(x_positions, normed[idx], color="#e74c3c", alpha=0.8, linewidth=2.5)

    # Axis labels
    direction_arrows = ["↓ better" if d == -1 else "↑ better" for d in METRIC_DIRS]
    labels = [f"{ml}\n({da})" for ml, da in zip(METRIC_LABELS, direction_arrows)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Normalized value (up = better)", fontsize=10)

    # Add raw value annotations on axes
    for i in range(len(METRIC_COLS)):
        col = obj_vals[:, i]
        if METRIC_DIRS[i] == -1:
            ax.annotate(f"{col.min():.3f}", xy=(i, 1.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="bottom", color="green")
            ax.annotate(f"{col.max():.3f}", xy=(i, 0.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="top", color="red")
        else:
            ax.annotate(f"{col.max():.3g}", xy=(i, 1.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="bottom", color="green")
            ax.annotate(f"{col.min():.3g}", xy=(i, 0.0), xycoords=("data", "axes fraction"),
                        fontsize=7, ha="center", va="top", color="red")

    legend_elements = [
        Line2D([0], [0], color='#e74c3c', linewidth=2.5, label=f'Pareto optimal ({pareto_mask.sum()})'),
        Line2D([0], [0], color='#3498db', linewidth=1, alpha=0.5, label=f'Dominated ({(~pareto_mask).sum()})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "parallel_coordinates.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def print_summary_table(df):
    """Print metric summary with baseline comparison."""
    # Pareto
    obj_vals = df[METRIC_COLS].values.copy()
    for i, d in enumerate(METRIC_DIRS):
        obj_vals[:, i] *= d
    pareto_mask = is_pareto_dominant(obj_vals)

    print("\n" + "=" * 70)
    print("METRIC SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Min':>10} {'Max':>10} {'Mean':>10} {'Baseline':>10}")
    print("-" * 70)
    # We don't have baseline metric values, so just show design params
    for col, label in zip(METRIC_COLS, METRIC_LABELS):
        vals = df[col]
        print(f"{label:<20} {vals.min():>10.4f} {vals.max():>10.4f} {vals.mean():>10.4f} {'—':>10}")

    print(f"\nTotal configs: {len(df)}")
    print(f"Pareto optimal: {pareto_mask.sum()}")

    print("\n" + "=" * 70)
    print("PARETO OPTIMAL CONFIGS")
    print("=" * 70)
    pareto_df = df[pareto_mask]
    display_cols = DESIGN_COLS + METRIC_COLS
    available = [c for c in display_cols if c in pareto_df.columns]
    print(pareto_df[available].to_string(index=False))

    # Pairwise correlations
    print("\n" + "=" * 70)
    print("METRIC PAIRWISE CORRELATIONS")
    print("=" * 70)
    for (c1, l1), (c2, l2) in combinations(zip(METRIC_COLS, METRIC_LABELS), 2):
        corr = df[c1].corr(df[c2])
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        print(f"  {l1:>15} vs {l2:<15}  r={corr:+.3f}  ({strength})")


def main():
    parser = argparse.ArgumentParser(description="Analyze LHS metrics")
    parser.add_argument("--csv", type=str, required=True, help="Results CSV path")
    parser.add_argument("--out_dir", type=str, default="results/analysis_plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=METRIC_COLS)
    print(f"Loaded {len(df)} feasible configs from {args.csv}")

    if len(df) < 3:
        print("ERROR: Need at least 3 feasible configs for analysis")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    plot_metric_correlations(df, args.out_dir)
    plot_design_vs_metrics(df, args.out_dir)
    plot_parallel_coordinates(df, args.out_dir)
    print_summary_table(df)

    print(f"\nAll plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
