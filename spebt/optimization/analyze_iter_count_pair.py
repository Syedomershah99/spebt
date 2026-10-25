#!/usr/bin/env python3
"""How the benchmark-vs-MOBO CNR gap depends on the ML-EM iteration count and
the count level (runs of submit_iter_count_pair.sh, collected on CCR by
collect_iter_count_pair.py and copied to archive_backup/misc/iter_count_pair/).

The two designs' CNR curves peak at very different iteration counts (in the
first runs, about 190 for the MOBO-optimized design and 370 for the benchmark),
so a gain at one matched iteration count mixes the design difference with how
far each design is from its own best. This prints both views, and neither alone
is "the gain":

  - matched iteration counts (25 to 400, T_sec 10): each design's sector-mean
    CNR as mean +/- SD over seeds 0-4, the gain, Welch's t and p, and whether
    every MOBO-optimized run beats every benchmark run; plus the last iteration
    with that full separation and where the difference stops being significant
  - each design at its own best iteration count, for every count level: from
    each seed's own peak, and at the peak of the design's mean curve
  - the count-level comparison at 150 iterations and at each design's best
  - per-rod-size gains, at matched counts and at each design's best

A seed fixes the noise draw only for one design at one count level, so the
gain is tested unpaired (Welch). The MOBO design was chosen by its CNR at 150
iterations, which favours designs that peak early; keep that in mind when
reading the best-against-best numbers.

Writes four figures and two tables to plots/iter_count_pair/, not to
deck_figures: nothing goes into the talk without Dr. Yao's approval.

Refuses to run unless validation.txt ends in PASS and its checksums match the
runs.csv and dense.csv next to it.

Usage (from optimization/):
  python analyze_iter_count_pair.py
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "archive_backup", "misc", "iter_count_pair")
OUT = os.path.join(HERE, "..", "plots", "iter_count_pair")
BENCH, MOBO = "tmi_reference_000", "nd2fine_960"
NAMES = {BENCH: "benchmark", MOBO: "MOBO-optimized"}
COLORS = {BENCH: "#eb6834", MOBO: "#2a78d6"}
ROD_RADII_MM = [0.100, 0.125, 0.150, 0.175, 0.200, 0.225]     # sector order
ITERS = [25, 50, 100, 150, 200, 300, 400]
CAMPAIGN_ITER, CAMPAIGN_T = 150, 10.0
SECTORS = [f"sector{k}" for k in range(len(ROD_RADII_MM))]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def check_gate(data):
    """validation.txt must end in PASS and name the checksums of these files."""
    val = open(os.path.join(data, "validation.txt")).read().strip().splitlines()
    if not val or val[-1] != "PASS":
        sys.exit("ERROR: validation.txt does not end in PASS; fix the runs first")
    for name in ("runs.csv", "dense.csv"):
        want = [l.split()[-1] for l in val if l.startswith(f"sha256 {name} ")]
        if not want or want[0] != sha256(os.path.join(data, name)):
            sys.exit(f"ERROR: {name} is not the file validation.txt passed; copy all three together")


def compare(b, m):
    """Both designs' mean/SD over seeds, the gain, Welch's t and p."""
    if len(b) != len(m) or len(b) < 5:
        sys.exit(f"ERROR: {len(b)} benchmark and {len(m)} MOBO-optimized values; "
                 f"expected 5 of each")
    t, p = stats.ttest_ind(m, b, equal_var=False)
    return dict(n=len(b), bench=b.mean(), bench_sd=b.std(ddof=1), mobo=m.mean(),
                mobo_sd=m.std(ddof=1), gain=m.mean() - b.mean(),
                gain_pct=100 * (m.mean() / b.mean() - 1), t=t, p=p,
                separated=bool(m.min() > b.max()))


def values(dense, design, t_sec, iteration, col="cnr_sector_mean"):
    g = dense[(dense.design == design) & (dense.T_sec == t_sec) & (dense.iteration == iteration)]
    return g.sort_values("seed")[col].to_numpy()


def line(r, label):
    return (f"  {label:>14} | {r['bench']:.3f} +/- {r['bench_sd']:.3f} | {r['mobo']:.3f} +/- "
            f"{r['mobo_sd']:.3f} | {r['gain']:+.3f}  {r['gain_pct']:+5.1f}%  Welch t={r['t']:5.1f}  "
            f"p={r['p']:.1e}  {'every MOBO run above every benchmark run' if r['separated'] else 'ranges overlap'}"
            f"  (n={r['n']} each)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    check_gate(a.data)
    runs = pd.read_csv(os.path.join(a.data, "runs.csv"))
    dense = pd.read_csv(os.path.join(a.data, "dense.csv"))
    n_run = int(runs.n_run.min())
    if (runs.n_run != runs.n_run.max()).any():
        print(f"NOTE: runs differ in length ({runs.n_run.min()} to {runs.n_run.max()} iterations); "
              f"only iterations 1 to {n_run} are compared")
        dense = dense[dense.iteration <= n_run]
    levels_t = sorted(dense.T_sec.unique())
    hdr = "  {:>14} | benchmark | MOBO-optimized | gain"

    # Matched iteration counts at the campaign's count level.
    print("\nCNR (sector mean) at matched ML-EM iteration counts, T_sec 10")
    print(hdr.format("iterations"))
    it_rows = []
    for n in ITERS:
        r = compare(values(dense, BENCH, CAMPAIGN_T, n), values(dense, MOBO, CAMPAIGN_T, n))
        it_rows.append(dict(level=n, **r))
        print(line(r, n))
    sep_last, nonsig = None, []
    for n in range(1, n_run + 1):
        r = compare(values(dense, BENCH, CAMPAIGN_T, n), values(dense, MOBO, CAMPAIGN_T, n))
        if r["separated"]:
            sep_last = n
        if r["p"] >= 0.05:
            nonsig.append(n)
    print(f"  every MOBO run beats every benchmark run at each iteration through {sep_last}")
    print(f"  Welch p >= 0.05 at {len(nonsig)} iteration counts"
          + (f", from {min(nonsig)} to {max(nonsig)}" if nonsig else ""))

    # Each design at its own best iteration count.
    print("\nEach design at its own best iteration count (iterations 1 to %d)" % n_run)
    best_rows, best_at = [], {}
    for t in levels_t:
        sub = dense[dense.T_sec == t]
        peaks = sub.loc[sub.groupby(["design", "seed"]).cnr_sector_mean.idxmax()]
        pb = peaks[peaks.design == BENCH].sort_values("seed")
        pm = peaks[peaks.design == MOBO].sort_values("seed")
        r_seed = compare(pb.cnr_sector_mean.to_numpy(), pm.cnr_sector_mean.to_numpy())
        mc = sub.groupby(["design", "iteration"]).cnr_sector_mean.mean()
        ib, im = int(mc[BENCH].idxmax()), int(mc[MOBO].idxmax())
        best_at[t] = {BENCH: ib, MOBO: im}
        r_mean = compare(values(dense, BENCH, t, ib), values(dense, MOBO, t, im))
        best_rows.append(dict(T_sec=t, bench_iter=ib, mobo_iter=im,
                              bench_seed_peak_iters=f"{pb.iteration.min()}-{pb.iteration.max()}",
                              mobo_seed_peak_iters=f"{pm.iteration.min()}-{pm.iteration.max()}",
                              **{f"seedpeak_{k}": v for k, v in r_seed.items()},
                              **{f"meanpeak_{k}": v for k, v in r_mean.items()}))
        print(f"  T_sec {t:g}: benchmark peaks at {ib} iterations (seeds {pb.iteration.min()} to "
              f"{pb.iteration.max()}), MOBO-optimized at {im} (seeds {pm.iteration.min()} to "
              f"{pm.iteration.max()})")
        print(hdr.format(""))
        print(line(r_seed, "seed peaks"))
        print(line(r_mean, "mean-curve peak"))
        for d, i_best in ((BENCH, ib), (MOBO, im)):
            c = mc[d]
            if i_best >= n_run:
                print(f"  WARNING: {NAMES[d]} peaks at the last iteration ({n_run}); "
                      f"its true peak may lie beyond the run")
            elif i_best >= 0.9 * n_run:
                print(f"  note: {NAMES[d]} peaks at {i_best} of {n_run}; CNR at {n_run} is "
                      f"{c.max() - c.loc[n_run]:.3f} below the peak, so the peak is inside the run")

    # Count level: at the campaign's 150 iterations, and at each design's best.
    print("\nCNR against count level (T_sec; 10 is the campaign's)")
    print(hdr.format("T_sec, iters"))
    ct_rows = []
    for t in levels_t:
        r150 = compare(values(dense, BENCH, t, CAMPAIGN_ITER), values(dense, MOBO, t, CAMPAIGN_ITER))
        ct_rows.append(dict(level=t, at="150", **r150))
        print(line(r150, f"{t:g}, 150"))
        ib, im = best_at[t][BENCH], best_at[t][MOBO]
        rb = compare(values(dense, BENCH, t, ib), values(dense, MOBO, t, im))
        ct_rows.append(dict(level=t, at=f"{ib}/{im}", **rb))
        print(line(rb, f"{t:g}, {ib}/{im}"))

    print("\nPer-rod-size gain (%, mean over seeds), T_sec 10")
    print("  iterations | " + " | ".join(f"r={r:.3f}" for r in ROD_RADII_MM))
    rows = [(f"{n}", n, n) for n in ITERS]
    ib, im = best_at[CAMPAIGN_T][BENCH], best_at[CAMPAIGN_T][MOBO]
    rows.append((f"best {ib}/{im}", ib, im))
    for label, nb, nm in rows:
        cells = [f"{100 * (values(dense, MOBO, CAMPAIGN_T, nm, s).mean() / values(dense, BENCH, CAMPAIGN_T, nb, s).mean() - 1):+6.1f}"
                 for s in SECTORS]
        print(f"  {label:>10} | " + "  | ".join(cells))

    os.makedirs(a.out, exist_ok=True)
    curve = dense.groupby(["design", "T_sec", "iteration"]).cnr_sector_mean.agg(
        ["mean", "std"]).reset_index()

    def finish(fig, ax, name):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, name), dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for d, g in curve[curve.T_sec == CAMPAIGN_T].groupby("design"):
        ax.plot(g.iteration, g["mean"], color=COLORS[d], lw=2, label=NAMES[d])
        ax.fill_between(g.iteration, g["mean"] - g["std"], g["mean"] + g["std"],
                        color=COLORS[d], alpha=0.18, lw=0)
        i_best = best_at[CAMPAIGN_T][d]
        y_best = g.loc[g.iteration == i_best, "mean"].iloc[0]
        ax.plot([i_best], [y_best], "o", color=COLORS[d], ms=6)
        # Benchmark label below its curve, so it stays clear of the other band.
        ax.annotate(f"peak, {i_best} iterations", (i_best, y_best), textcoords="offset points",
                    xytext=(0, 9 if d == MOBO else -15), ha="center", fontsize=9, color=COLORS[d])
    ax.axvline(CAMPAIGN_ITER, color="#52514e", lw=1, ls="--")
    ax.text(CAMPAIGN_ITER + 4, ax.get_ylim()[0] + 0.05, "150, as in the campaign",
            fontsize=9, color="#52514e")
    ax.set_xlabel("ML-EM iterations")
    ax.set_ylabel("sector-mean CNR (mean ± SD, 5 noise realizations)")
    ax.set_title("CNR against ML-EM iterations, T_sec 10")
    ax.legend(frameon=False, loc="lower right")
    finish(fig, ax, "cnr_vs_iterations.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    piv = curve.pivot_table(index=["T_sec", "iteration"], columns="design", values="mean")
    for t in levels_t:
        g = piv.loc[t]
        ln, = ax.plot(g.index, 100 * (g[MOBO] / g[BENCH] - 1), lw=2 if t == CAMPAIGN_T else 1.2,
                      label=f"T_sec {t:g}" + (" (campaign)" if t == CAMPAIGN_T else ""))
        row = [r for r in best_rows if r["T_sec"] == t][0]
        ax.plot([n_run + 12], [row["meanpeak_gain_pct"]], "<", color=ln.get_color(), ms=7,
                clip_on=False)
    ax.axvline(CAMPAIGN_ITER, color="#52514e", lw=1, ls="--")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xlim(0, n_run + 20)
    ax.set_xlabel("ML-EM iterations (both designs at the same count)")
    ax.set_ylabel("MOBO-optimized over benchmark, mean CNR (%)")
    ax.set_title("CNR gain against iterations; markers at right: each design at its own best")
    ax.legend(frameon=False, fontsize=9)
    finish(fig, ax, "gain_vs_iterations.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for d in (BENCH, MOBO):
        g = curve[(curve.design == d) & (curve.iteration == CAMPAIGN_ITER)]
        ax.errorbar(g.T_sec, g["mean"], yerr=g["std"], color=COLORS[d], marker="o",
                    capsize=3, lw=2, label=f"{NAMES[d]}, 150 iterations")
        pk = [curve[(curve.design == d) & (curve.T_sec == t) & (curve.iteration == best_at[t][d])]
              .iloc[0] for t in levels_t]
        ax.errorbar(levels_t, [p["mean"] for p in pk], yerr=[p["std"] for p in pk],
                    color=COLORS[d], marker="s", mfc="white", capsize=3, lw=1.2, ls="--",
                    label=f"{NAMES[d]}, own best iteration count")
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels_t)
    ax.set_xticklabels([f"{t:g}" for t in levels_t])
    ax.set_xlabel("scan time T_sec (counts scale with it; 10 in the campaign)")
    ax.set_ylabel("sector-mean CNR (mean ± SD, 5 realizations)")
    ax.set_title("CNR against count level")
    ax.legend(frameon=False, fontsize=8.5)
    finish(fig, ax, "cnr_vs_count_level.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for d in (BENCH, MOBO):
        ax.plot(levels_t, [best_at[t][d] for t in levels_t], marker="o", color=COLORS[d], lw=2,
                label=NAMES[d])
    ax.axhline(CAMPAIGN_ITER, color="#52514e", lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels_t)
    ax.set_xticklabels([f"{t:g}" for t in levels_t])
    ax.set_xlabel("scan time T_sec")
    ax.set_ylabel("iteration count with the highest mean CNR")
    ax.set_title("Best ML-EM iteration count against count level")
    ax.legend(frameon=False)
    finish(fig, ax, "best_iteration_vs_count_level.png")

    pd.DataFrame(it_rows).to_csv(os.path.join(a.out, "gain_vs_iterations.csv"), index=False)
    pd.DataFrame(ct_rows).to_csv(os.path.join(a.out, "gain_vs_count_level.csv"), index=False)
    pd.DataFrame(best_rows).to_csv(os.path.join(a.out, "gain_at_best_iterations.csv"), index=False)
    print(f"\nwrote figures and tables to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
