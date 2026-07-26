#!/usr/bin/env python3
"""
Summarise the repeat-run CNR measurements produced by submit_cnr_repeats.sh.

Reports mean +/- std over seeds for each top design, plus the per-sector CNR
spread, so CNR can be quoted with an error bar instead of as a bare point
estimate. Also prints pairwise design gaps against the measurement noise, which
is the number that decides whether one design actually beats another.

Usage:
  python3 analyze_cnr_repeats.py
  python3 analyze_cnr_repeats.py --results_dir results
"""
import argparse
import glob
import os
import re

import numpy as np

# Top designs from the 180-iteration campaign. Keep in sync with
# submit_cnr_repeats.sh.
CONFIGS = [
    "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230",
    "mobo_0177_ap0.3512_nap97_nd1_604_nd2_584",
    "mobo_0173_ap0.3500_nap117_nd1_446_nd2_236",
]


def short_name(config: str) -> str:
    m = re.match(r"(mobo_\d+)", config)
    return m.group(1) if m else config[:12]


def collect(results_dir: str, config: str):
    """Return (seeds, overall_cnrs, sector_matrix) for one config."""
    pattern = os.path.join(results_dir, config, "cnr_repeat_seed*", "cnr_results.npz")
    seeds, overall, sectors = [], [], []
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"cnr_repeat_seed(\d+)", path)
        if not m:
            continue
        d = np.load(path, allow_pickle=True)
        seeds.append(int(m.group(1)))
        overall.append(float(d["overall_cnr"]))
        sectors.append(np.asarray(d["sector_cnrs"], dtype=float))
    if not overall:
        return [], np.array([]), np.empty((0, 0))
    order = np.argsort(seeds)
    return (
        [seeds[i] for i in order],
        np.array([overall[i] for i in order]),
        np.vstack([sectors[i] for i in order]),
    )


def main():
    ap = argparse.ArgumentParser(description="Summarise repeat-run CNR measurements")
    ap.add_argument("--results_dir", default="results",
                    help="Directory holding the per-config work dirs (default: results)")
    args = ap.parse_args()

    stats = {}

    print("=" * 72)
    print("PER-DESIGN CNR ACROSS SEEDS")
    print("=" * 72)
    for config in CONFIGS:
        seeds, overall, sector_mat = collect(args.results_dir, config)
        name = short_name(config)
        if overall.size == 0:
            print(f"\n{name}: no repeat runs found "
                  f"(expected {args.results_dir}/{config}/cnr_repeat_seed*/cnr_results.npz)")
            continue

        mean, std = overall.mean(), overall.std(ddof=1) if overall.size > 1 else 0.0
        stats[name] = (mean, std, overall.size)

        print(f"\n{name}  (n={overall.size})")
        print(f"  seeds:   {seeds}")
        print(f"  CNR:     {np.array2string(overall, precision=4, floatmode='fixed')}")
        print(f"  mean:    {mean:.4f}")
        print(f"  std:     {std:.4f}")
        print(f"  range:   {overall.min():.4f} - {overall.max():.4f}  "
              f"(spread {overall.max() - overall.min():.4f})")

        if sector_mat.size:
            print("  per-sector mean +/- std:")
            for s in range(sector_mat.shape[1]):
                col = sector_mat[:, s]
                sd = col.std(ddof=1) if col.size > 1 else 0.0
                print(f"    sector {s}: {col.mean():.3f} +/- {sd:.3f}")

    if len(stats) < 2:
        print("\nNeed at least two designs with repeat runs for a pairwise comparison.")
        return

    # Pooled std across all designs = the measurement noise floor.
    pooled = np.sqrt(np.mean([s ** 2 for (_, s, n) in stats.values() if n > 1]))

    print()
    print("=" * 72)
    print("PAIRWISE GAPS vs MEASUREMENT NOISE")
    print("=" * 72)
    print(f"\nPooled per-run std (noise floor): {pooled:.4f}")
    print("A gap is only meaningful if it clears ~2x the std of the difference,")
    print(f"i.e. roughly {2 * pooled * np.sqrt(2):.4f} CNR for a single-run comparison.\n")

    names = list(stats.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            (ma, sa, na), (mb, sb, nb) = stats[a], stats[b]
            gap = ma - mb
            # Standard error of the difference of two means
            se = np.sqrt((sa ** 2 / na if na else 0) + (sb ** 2 / nb if nb else 0))
            verdict = "DISTINGUISHABLE" if se > 0 and abs(gap) > 2 * se else "indistinguishable"
            se_str = f"{abs(gap) / se:.2f}" if se > 0 else "n/a"
            print(f"  {a} vs {b}:")
            print(f"    gap = {gap:+.4f}   SE(diff) = {se:.4f}   |gap|/SE = {se_str}")
            print(f"    -> {verdict}")

    print()
    print("=" * 72)
    print("QUOTABLE SUMMARY")
    print("=" * 72)
    for name, (mean, std, n) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name}: CNR = {mean:.2f} +/- {std:.2f}  (n={n} seeds)")


if __name__ == "__main__":
    main()
