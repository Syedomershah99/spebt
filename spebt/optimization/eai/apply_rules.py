#!/usr/bin/env python3
"""ANALYSIS_PLAN.md rules 3 to 5 on the re-measurement (remeasure.sh outputs).

  selection  Seeds 0 to 39 at T_sec 10; every design must have all 40 runs.
             Per design: mean, SD and SE of CNR at 150 iterations, and its own
             best iteration count (the peak of its mean curve).
             Rule 3: the headline is the CNR-alone (arm A) candidate with the
             highest mean. Reliability: each CNR-alone search (all of them,
             from searches.csv; one without a candidate counts as not
             reliable) whose best candidate reaches 4.603 (nd2fine_960's
             step-2 Grace mean, 4.703, minus 0.10; also minus 0.05 and 0.15),
             k of N with a Clopper-Pearson 95% interval. The nine shared
             starts: each CNR-alone search's best candidate minus the
             multi-objective one's, their mean and 95% t-interval. Ring 2 of
             each search's best candidate, and CCR's arms' best re-measured
             candidates. Runs only from committed code. Writes the headline
             to --headline_file (which must not exist yet; commit it before
             the held-out runs) and, given --designs, the held-out list
             heldout_designs.txt (the headline and the two references).
  heldout    Seeds 40 to 79 at T_sec 10 for the committed headline, the
             benchmark and nd2fine_960. Rule 5 against nd2fine_960, and the
             gain over the benchmark, with Welch intervals; each design's own
             best iteration count (from the selection seeds) as a secondary,
             descriptive read; nd2fine_960 on all 80 seeds (pass the
             selection folder too); the count levels (T_sec 2.5, 5, 20, 40 on
             seeds 40 to 59), descriptive.

Usage (from optimization/):
  python eai/apply_rules.py selection --remeasure SEL_RUN [...] --candidates OUT/candidates.csv \
      --ccr_candidates OUT/ccr_candidates.csv --searches OUT/searches.csv --designs designs.txt --out RULES
  python eai/apply_rules.py heldout --remeasure SEL_RUN HELD_RUN [...] --out RULES
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats

ITER, T_MAIN, N_ITER = 150, 10.0, 400
SEL, HELD = range(0, 40), range(40, 80)
COUNT_T, COUNT_SEEDS = (2.5, 5.0, 20.0, 40.0), range(40, 60)
MIC_STEP2, DELTA = 4.703, 0.10           # nd2fine_960's step-2 mean on Grace; equivalence margin
MARGINS = (0.10, 0.05, 0.15)            # reliability threshold MIC_STEP2 minus each; 0.10 is the rule
N_SEARCHES_A, N_PAIRS = 24, 9
BENCH, MIC = "step2a__tmi_reference_000", "step2a__nd2fine_960"
RING2_CAP = 960
HERE = os.path.dirname(os.path.abspath(__file__))
HEADLINE = os.path.join(HERE, "headline.txt")
TAG = re.compile(r"remeasure_N(\d+)_T([\d.]+)_seed(\d+)$")


def load(dirs):
    """One row per run: design name, T_sec, seed and its CNR curve (sector mean,
    iterations 1..400 in order). Every curve must have 400 iterations, and a
    run found in two folders must be identical in both."""
    runs, short = [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*", "remeasure_*", "dense_cnr.csv"))):
            m = TAG.search(os.path.basename(os.path.dirname(f)))
            if not m:
                continue
            c = pd.read_csv(f, float_precision="round_trip").sort_values("iteration")
            if int(m[1]) != N_ITER or len(c) != N_ITER:
                short.append(f)
                continue
            runs.append(dict(name=os.path.basename(os.path.dirname(os.path.dirname(f))),
                             T=float(m[2]), seed=int(m[3]), curve=c.cnr_sector_mean.to_numpy()))
    if short:
        sys.exit(f"ERROR: {len(short)} runs without {N_ITER} iterations, e.g. {short[:3]}")
    r = pd.DataFrame(runs, columns=["name", "T", "seed", "curve"])
    for key, g in r.groupby(["name", "T", "seed"]):
        if len(g) > 1 and not all(np.array_equal(g.curve.iloc[0], c) for c in g.curve):
            sys.exit(f"ERROR: run {key} differs between the --remeasure folders")
    return r.drop_duplicates(["name", "T", "seed"])


def block(runs, name, T, seeds):
    """(seeds x iterations array, missing seeds) for one design at one T_sec."""
    r = runs[(runs.name == name) & (runs["T"] == T) & runs.seed.isin(list(seeds))].sort_values("seed")
    missing = sorted(set(seeds) - set(r.seed))
    return (np.vstack(r.curve.to_list()) if len(r) else np.empty((0, 0))), missing


def summarize(a):
    m = a.mean(axis=0)
    x = a[:, ITER - 1]
    best = int(np.argmax(m)) + 1
    return dict(n=len(x), mean=x.mean(), sd=x.std(ddof=1), se=x.std(ddof=1) / np.sqrt(len(x)),
                best_iter=best, mean_at_best=m[best - 1])


def welch(x, y, level):
    """Mean of x minus mean of y, its Welch interval at `level`, and the two-sided p."""
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    d, se = x.mean() - y.mean(), np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    h = stats.t.ppf(0.5 + level / 2, df) * se
    return d, d - h, d + h, 2 * stats.t.sf(abs(d) / se, df)


def rule5(x, y):
    """Rule 5 (addendum 3): x against y at the primary endpoint. "higher" and
    "lower" (95% interval) come first; within_margin says whether the 90%
    interval also lies inside +/-0.10, which can hold together with either."""
    d, lo, hi, p = welch(x, y, 0.95)
    _, lo90, hi90, _ = welch(x, y, 0.90)
    within = bool(-DELTA < lo90 and hi90 < DELTA)
    verdict = "higher" if lo > 0 else "lower" if hi < 0 else "equivalent" if within else "inconclusive"
    return dict(diff=d, ci95_lo=lo, ci95_hi=hi, ci90_lo=lo90, ci90_hi=hi90, p=p, verdict=verdict,
                within_margin=within)


def clopper_pearson(k, n, level=0.95):
    a = 1 - level
    return (stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0,
            stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0)


def git(path, *cmd):
    """(return code, output) of a git command run in `path` (a folder) or where it lives."""
    cwd = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    r = subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def committed(path):
    """True if `path` is tracked by git and unchanged since the last commit."""
    f = os.path.basename(path)
    return git(path, "ls-files", "--error-unmatch", f)[0] == 0 and git(path, "diff", "--quiet", "HEAD", "--", f)[0] == 0


def selection(a, runs):
    if os.path.exists(a.headline_file):
        sys.exit(f"ERROR: {a.headline_file} exists: the headline is chosen once (rule 4)")
    code = [os.path.join(HERE, f) for f in ("apply_rules.py", "analyze_campaign.py")]
    if not all(committed(f) for f in code):
        sys.exit("ERROR: commit apply_rules.py and analyze_campaign.py first: the selection runs from committed code")
    d = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in [a.candidates] + a.ccr_candidates],
                  ignore_index=True).drop_duplicates("name")
    rows, missing = [], {}
    for _, c in d.iterrows():
        arr, miss = block(runs, c["name"], T_MAIN, SEL)
        if miss:
            missing[c["name"]] = miss
            continue
        rows.append(dict(name=c["name"], arm=c["arm"], search=c.get("search"),
                         n_det_ring2=c.get("n_det_ring2"), **summarize(arr)))
    if missing:
        for n, m in missing.items():
            print(f"missing {n}: {len(m)} seeds ({m[:5]}{' ...' if len(m) > 5 else ''})")
        sys.exit(f"ERROR: {len(missing)} designs lack selection seeds; rerun them (rule 3 uses complete designs)")
    tab = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    tab.to_csv(os.path.join(a.out, "designs_selection.csv"), index=False)
    pd.set_option("display.width", 200)
    A = tab[tab.arm == "A"]
    head = A.loc[A["mean"].idxmax()]
    print(f"Selection seeds {SEL.start}-{SEL.stop - 1}, CNR at {ITER} iterations, T_sec {T_MAIN:g}.\n")
    print("Best CNR-alone candidates (rule 3 picks the first):")
    print(A.nlargest(5, "mean")[["name", "mean", "se", "best_iter", "mean_at_best", "n_det_ring2"]]
          .round(4).to_string(index=False))
    for n in (BENCH, MIC):
        r = tab[tab.name == n].iloc[0]
        print(f"{n}: {r['mean']:.4f} +/- {r['sd']:.4f} (SD), best at {r.best_iter} iterations ({r.mean_at_best:.4f})")

    sea = pd.read_csv(a.searches)
    names_A = sorted(sea[sea.arm == "A"].search)
    best = tab.loc[tab[tab.arm.isin(["A", "B"])].groupby("search")["mean"].idxmax()]
    best.to_csv(os.path.join(a.out, "searches_best.csv"), index=False)
    bA = best[best.arm == "A"].set_index("search")
    no_cand = [s for s in names_A if s not in bA.index]
    print(f"\nReliability over {len(names_A)} CNR-alone searches (each search's best candidate"
          + (f"; {len(no_cand)} without a candidate count as not reliable: {no_cand}" if no_cand else "") + "):")
    if len(names_A) != N_SEARCHES_A:
        print(f"  WARNING: the plan has {N_SEARCHES_A} CNR-alone searches, searches.csv lists {len(names_A)}")
    for m in MARGINS:
        k = int((bA["mean"] >= MIC_STEP2 - m).sum())
        lo, hi = clopper_pearson(k, len(names_A))
        print(f"  mean >= {MIC_STEP2 - m:.3f} ({MIC_STEP2} - {m:.2f}){' [the rule]' if m == 0.10 else ''}: "
              f"{k} of {len(names_A)}, 95% interval {lo:.0%} to {hi:.0%}")
    at_cap = int((bA.n_det_ring2 == RING2_CAP).sum())
    print(f"  ring 2 at the {RING2_CAP} cap: {at_cap} of {len(bA)} best candidates; headline "
          f"{int(head.n_det_ring2)}{' (at the cap: if rule 5 says higher, also re-measure 944 and 980)' if head.n_det_ring2 == RING2_CAP else ''}")

    by = best.set_index("search")["mean"]
    pairs = pd.DataFrame([dict(start=s[6:], cnr_alone=by[s[6:]], several_objectives=v, diff=by[s[6:]] - v)
                          for s, v in by.items() if s.startswith("armB2_") and s[6:] in by])
    unpaired = [s for s in sea[sea.arm == "B"].search if s not in by.index or s[6:] not in by.index]
    if len(pairs) != N_PAIRS or unpaired:
        print(f"\n  WARNING: {len(pairs)} of the plan's {N_PAIRS} shared starts can be compared; "
              f"without a candidate on one side: {unpaired}")
    if len(pairs) > 1:
        pairs.to_csv(os.path.join(a.out, "pairs_remeasured.csv"), index=False)
        x = pairs["diff"].to_numpy()
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        print(f"\nShared starts ({len(x)}), CNR alone minus several objectives, best candidates:")
        print(pairs.round(4).to_string(index=False))
        print(f"  mean {x.mean():+.4f}, 95% t-interval {x.mean() - h:+.4f} to {x.mean() + h:+.4f}: "
              + ("a difference" if x.mean() - h > 0 or x.mean() + h < 0 else "no difference claimed"))
    ccr = tab[tab.arm.astype(str).str.startswith("ccr_")]
    if len(ccr):
        cb = ccr.loc[ccr.groupby("arm")["mean"].idxmax()].assign(ring2_above_960=lambda t: t.n_det_ring2 > RING2_CAP)
        print("\nCCR arms, best re-measured candidate (same re-measurement; descriptive). Budgets differ: "
              "CCR's candidates come from 78 to 80 optimizer designs per arm, in a box with ring 2 up to "
              "1362, Empire AI's from 100 per search with ring 2 up to 960:")
        print(cb[["arm", "name", "mean", "se", "n_det_ring2", "ring2_above_960"]].round(4).to_string(index=False))
        print(f"  CNR-alone searches' best candidates: median {bA['mean'].median():.4f}, "
              f"range {bA['mean'].min():.4f} to {bA['mean'].max():.4f}")

    sha = git(HERE, "rev-parse", "HEAD")[1]
    with open(a.headline_file, "w") as f:
        f.write(f"{head['name']}\n# rule 3: arm A, mean CNR {head['mean']:.6f} at {ITER} iterations over seeds "
                f"{SEL.start}-{SEL.stop - 1}, own best {head.best_iter} iterations; chosen by code {sha}\n")
    print(f"\nHeadline: {head['name']} (written to {a.headline_file}; commit it before the held-out runs)")
    if a.designs:
        keep = {head["name"], BENCH, MIC}
        dirs = [l.strip() for l in open(a.designs) if l.strip() and not l.startswith("#")]
        held = [p for p in dirs if f"{os.path.basename(os.path.dirname(p))}__{os.path.basename(p)}" in keep]
        if len(held) != 3:
            sys.exit(f"ERROR: {a.designs} holds {len(held)} of the 3 held-out designs")
        with open(os.path.join(a.out, "heldout_designs.txt"), "w") as f:
            f.writelines(p + "\n" for p in held)
        print(f"Held-out list: {os.path.join(a.out, 'heldout_designs.txt')}")


def heldout(a, runs):
    if not committed(a.headline_file):
        sys.exit(f"ERROR: {a.headline_file} is not committed as it stands: commit the headline first (rule 4)")
    head = open(a.headline_file).readline().strip()
    sel = pd.read_csv(os.path.join(a.out, "designs_selection.csv")).set_index("name")
    names = {"headline": head, "benchmark": BENCH, "nd2fine_960": MIC}
    x, missing = {}, {}
    for lab, n in names.items():
        arr, miss = block(runs, n, T_MAIN, HELD)
        if miss:
            missing[n] = miss
        x[lab] = arr
    if missing:
        sys.exit(f"ERROR: held-out seeds missing: { {n: len(m) for n, m in missing.items()} }")
    print(f"Held-out seeds {HELD.start}-{HELD.stop - 1}, T_sec {T_MAIN:g}; headline {head}\n")
    best_it = {lab: int(sel.loc[n, "best_iter"]) for lab, n in names.items()}
    tab = pd.DataFrame([dict(design=lab, name=names[lab], mean_150=arr[:, ITER - 1].mean(),
                             sd_150=arr[:, ITER - 1].std(ddof=1), own_best_iter=best_it[lab],
                             mean_own_best=arr[:, best_it[lab] - 1].mean(),
                             sd_own_best=arr[:, best_it[lab] - 1].std(ddof=1)) for lab, arr in x.items()])
    print(tab.round(4).to_string(index=False))
    h = x["headline"][:, ITER - 1]
    r5 = rule5(h, x["nd2fine_960"][:, ITER - 1])
    mic = x["nd2fine_960"][:, ITER - 1].mean()
    print(f"\nRule 5, headline minus nd2fine_960 at {ITER} iterations: {r5['diff']:+.4f} "
          f"({r5['diff'] / mic:+.1%}), 95% interval {r5['ci95_lo']:+.4f} to {r5['ci95_hi']:+.4f} "
          f"({r5['ci95_lo'] / mic:+.1%} to {r5['ci95_hi'] / mic:+.1%}), 90% interval {r5['ci90_lo']:+.4f} "
          f"to {r5['ci90_hi']:+.4f}: {r5['verdict'].upper()}"
          + (f" (and within the +/-{DELTA:.2f} equivalence margin)" if r5["within_margin"] and r5["verdict"] != "equivalent" else ""))
    bm = x["benchmark"][:, ITER - 1]
    d, lo, hi, p = welch(h, bm, 0.95)
    print(f"Headline minus the benchmark: {d:+.4f} ({d / bm.mean():+.1%}), 95% interval {lo:+.4f} to {hi:+.4f} "
          f"({lo / bm.mean():+.1%} to {hi / bm.mean():+.1%}), p = {p:.2g}")
    hb = best_it["headline"]
    for lab in ("nd2fine_960", "benchmark"):
        d2, lo2, hi2, _ = welch(x["headline"][:, hb - 1], x[lab][:, best_it[lab] - 1], 0.95)
        print(f"Secondary, no claim: each at its own best count ({hb} and {best_it[lab]}), headline minus "
              f"{lab} {d2:+.4f}, 95% interval {lo2:+.4f} to {hi2:+.4f}")
    if sel.loc[head, "n_det_ring2"] == RING2_CAP and r5["verdict"] == "higher":
        print(f"The headline has ring 2 at the {RING2_CAP} cap: re-measure it at 944 and 980 crystals too.")
    tab.to_csv(os.path.join(a.out, "heldout_designs.csv"), index=False)
    out = [dict(T_sec=T_MAIN, seeds=f"{HELD.start}-{HELD.stop - 1}", **r5)]
    pd.DataFrame(out).to_csv(os.path.join(a.out, "heldout_rule5.csv"), index=False)

    allm, miss = block(runs, MIC, T_MAIN, range(0, 80))
    if miss:
        print(f"nd2fine_960 on all 80 seeds: NOT REPORTED, {len(miss)} seeds missing "
              f"(pass the selection folder to --remeasure too)")
    else:
        print(f"nd2fine_960 on all 80 seeds: {allm[:, ITER - 1].mean():.4f} +/- {allm[:, ITER - 1].std(ddof=1):.4f} (SD)")
    levels = []
    for T in COUNT_T:
        b = {lab: block(runs, n, T, COUNT_SEEDS) for lab, n in names.items()}
        gaps = {lab: len(m) for lab, (_, m) in b.items() if m}
        if gaps:
            print(f"T_sec {T:g}: not reported, seeds missing {gaps}")
            continue
        d3, lo3, hi3, _ = welch(b["headline"][0][:, ITER - 1], b["nd2fine_960"][0][:, ITER - 1], 0.95)
        print(f"T_sec {T:g} (seeds {COUNT_SEEDS.start}-{COUNT_SEEDS.stop - 1}, descriptive): "
              + ", ".join(f"{lab} {v[0][:, ITER - 1].mean():.4f}" for lab, v in b.items())
              + f"; headline minus nd2fine_960 {d3:+.4f} ({lo3:+.4f} to {hi3:+.4f})")
        levels.append(dict(T_sec=T, seeds=f"{COUNT_SEEDS.start}-{COUNT_SEEDS.stop - 1}", diff=d3,
                           ci95_lo=lo3, ci95_hi=hi3, **{f"mean_{lab}": v[0][:, ITER - 1].mean() for lab, v in b.items()}))
    if levels:
        pd.DataFrame(levels).to_csv(os.path.join(a.out, "count_levels.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["selection", "heldout"])
    ap.add_argument("--remeasure", nargs="+", required=True, help="remeasure.sh output folders ($P/$RUN)")
    ap.add_argument("--candidates", help="analyze_campaign.py's candidates.csv (selection)")
    ap.add_argument("--ccr_candidates", nargs="*", default=[], help="its ccr_candidates.csv (selection)")
    ap.add_argument("--searches", help="its searches.csv: the list of searches (selection)")
    ap.add_argument("--designs", help="rebuild.py's designs.txt, to write the held-out list (selection)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headline_file", default=HEADLINE)
    a = ap.parse_args()
    runs = load(a.remeasure)
    print(f"{len(runs)} runs of {runs.name.nunique()} designs in {len(a.remeasure)} folder(s)")
    if a.stage == "selection":
        if not (a.candidates and a.searches):
            sys.exit("ERROR: selection needs --candidates and --searches")
        selection(a, runs)
    else:
        heldout(a, runs)


if __name__ == "__main__":
    main()
