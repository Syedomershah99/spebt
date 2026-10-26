#!/usr/bin/env python3
"""Search-level results of the Empire AI campaign (ANALYSIS_PLAN.md, addendum 3).

Reads the mirror of the searches (run_node_searches.sh copies every search's
CSVs and logs to /projects/.../mirror; copy that folder here) and CCR's
head-to-head arms (archive_backup/h2h), and writes to --out:
  searches.csv        per search: arm (A = CNR alone, B = CNR with FWHM, ASCI
                      and PPDS), start, designs run on Empire AI, optimizer
                      designs (manifest rows, with a CNR, lost, still
                      running), best in-window in-loop CNR at --at optimizer
                      designs and at the end, window adherence at both points,
                      median minutes per design (from each design's own log)
  best_so_far.png     best in-window in-loop CNR against designs, one panel per
                      arm, with the median and interquartile range over the
                      searches that all reached each point
  paired_ccr.csv      the three paired starts, both arms, against CCR's 2- and
                      5-objective arms with the same 21 starting designs, read
                      after the same number of proposals (the furthest all
                      reached, at most --at), as run and with ring 2 <= 960
  pairs.csv           the nine starts shared by the two arms, in-loop; the
                      comparison itself uses the re-measurement (apply_rules.py)
  candidates.csv      per search, the 3 best in-window optimizer designs among
                      its first 100, after the two references; written only
                      once all 33 searches have ended at the final point
                      (--allow_partial writes candidates_partial.csv before)
  ccr_candidates.csv  the same for CCR's six complete arms, to rebuild on Grace
                      (both files, with design vectors, are rebuild.py's input)
In-loop CNR is one noise draw per design; the re-measurement settles the CNR.
A design's repeated rows (eai-v1 launcher) are dropped as the optimizer drops
them (mobo_agent.clean_repeated_rows). Optimizer designs are placed by their
manifest index (CCR's by the index in their names), so a lost design still uses
up its place in the budget.

Usage (from optimization/):
  python eai/analyze_campaign.py --mirror MIRROR --h2h ../archive_backup/h2h --out OUT
"""
import argparse
import os
import re
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mobo_agent import clean_repeated_rows  # noqa: E402

LO, HI = 1.5, 2.5
N_START, N_BO, RING2_CAP = 21, 100, 960
CNR, MPXI, RING2 = "cnr_sector_mean", "mpxi_windowed_active_mean", "n_det_ring2"
PARAMS = ["aperture_diam_mm", "n_apertures", "n_det_ring1", "n_det_ring2", "d2_inner_mm", "d3_inner_mm"]
ARMS = {"A": "CNR alone", "B": "CNR + FWHM + ASCI + PPDS"}
# The campaign's 33 searches: 24 CNR alone (nodes 1, 2, 4, 7), 9 multi-objective (nodes 3, 5, 6)
EXPECTED = ([f"warm_r{i}" for i in range(3)] + [f"fresh_s{i}" for i in range(10, 31)]
            + [f"armB2_warm_r{i}" for i in range(3)] + [f"armB2_fresh_s{i}" for i in range(13, 19)])
# Paired start -> CCR arms with the same 21 starting designs (r1 = r4, r2 = r5)
CCR_PAIRS = {"warm_r0": ("2obj", "5obj"), "warm_r1": ("2obj_r4", "5obj_r4"),
             "warm_r2": ("2obj_r5", "5obj_r5")}
REFS = {"tmi_reference_000": "/ddn/scratch/suny/oshah/step2a/tmi_reference_000",
        "nd2fine_960": "/ddn/scratch/suny/oshah/step2a/nd2fine_960"}
# Their design vectors: step 2's reference list, as used on Grace
REF_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_refs.csv")
STAMP = re.compile(r"(SAI Pipeline|PIPELINE COMPLETE)[^|\n]*\| \w+ (\w+ +\d+ [\d:]+(?: [AP]M)?) (\w+) (\d{4})")
TZ_HOURS = {"EDT": -4, "EST": -5, "UTC": 0, "GMT": 0}


def arm_of(search):
    """('A' or 'B', 'paired' or 'fresh') from a search name, or None."""
    m = re.fullmatch(r"(armB2_)?(warm_r|fresh_s)\d+", search)
    return (("B" if m[1] else "A"), ("paired" if m[2] == "warm_r" else "fresh")) if m else None


def inside(df):
    return (df[MPXI] >= LO) & (df[MPXI] <= HI)


def window_cnr(rows, cap=None):
    """In-loop CNR of in-window designs (NaN elsewhere), optionally ring 2 <= cap."""
    keep = inside(rows) & (rows[RING2] <= cap if cap else True)
    return rows[CNR].where(keep).to_numpy(float)


def running_best(v):
    """Running maximum; -inf until the first value, so that a band over searches
    always has the same searches behind it."""
    return np.maximum.accumulate(np.where(np.isnan(v), -np.inf, v)) if len(v) else np.array([])


def at(c, n):
    """Best after n designs, NaN if not reached or nothing in the window yet."""
    return float(c[n - 1]) if 1 <= n <= len(c) and np.isfinite(c[n - 1]) else np.nan


def load_search(d):
    """Cleaned starting rows, the optimizer's designs in manifest order up to the
    final point (the first 100) with their results, and the names of designs
    that had repeated rows."""
    raw = pd.read_csv(os.path.join(d, "results_summary_mobo.csv"), float_precision="round_trip")
    for c in [CNR, MPXI] + PARAMS:
        if c not in raw:
            raw[c] = np.nan
    df = clean_repeated_rows(raw)
    repeated = set(raw.config[raw.config.duplicated()].astype(str))
    p = os.path.join(d, "mobo_manifest.csv")
    man = (pd.read_csv(p, float_precision="round_trip") if os.path.exists(p)
           else pd.DataFrame(columns=["idx", "config_name", "work_dir", "status"]))
    man = man.sort_values("idx").reset_index(drop=True)[["idx", "config_name", "work_dir", "status"]]
    man = man[man.idx < N_BO]
    bo = man.join(df.set_index(df.config.astype(str))[[CNR, MPXI] + PARAMS], on="config_name")
    start = df[~df.config.astype(str).str.startswith("mobo_")]
    return start, bo, repeated


def curve(start, bo, cap=None):
    """Best in-window in-loop CNR after each design: the starting designs in row
    order (padded to 21 once the optimizer has started), then one place per
    manifest row."""
    s = window_cnr(start, cap)[:N_START]
    if len(bo):
        s = np.concatenate([s, np.full(N_START - len(s), np.nan)])
    return running_best(np.concatenate([s, window_cnr(bo, cap)]))


def ccr_arm(df):
    """A CCR arm placed like an Empire AI search: its 21 starting rows, then one
    place per proposal (mobo_NNNN), with no result where a proposal left no row."""
    rest = df.iloc[N_START:]
    idx = rest.config.str.extract(r"^mobo_(\d+)")[0].astype(int).to_numpy()
    bo = pd.DataFrame({"idx": np.arange(idx.max() + 1)}).merge(rest.assign(idx=idx), on="idx", how="left")
    return df.iloc[:N_START], bo


def live(bo):
    """1 if the last manifest row is a design still running, else 0."""
    return int(len(bo) > 0 and bo.status.iloc[-1] in ("running", "pending") and pd.isna(bo[CNR].iloc[-1]))


def adherence(bo, k=None):
    """(in window, with MPXI measured) among the first k optimizer designs."""
    b = (bo if k is None else bo[bo.idx < k]).dropna(subset=[MPXI])
    return int(inside(b).sum()), len(b)


def minutes(log):
    """Wall time of one pipeline run, from its log's first start stamp to its
    last completion stamp; NaN if either is missing."""
    try:
        text = open(log, errors="replace").read()
    except OSError:
        return np.nan
    t0 = t1 = None
    for kind, hms, tz, year in STAMP.findall(text):
        fmt = "%b %d %I:%M:%S %p %Y" if hms.endswith("M") else "%b %d %H:%M:%S %Y"
        t = datetime.strptime(f"{hms} {year}", fmt) - timedelta(hours=TZ_HOURS.get(tz, 0))
        if kind == "SAI Pipeline":
            t0 = t0 or t
        else:
            t1 = t
    return (t1 - t0).total_seconds() / 60 if t0 and t1 and t1 >= t0 else np.nan


def top3(rows):
    ok = rows.dropna(subset=[CNR])
    return ok[inside(ok)].nlargest(3, CNR)


def params(r):
    """The design vector of a row, counts as integers (the rebuild passes them on)."""
    return {k: int(r[k]) if k.startswith("n_") else r[k] for k in PARAMS}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--h2h", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--at", type=int, default=80, help="optimizer designs at the main comparison point")
    ap.add_argument("--allow_partial", action="store_true",
                    help="before every search has ended: write the candidates as candidates_partial.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows, data, cands = [], {}, []
    for s in sorted(os.listdir(a.mirror)):
        d = os.path.join(a.mirror, s)
        if not os.path.isdir(d):
            continue
        if arm_of(s) is None:
            print(f"skipped {s}: not a campaign search name")
            continue
        if not os.path.exists(os.path.join(d, "results_summary_mobo.csv")):
            print(f"skipped {s}: no results yet")
            continue
        arm, start_type = arm_of(s)
        start, bo, repeated = load_search(d)
        c = curve(start, bo)
        data[s] = (start, bo, c)
        own = list(bo.config_name) + (list(start.config) if start_type == "fresh" else [])
        mins = [minutes(os.path.join(d, "slurm_logs", "out", f"{n}_local.out"))
                for n in own if n not in repeated]
        mins = [m for m in mins if not np.isnan(m)]
        k_at, m_at = adherence(bo, a.at)
        k_end, m_end = adherence(bo)
        rows.append(dict(
            search=s, arm=arm, start=start_type, start_rows=len(start),
            repeated_designs=len(repeated),
            designs_run_on_eai=(len(start) if start_type == "fresh" else 0) + len(bo),
            optimizer_designs=len(bo), optimizer_with_cnr=int(bo[CNR].notna().sum()),
            lost=int(bo[CNR].isna().sum()) - live(bo), running=live(bo),
            best_inwindow_at=at(c, N_START + a.at), best_inwindow_end=at(c, len(c)),
            inwindow_at=k_at, measured_at=m_at, inwindow_end=k_end, measured_end=m_end,
            minutes_per_design=float(np.median(mins)) if mins else np.nan, timed_designs=len(mins)))
        work = dict(zip(bo.config_name, bo.work_dir))
        for _, r in top3(bo).iterrows():
            cands.append(dict(name=f"{s}__{r.config_name}", arm=arm, start=start_type, search=s,
                              config=r.config_name, inloop_cnr=r[CNR], mpxi=r[MPXI],
                              **params(r), ring2_at_cap=r[RING2] == RING2_CAP,
                              work_dir=work[r.config_name]))

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(a.out, "searches.csv"), index=False)
    rv = pd.read_csv(REF_CSV, float_precision="round_trip").set_index("config")
    refs = [dict(name=f"step2a__{n}", arm="ref", search="step2a", config=n, **params(rv.loc[n]),
                 work_dir=w) for n, w in REFS.items()]
    # Candidates are read at the final point: every search must have ended there.
    unfinished = [f"{s}: missing" for s in EXPECTED if s not in data] + [
        f"{s}: {len(bo)} of {N_BO} optimizer designs" + (", the last still running" if live(bo) else "")
        for s, (_, bo, _) in data.items() if len(bo) < N_BO or live(bo)]
    cand_file = "candidates_partial.csv" if unfinished else "candidates.csv"
    if not unfinished or a.allow_partial:
        pd.DataFrame(refs + cands).to_csv(os.path.join(a.out, cand_file), index=False)
    pd.set_option("display.width", 250)
    print(tab.drop(columns=["timed_designs"]).round(3).to_string(index=False))
    for arm, g in tab.groupby("arm"):
        print(f"\nArm {arm} ({ARMS[arm]}): {len(g)} searches, {int(g.designs_run_on_eai.sum())} designs "
              f"run on Empire AI ({int(g.optimizer_designs.sum())} chosen by the optimizer); in window: "
              f"{int(g.inwindow_at.sum())} of {int(g.measured_at.sum())} measured at {a.at}, "
              f"{int(g.inwindow_end.sum())} of {int(g.measured_end.sum())} at the end")

    # CCR's six complete arms: the paired comparison and their candidates.
    ccr = {k: pd.read_csv(os.path.join(a.h2h, f"results_h2h_{k}.csv"), float_precision="round_trip")
           for pair in CCR_PAIRS.values() for k in pair}
    pd.DataFrame([dict(name=f"ccr_{k}__{r.config}", arm=f"ccr_{k}", config=r.config,
                       inloop_cnr=r[CNR], mpxi=r[MPXI], **params(r))
                  for k, df in ccr.items() for _, r in top3(df.iloc[N_START:]).iterrows()]
                 ).to_csv(os.path.join(a.out, "ccr_candidates.csv"), index=False)

    prow = []
    for s, (two, five) in CCR_PAIRS.items():
        arms = {lab: data[n] for lab, n in (("eai_A", s), ("eai_B", "armB2_" + s)) if n in data}
        if not arms:
            continue
        arms.update({lab: (*ccr_arm(ccr[k]), None) for lab, k in (("ccr_2obj", two), ("ccr_5obj", five))})
        curves = {lab: (curve(st, bo), curve(st, bo, RING2_CAP)) for lab, (st, bo, _) in arms.items()}
        n = min([N_START + a.at] + [len(c) for c, _ in curves.values()])
        r = dict(start=s, proposals=n - N_START)
        for lab, (c, c960) in curves.items():
            k_in, k_m = adherence(arms[lab][1], n - N_START)
            r.update({lab: at(c, n), f"{lab}_ring2_le_960": at(c960, n), f"{lab}_inwindow": f"{k_in}/{k_m}"})
        prow.append(r)
    if prow:
        p = pd.DataFrame(prow)
        p.to_csv(os.path.join(a.out, "paired_ccr.csv"), index=False)
        print(f"\nPaired starts against CCR, best in-window in-loop CNR after the same number of proposals "
              f"(the furthest all reached, at most {a.at}; a lost proposal counts), descriptive only:")
        print(p.round(3).to_string(index=False))

    pairs = []
    for s in data:
        if s.startswith("armB2_") and s[6:] in data:
            cA, cB = data[s[6:]][2], data[s][2]
            n = min(len(cA), len(cB), N_START + N_BO)
            pairs.append(dict(start=s[6:], designs=n, cnr_alone=at(cA, n), several_objectives=at(cB, n)))
    if pairs:
        pd.DataFrame(pairs).to_csv(os.path.join(a.out, "pairs.csv"), index=False)
        print("\nShared starts, best in-window in-loop CNR at the furthest common point (in-loop only):")
        print(pd.DataFrame(pairs).round(3).to_string(index=False))

    # Best-so-far curves, one panel per arm. The band is drawn only where every
    # search of the arm has data, so its population never changes along x.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, arm in zip(axes, ARMS):
        cs = {s: c for s, (_, _, c) in data.items() if arm_of(s)[0] == arm and len(c)}
        for s, c in sorted(cs.items()):
            y = np.where(np.isfinite(c), c, np.nan)
            ax.plot(np.arange(1, len(c) + 1), y, lw=0.8, alpha=0.5,
                    color="#2a78d6" if arm_of(s)[1] == "paired" else "#8a8a8a")
        if cs:
            n = min(len(c) for c in cs.values())
            grid = np.array([c[:n] for c in cs.values()])
            with np.errstate(invalid="ignore"):   # -inf (none in the window yet) gives -inf or NaN
                q1, med, q3 = (np.percentile(grid, q, axis=0) for q in (25, 50, 75))
            x = np.arange(1, n + 1)
            ok = np.isfinite(q1)
            ax.plot(x, np.where(np.isfinite(med), med, np.nan), color="#1a1a1a", lw=2,
                    label=f"median of {len(cs)} searches")
            ax.fill_between(x[ok], q1[ok], q3[ok], color="#1a1a1a", alpha=0.12, lw=0,
                            label="interquartile range")
            ax.legend(frameon=False, loc="lower right")
        ax.axvline(N_START, color="#52514e", lw=1, ls=":")
        ax.set_title(f"{ARMS[arm]} (blue: paired starts)")
        ax.set_xlabel("designs (21 starting designs, then the optimizer's)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("best in-window in-loop CNR so far")
    fig.suptitle("Empire AI searches, MPXI steered into 1.5 to 2.5")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "best_so_far.png"), dpi=200)
    wrote = cand_file if not unfinished or a.allow_partial else "no candidates"
    print(f"\nwrote searches.csv, {wrote}, ccr_candidates.csv, paired_ccr.csv, pairs.csv and "
          f"best_so_far.png to {os.path.abspath(a.out)}")
    if unfinished:
        print(f"{len(unfinished)} searches have not reached the final point:\n  " + "\n  ".join(unfinished))
        if not a.allow_partial:
            sys.exit("ERROR: candidates are read at the final point; rerun once every search has ended "
                     "(--allow_partial writes candidates_partial.csv)")


if __name__ == "__main__":
    main()
