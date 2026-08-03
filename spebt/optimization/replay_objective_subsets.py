#!/usr/bin/env python3
"""
Would a smaller objective set have found the winning design sooner?

analyze_objective_redundancy.py showed the five objectives span about three
independent directions, and that CNR + MPXI + FWHM already puts 4 of the top 5
designs on its Pareto front. That is a statement about the FINAL archive. It
does not answer the question the campaign actually cares about: would the search
have GOT there faster?

This answers it by replay. The archive is treated as a fixed candidate pool, and
each objective subset re-runs the acquisition loop over that pool -- same GP,
same qLogNEHVI, same 21-design random start -- picking one archive point per
step. Nothing new is simulated; every objective value is one already measured.

WHAT THIS CAN AND CANNOT CLAIM
------------------------------
It CAN claim: given the same candidate pool, subset X selects the good designs
in fewer evaluations than the full set. That is a selection-efficiency result
and it is the one that matters for the next campaign's budget.

It CANNOT claim to be a clean counterfactual. The pool exists because the
FIVE-objective campaign chose to evaluate those 204 designs. A subset run in the
wild would have proposed different points and built a different pool. So the
replay is biased toward the five-objective set -- the subsets are being made to
search a space someone else mapped. Any subset that still wins here wins despite
that handicap, which makes a positive result trustworthy and a negative one
inconclusive.

The random baseline is the control for how much of any gap is just the pool
being rich in good designs.

Usage:
  python replay_objective_subsets.py --results_csv results/results_summary_mobo.csv
  python replay_objective_subsets.py --results_csv ... --n_repeats 10 --n_steps 80
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

import mobo_agent as ma
from analyze_objective_redundancy import SUBSETS, SUBSET_TAGS

# The real campaign seeded with 21 LHS designs before the first proposal, so the
# replay starts from the same budget. Changing this changes what "iterations to
# find" is measured from, so it is not a free knob.
N_INIT = 21
N_STEPS_DEFAULT = 80
N_REPEATS_DEFAULT = 5

# The outcome every subset is scored on, regardless of what it optimizes. A
# subset that does not include CNR is still judged by the CNR of what it picked
# -- that is the point of the "no CNR" control.
OUTCOME_COL = "cnr_sector_mean"


def fit_model(train_x_norm, train_y_std):
    """One SingleTaskGP per objective, wrapped in a ModelListGP."""
    models = []
    for i in range(train_y_std.shape[1]):
        gp = SingleTaskGP(train_x_norm, train_y_std[:, i:i + 1])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)
    return ModelListGP(*models)


def replay(x_norm, y_max, rng, n_steps, use_acqf=True):
    """Run one replay over the archive.

    x_norm  (N, d) design matrix, normalized to the campaign bounds
    y_max   (N, m) objectives already in maximization space
    returns the order in which archive rows were selected
    """
    n = len(x_norm)
    chosen = list(rng.choice(n, size=N_INIT, replace=False))

    for _ in range(n_steps):
        remaining = np.setdiff1d(np.arange(n), chosen)
        if len(remaining) == 0:
            break
        if not use_acqf:
            chosen.append(int(rng.choice(remaining)))
            continue

        tx = x_norm[chosen]
        ty = y_max[chosen]
        # Standardize on the observed subset only -- using the full archive's
        # mean and variance would leak information the search has not seen yet.
        mu, sd = ty.mean(dim=0, keepdim=True), ty.std(dim=0, keepdim=True).clamp(min=1e-6)
        ty_std = (ty - mu) / sd

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_model(tx, ty_std)
            acqf = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=(ty_std.min(dim=0).values - 0.1).tolist(),
                X_baseline=tx,
                prune_baseline=True,
                cache_root=False,
            )
            cand = x_norm[remaining].unsqueeze(1)  # (n_rem, q=1, d)
            with torch.no_grad():
                vals = acqf(cand)
        chosen.append(int(remaining[int(torch.argmax(vals))]))

    return chosen


def steps_to_reach(order, outcome, threshold):
    """Evaluations after the seed set before `threshold` is first met."""
    for k, idx in enumerate(order):
        if outcome[idx] >= threshold:
            return max(0, k - N_INIT + 1)
    return None


def main():
    ap = argparse.ArgumentParser(description="Replay the archive under smaller objective sets")
    ap.add_argument("--results_csv", required=True)
    ap.add_argument("--n_steps", type=int, default=N_STEPS_DEFAULT)
    ap.add_argument("--n_repeats", type=int, default=N_REPEATS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.results_csv):
        print(f"ERROR: results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv).dropna(subset=ma.OBJ_COLUMNS + ma.PARAM_NAMES)
    if {"d2_inner_mm", "d3_inner_mm"}.issubset(df.columns):
        df = df[df.apply(lambda r: ma.is_ring_ordering_ok(r.d2_inner_mm, r.d3_inner_mm), axis=1)]
    df = df.reset_index(drop=True)
    n = len(df)
    if n < N_INIT + 10:
        print(f"ERROR: only {n} designs in the archive; need at least {N_INIT + 10}")
        sys.exit(1)

    bounds = torch.tensor([ma.BOUNDS_MIN, ma.BOUNDS_MAX], dtype=torch.double)
    x_norm = normalize(torch.tensor(df[ma.PARAM_NAMES].values, dtype=torch.double), bounds)
    outcome = df[OUTCOME_COL].values

    best = outcome.max()
    top5 = np.sort(outcome)[-5]
    best_cfg = str(df.loc[int(np.argmax(outcome)), "config"])
    print(f"{n} designs in the archive, {args.n_repeats} repeats x {args.n_steps} steps")
    print(f"best CNR {best:.3f} ({best_cfg});  top-5 threshold {top5:.3f}")
    print(f"seed set {N_INIT} designs, matching the campaign's LHS start\n")

    runs = dict(SUBSETS)
    runs["random (control)"] = None
    tags = dict(SUBSET_TAGS)
    tags["random (control)"] = "random"

    print("=" * 78)
    print(f"EVALUATIONS AFTER THE SEED SET TO REACH EACH TARGET (mean +/- sd over repeats)")
    print("=" * 78)
    print(f"\n{'objective set':<26} {'-> top-5 design':>18} {'-> best design':>18} {'best CNR found':>16}")
    print("-" * 80)

    summary = {}
    for label, cols in runs.items():
        use_acqf = cols is not None
        if use_acqf:
            cols = [c for c in cols if c in df.columns]
            y_max = torch.tensor(
                df[cols].values * np.array(
                    [ma.OBJ_DIRECTIONS[ma.OBJ_COLUMNS.index(c)] for c in cols], dtype=float),
                dtype=torch.double)
        else:
            y_max = None

        hit5, hitbest, finals = [], [], []
        for r in range(args.n_repeats):
            rng = np.random.default_rng(args.seed + r)
            order = replay(x_norm, y_max, rng, args.n_steps, use_acqf=use_acqf)
            hit5.append(steps_to_reach(order, outcome, top5))
            hitbest.append(steps_to_reach(order, outcome, best))
            finals.append(outcome[order].max())

        def fmt(vals):
            got = [v for v in vals if v is not None]
            if not got:
                return "not found"
            s = f"{np.mean(got):.0f} +/- {np.std(got):.0f}"
            if len(got) < len(vals):
                s += f" ({len(got)}/{len(vals)})"
            return s

        summary[label] = (hit5, hitbest, finals)
        print(f"{label:<26} {fmt(hit5):>18} {fmt(hitbest):>18} "
              f"{np.mean(finals):>10.3f}+/-{np.std(finals):.3f}")

    print("""
Reading it:
  - "not found" means the target was never reached inside the step budget. A
    parenthesised (k/n) means only k of n repeats got there; the mean is over
    those k and therefore flatters the set.
  - The comparison that matters is each subset against "random (control)". The
    pool is dense in good designs, so random does better than intuition
    suggests, and a subset only earns a claim by beating it.
  - A subset beating "all five" here beats it despite searching a pool the
    five-objective campaign built. See the header note before quoting this as a
    counterfactual.""")


if __name__ == "__main__":
    main()
