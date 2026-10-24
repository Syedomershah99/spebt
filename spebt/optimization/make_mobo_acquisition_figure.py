#!/usr/bin/env python3
"""How the search chooses its next design, on one real step.

Dr. Yao (17 Oct 2026 meeting, 11:32 to 13:33) asked for the mechanism behind
two slide-7 lines, "A statistical model predicts the scores of untried
designs, with an uncertainty" and "The next design is the one expected to
improve the best trade-off the most", in terms an educated non-specialist can
follow, equations welcome. This backup figure shows one real step of the
search slide 7 animates, with the search's own model rebuilt.

Data: archive_backup/h2h/results_h2h_2obj_r5.csv, the two-objective arm of
head-to-head replicate 5. Its objectives are CNR (cnr_sector_mean) and MPXI
(mpxi_windowed_active_mean), both maximized: the set is read from the 2obj
branch of submit_mobo_headtohead.sh below and the directions come from
mobo_agent.py, so neither is restated here. 21 starting designs, then
proposals mobo_0000, mobo_0001, ... chosen one at a time (q = 1).

Step shown: proposal mobo_0012, the 13th design the model chose and the 34th
design overall ("design 34 of 99" in slide 7's counter), chosen from the 33
designs before it. Why this step:
  * rerunning the search's own code on those 33 designs proposes nearly the
    same design (normalized distance 0.0097 with torch seed 0, under the 0.02
    the search itself treats as "the same design"). Over seeds 0 to 19, 9 of 20
    reruns land within 0.02 and 18 of 20 within 0.05; all agree on aperture,
    aperture count, ring-I and ring-II counts and ring-II diameter, and differ
    only in ring III's diameter, along which the model is nearly flat. How
    often a rerun repeats the recorded choice depends on the seed and the
    procedure, so do not generalize this step to others;
  * the measured scores fall within one model SD of the prediction;
  * the predicted gain is large enough to see, and the design then extended
    the best trade-off, so the figure shows the mechanism working as described.

Model: rebuilt line for line from mobo_agent.get_next_candidate: inputs
normalized to the search bounds, each score standardized (mean, unbiased SD),
one default SingleTaskGP per score fitted by fit_gpytorch_mll, a ModelListGP,
reference point = each standardized score's minimum less 0.1, and
qLogNoisyExpectedHypervolumeImprovement over the 33 designs. On top of that
the script runs mobo_agent.get_next_candidate itself on the same 33 rows and
refuses to draw unless it returns the recorded design. The archive was made
with botorch 0.16.1 on CCR (requirements.txt); the Mac has 0.18.1. Both give
the same SingleTaskGP (RBF kernel, Standardize outcome transform) and the same
fitted hyperparameters, predictions and acquisition value at this step.

What is drawn, in raw score units (the model works in standardized units;
the map is linear per axis, so shapes and area ratios are unchanged):
  grey dots      the 33 designs simulated before the step
  green steps    their non-dominated set, "P: Pareto front so far"
  light fill     HV(P), the area that front covers, measured from the reference point
  blue cross     the model's prediction for the chosen design, mean +/- 2 SD
                 of the fitted score (not including measurement noise)
  coral fill     the area the design adds if it lands on its predicted mean
  blue dot       what the simulation then returned (right panel only: the
                 left panel is the state before design 34 was simulated)
The gain drawn is the gain at the predicted mean. Its expectation over the
prediction's spread, computed exactly in two dimensions, differs from it by
under 1% at this step, and the script checks that before drawing. The
search's own acquisition value is slightly higher, because qLogNEHVI also
treats the measured front as uncertain; all three are printed.

Usage:  python make_mobo_acquisition_figure.py [--out PATH]
"""
import argparse
import logging
import math
import os
import re
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import matplotlib.text

HERE = os.path.dirname(os.path.abspath(__file__))
ARCHIVE = os.path.join(HERE, "..", "archive_backup", "h2h", "results_h2h_2obj_r5.csv")
SUBMIT = os.path.join(HERE, "submit_mobo_headtohead.sh")
DEFAULT_OUT = os.path.join(HERE, "..", "..", "deck_figures", "mobo_acquisition.png")

# The objective set this arm ran with, read from its submit script rather than
# restated, then handed to mobo_agent the way the controller's job did.
_sub = open(SUBMIT).read()
_m = re.search(r'"\$ARM" == "2obj" \]\]; then\s*\n\s*export MOBO_OBJECTIVES="([^"]+)"', _sub)
if _m is None:
    raise SystemExit(f"ERROR: no 2obj MOBO_OBJECTIVES export found in {SUBMIT}")
os.environ["MOBO_OBJECTIVES"] = _m.group(1)
sys.path.insert(0, HERE)
import mobo_agent as ma  # noqa: E402

from botorch.models import SingleTaskGP  # noqa: E402
from botorch.models.model_list_gp_regression import ModelListGP  # noqa: E402
from botorch.fit import fit_gpytorch_mll  # noqa: E402
from botorch.utils.transforms import normalize  # noqa: E402
from botorch.acquisition.multi_objective.logei import (  # noqa: E402
    qLogNoisyExpectedHypervolumeImprovement)
from gpytorch.mlls import ExactMarginalLogLikelihood  # noqa: E402

CNR, MPXI = "cnr_sector_mean", "mpxi_windowed_active_mean"
if sorted(ma.OBJ_COLUMNS) != sorted([CNR, MPXI]):
    raise SystemExit(f"ERROR: the 2obj arm optimizes {ma.OBJ_COLUMNS}, not CNR and MPXI")
if any(ma.OBJ_DIRECTIONS[ma.OBJ_COLUMNS.index(c)] != 1.0 for c in (CNR, MPXI)):
    raise SystemExit("ERROR: this figure assumes both scores are maximized")
I_CNR, I_MPXI = ma.OBJ_COLUMNS.index(CNR), ma.OBJ_COLUMNS.index(MPXI)

N_START = 21          # space-filling starting designs, run before any model
K = 12                # proposal index of the step shown (mobo_0012)
DEDUP = 0.02          # mobo_agent's "same design" distance in normalized units

# Deck palette.
UB_BLUE, CORAL, GREEN = "#005BBB", "#E56D54", "#1E8449"
INK, GREY, FAINT, TUNGSTEN = "#1A1A1A", "#8E8E8E", "#C9CDD2", "#6E6E73"
AREA = "#E1EBF6"      # UB blue at 12% on white: the covered area, kept light
FONT = "Helvetica Neue"
MIN_CAP_PX = 36       # legibility floor for every glyph in the figure

# Sizes in points; the figure is placed on the slide at 7.2 x 4.8 in, scale 1.
FS_TITLE, FS_LABEL, FS_TICK, FS_ANN, FS_NOTE = 14.5, 13.5, 13, 13.5, 13
FW, FH, DPI = 7.2, 4.8, 300
BOX_LW = 1.2          # zoom box on the left panel and the right panel's frame

# Panel boxes in inches (left, bottom, width, height) and data windows.
LEFT_BOX = (0.50, 0.98, 2.84, 3.32)
RIGHT_BOX = (3.92, 0.98, 3.20, 3.32)
LEFT_LIM = ((0.8, 3.8), (0.5, 5.0))
ZOOM_LIM = ((0.95, 2.42), (4.0, 4.9))
TITLE_Y, NOTE_Y = 4.52, 0.13       # baselines, inches from the bottom


# ---------------------------------------------------------------- data
def load():
    """The arm in evaluation order, with the assumptions the step relies on checked."""
    d = pd.read_csv(ARCHIVE)
    cfg = d["config"].astype(str)
    if len(d) != 99:
        raise SystemExit(f"ERROR: expected 99 scored designs, found {len(d)}")
    start = cfg.str.startswith("lhs").to_numpy()
    if not (start[:N_START].all() and cfg[N_START:].str.startswith("mobo").all()):
        raise SystemExit("ERROR: rows are not 21 starting designs then model proposals")
    idx = cfg.str.extract(r"^mobo_(\d{4})_")[0]
    d["prop"] = pd.to_numeric(idx, errors="coerce")
    if not np.all(np.diff(d["prop"][N_START:].to_numpy()) > 0):
        raise SystemExit("ERROR: proposals are not in file order of proposal index")
    for name, ap, nap in zip(cfg[N_START:], d.aperture_diam_mm[N_START:], d.n_apertures[N_START:]):
        m = re.search(r"_ap([0-9.]+)_nap(\d+)_", name)
        if m is None or abs(float(m.group(1)) - ap) > 6e-5 or int(m.group(2)) != nap:
            raise SystemExit(f"ERROR: {name} disagrees with its parameter columns")
    return d


def history(d, k):
    """What the search's results CSV held when it chose proposal k: every
    starting design and every proposal before k, in the order they landed."""
    before = d[d["prop"].isna() | (d["prop"] < k)]
    if not before.index.equals(pd.RangeIndex(len(before))):
        raise SystemExit("ERROR: the rows before the step are not the first rows of the file")
    if sorted(before["prop"].dropna().astype(int)) != list(range(k)):
        raise SystemExit(f"ERROR: a proposal before {k} is missing, so the training set "
                         f"at this step cannot be rebuilt from the archive")
    chosen = d[d["prop"] == k]
    if len(chosen) != 1:
        raise SystemExit(f"ERROR: expected one row for proposal {k}")
    return before, chosen.iloc[0]


# ---------------------------------------------------------------- model
def rerun_search(before):
    """Run mobo_agent.get_next_candidate itself on the rows the search had."""
    msgs = []

    class Grab(logging.Handler):
        def emit(self, rec):
            msgs.append(rec.getMessage())

    grab = Grab(level=logging.INFO)
    ma.logger.addHandler(grab)
    quiet = [(h, h.level) for h in logging.getLogger().handlers]
    for h, _ in quiet:
        h.setLevel(logging.WARNING)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results_summary_mobo.csv")
            before.drop(columns=["prop"]).to_csv(path, index=False)
            torch.manual_seed(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cand = ma.get_next_candidate(path)
    finally:
        ma.logger.removeHandler(grab)
        for h, lv in quiet:
            h.setLevel(lv)
    acq = [float(m.split(":")[1]) for m in msgs if m.startswith("Acquisition value:")]
    return np.array(cand, dtype=float), (acq[-1] if acq else float("nan"))


def refit(before):
    """mobo_agent.get_next_candidate's model, rebuilt so its posterior can be read."""
    ok = before.apply(lambda r: ma.is_ring_ordering_ok(r["d2_inner_mm"], r["d3_inner_mm"]), axis=1)
    valid = before.dropna(subset=ma.OBJ_COLUMNS)
    if not ok.all() or len(valid) != len(before):
        raise SystemExit("ERROR: this step's history has excluded or failed rows; "
                         "the penalty path of mobo_agent would apply and is not rebuilt here")
    train_x = torch.tensor(valid[ma.PARAM_NAMES].values, dtype=torch.double)
    train_y = (torch.tensor(valid[ma.OBJ_COLUMNS].values, dtype=torch.double)
               * torch.tensor(ma.OBJ_DIRECTIONS, dtype=torch.double))
    bounds = torch.tensor([ma.BOUNDS_MIN, ma.BOUNDS_MAX], dtype=torch.double)
    if ma.report_out_of_bounds(train_x):
        raise SystemExit("ERROR: training designs lie outside the search bounds")
    x_norm = normalize(train_x, bounds)
    y_mean = train_y.mean(dim=0, keepdim=True)
    y_std = train_y.std(dim=0, keepdim=True).clamp(min=1e-6)
    y_stdzd = (train_y - y_mean) / y_std
    gps = []
    for i in range(y_stdzd.shape[1]):
        gp = SingleTaskGP(x_norm, y_stdzd[:, i:i + 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        gps.append(gp)
    model = ModelListGP(*gps)
    ref_std = y_stdzd.min(dim=0).values - 0.1
    torch.manual_seed(0)
    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=ref_std.tolist(), X_baseline=x_norm,
        prune_baseline=True, cache_root=False)
    return dict(model=model, acqf=acqf, bounds=bounds, y_mean=y_mean[0], y_std=y_std[0],
                ref_std=ref_std, gps=gps)


def predict(fit, params):
    """Posterior mean and SD of each score at one design, in raw units."""
    x = normalize(torch.tensor(params, dtype=torch.double)[None], fit["bounds"])
    out = {}
    with torch.no_grad():
        for noisy in (False, True):
            p = fit["model"].posterior(x, observation_noise=noisy)
            mu = (fit["y_mean"] + p.mean[0] * fit["y_std"]).numpy()
            sd = (p.variance[0].sqrt() * fit["y_std"]).numpy()
            out[noisy] = (mu, sd)
        log_acq = fit["acqf"](x[None]).item()
    return out[False][0], out[False][1], out[True][1], log_acq, x


def alternatives(fit, F, ref, xy, e_chosen, log_acq_chosen, n=8192, seed=0):
    """The chosen design against random designs the search could have tried,
    scored by the same refit model: expected gain over the measured front, and
    the search's own acquisition value."""
    sob = torch.quasirandom.SobolEngine(dimension=ma.DIM, scramble=True, seed=seed)
    X = sob.draw(n, dtype=torch.double)
    lo, hi = fit["bounds"]
    phys = lo + X * (hi - lo)
    ok = torch.tensor([ma.is_feasible_full(*[float(v) for v in row]) for row in phys])
    X = X[ok]
    with torch.no_grad():
        post = fit["model"].posterior(X)
        mu = (fit["y_mean"] + post.mean * fit["y_std"]).numpy()[:, xy]
        sd = (post.variance.sqrt() * fit["y_std"]).numpy()[:, xy]
        acq = torch.cat([fit["acqf"](c.unsqueeze(1)) for c in torch.split(X, 512)]).numpy()
    e = np.array([expected_gain(F, ref, m, s) for m, s in zip(mu, sd)])
    return dict(n=len(X), best_e=float(e.max()), n_e_better=int((e > e_chosen).sum()),
                best_acq=float(np.exp(acq.max())), n_acq_better=int((acq > log_acq_chosen).sum()))


# ---------------------------------------------------------------- geometry
def front(P):
    """Non-dominated rows of P (both columns maximized), sorted by column 0."""
    keep = [i for i in range(len(P))
            if not (np.all(P >= P[i], axis=1) & np.any(P > P[i], axis=1)).any()]
    return sorted(keep, key=lambda i: P[i, 0])


def check_front(P, f):
    fx, fy = P[f, 0], P[f, 1]
    if len(f) < 2 or not (np.all(np.diff(fx) > 0) and np.all(np.diff(fy) < 0)):
        raise SystemExit("ERROR: the front is not a strict staircase")


def hypervolume(F, ref):
    """Area dominated by the staircase F (sorted by x, y falling) above ref."""
    F = F[np.all(F > ref, axis=1)]
    area, x_prev = 0.0, ref[0]
    for x, y in F:
        area += (x - x_prev) * (y - ref[1])
        x_prev = x
    return area


def covered_polygon(F, ref):
    v = [(ref[0], ref[1]), (ref[0], F[0, 1])]
    for i in range(len(F)):
        v.append((F[i, 0], F[i, 1]))
        if i + 1 < len(F):
            v.append((F[i, 0], F[i + 1, 1]))
    v.append((F[-1, 0], ref[1]))
    return np.array(v)


def stair(F):
    """The best-trade-off line as slide 7 draws it: from each front design drop
    to the next one's CNR, then run across to it."""
    v = []
    for a, b in zip(F[:-1], F[1:]):
        v += [(a[0], a[1]), (a[0], b[1])]
    v.append(tuple(F[-1]))
    return np.array(v)


def gain_segments(F, ref, p):
    """Vertical strips of the area point p adds: (x_lo, x_hi, floor), floor
    being the front's height there. The front covers (x_{i-1}, x_i] up to y_i."""
    lefts = [ref[0]] + list(F[:, 0])
    rights = list(F[:, 0]) + [np.inf]
    floors = list(F[:, 1]) + [ref[1]]
    segs = []
    for a, b, h in zip(lefts, rights, floors):
        lo, hi = max(a, ref[0]), min(b, p[0])
        if hi > lo and p[1] > h:
            segs.append((lo, hi, max(h, ref[1])))
    return segs


def gain_polygon(segs, p):
    v = []
    for lo, hi, h in segs:
        v += [(lo, h), (hi, h)]
    v += [(segs[-1][1], p[1]), (segs[0][0], p[1])]
    return np.array(v)


def shoelace(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def tail_integral(a, mu, sd):
    """Integral from a to infinity of P(Y >= t) dt for Y ~ N(mu, sd^2)."""
    al = (a - mu) / sd
    return sd * (norm.pdf(al) - al * norm.sf(al))


def expected_gain(F, ref, mu, sd):
    """Exact expected area added by a design whose two scores are independent
    normals (one GP per score): the integral, over every point the front does
    not yet cover, of the chance the design covers it."""
    lefts = [ref[0]] + list(F[:, 0])
    rights = list(F[:, 0]) + [np.inf]
    floors = list(F[:, 1]) + [ref[1]]
    total = 0.0
    for a, b, h in zip(lefts, rights, floors):
        ix = tail_integral(a, mu[0], sd[0])
        if np.isfinite(b):
            ix -= tail_integral(b, mu[0], sd[0])
        total += ix * tail_integral(h, mu[1], sd[1])
    return total


def expected_gain_mc(F, ref, mu, sd, n=1_000_000, seed=0):
    Y = mu + sd * np.random.default_rng(seed).standard_normal((n, 2))
    gx, gy = np.maximum(Y[:, 0], ref[0]), np.maximum(Y[:, 1], ref[1])
    box = (gx - ref[0]) * (gy - ref[1])
    covered = np.zeros(n)
    edges = np.concatenate([[ref[0]], F[:, 0]])
    for i in range(len(F)):
        w = np.clip(np.minimum(gx, edges[i + 1]) - edges[i], 0, None)
        covered += w * np.clip(np.minimum(gy, F[i, 1]) - ref[1], 0, None)
    return float(np.mean(box - covered))


# ---------------------------------------------------------------- drawing
def cap_px(fp):
    """Rendered height of a capital H in this font, in output pixels."""
    f = plt.figure(figsize=(1, 1), dpi=DPI)
    f.text(0.2, 0.3, "H", fontproperties=fp, color="black")
    f.canvas.draw()
    ink = np.asarray(f.canvas.buffer_rgba())[..., 0] < 128
    plt.close(f)
    rows = np.where(ink.any(axis=1))[0]
    return int(rows.max() - rows.min() + 1)


def check_layout(fig, labels, marks, lines):
    """No text collides with other text, a design, a bar or the front line, and
    every glyph clears the legibility floor."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    for t in texts:
        h = cap_px(t.get_fontproperties())
        if h < MIN_CAP_PX:
            raise SystemExit(f"ERROR: '{t.get_text()}' has cap height {h} px < {MIN_CAP_PX}")
        if t.get_fontname() != FONT:
            raise SystemExit(f"ERROR: '{t.get_text()}' rendered in {t.get_fontname()}, not {FONT}")
    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0).padded(4)) for t in texts]
    W, H = fig.canvas.get_width_height()
    for i, (t, b) in enumerate(boxes):
        if b.x0 < 0 or b.y0 < 0 or b.x1 > W or b.y1 > H:
            raise SystemExit(f"ERROR: '{t.get_text()}' leaves the figure")
        for u, c in boxes[i + 1:]:
            g = getattr(t, "group", None)
            if b.overlaps(c) and not (g is not None and g == getattr(u, "group", None)):
                raise SystemExit(f"ERROR: text overlap: '{t.get_text()}' and '{u.get_text()}'")
    for t in labels:
        b = t.get_window_extent(r).padded(5)
        for ax, pts, rad in marks:
            for p in ax.transData.transform(pts):
                if (b.x0 - rad < p[0] < b.x1 + rad) and (b.y0 - rad < p[1] < b.y1 + rad):
                    raise SystemExit(f"ERROR: label '{t.get_text()}' covers a marker at {p}")
        for ax, seg, name, owner in lines:
            if owner is t:
                continue          # a leader may touch its own label
            q = ax.transData.transform(seg)
            for s in np.linspace(0, 1, 200):
                for a0, a1 in zip(q[:-1], q[1:]):
                    c = a0 + (a1 - a0) * s
                    if b.x0 < c[0] < b.x1 and b.y0 < c[1] < b.y1:
                        raise SystemExit(f"ERROR: label '{t.get_text()}' sits on the {name}")


def style_axes(ax, xlim, ylim, xticks, yticks, fmt):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([fmt(t) for t in xticks])
    ax.set_yticklabels([fmt(t) for t in yticks])
    ax.tick_params(colors=TUNGSTEN, labelsize=FS_TICK, length=4, width=1.0, pad=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(FAINT)
        ax.spines[sp].set_linewidth(1.2)
    ax.set_facecolor("white")


def draw_common(ax, P, F, ref, cov, gain_v, mu, sd, act, dot_s, big_s):
    """Everything the search knew when it chose (designs, front, area, the
    prediction and its gain), plus the measured result when act is given."""
    ax.add_patch(Polygon(cov, closed=True, facecolor=AREA, edgecolor="none", zorder=1))
    ax.add_patch(Polygon(gain_v, closed=True, facecolor=CORAL, alpha=0.55, edgecolor=CORAL,
                         lw=1.2, zorder=2))
    st = stair(F)
    ax.plot(st[:, 0], st[:, 1], color=GREEN, lw=2.6, solid_capstyle="round",
            solid_joinstyle="miter", zorder=3)
    on = set(map(tuple, F))
    off = np.array([p for p in P if tuple(p) not in on])
    ax.scatter(off[:, 0], off[:, 1], s=dot_s, color=GREY, edgecolors="white", linewidths=0.8,
               zorder=4)
    ax.scatter(F[:, 0], F[:, 1], s=dot_s, color=GREY, edgecolors="white", linewidths=0.8,
               zorder=5)
    ax.errorbar([mu[0]], [mu[1]], xerr=[[2 * sd[0]], [2 * sd[0]]], yerr=[[2 * sd[1]], [2 * sd[1]]],
                fmt="none", ecolor=UB_BLUE, elinewidth=1.8, capsize=big_s ** 0.5 * 0.45,
                capthick=1.8, zorder=6)
    ax.scatter([mu[0]], [mu[1]], s=big_s, facecolor="white", edgecolor=UB_BLUE, linewidths=2.0,
               zorder=7)
    if act is not None:
        ax.scatter([act[0]], [act[1]], s=big_s, color=UB_BLUE, edgecolors="white",
                   linewidths=1.0, zorder=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()

    d = load()
    before, chosen = history(d, K)
    n_before = len(before)
    params = chosen[ma.PARAM_NAMES].to_numpy(float)

    # 1. The search's own code, rerun on what it had: does it pick this design?
    rerun, rerun_log_acq = rerun_search(before)
    rng = np.array(ma.BOUNDS_MAX) - np.array(ma.BOUNDS_MIN)
    dist = float(np.linalg.norm((rerun - params) / rng))
    if dist >= DEDUP:
        raise SystemExit(f"ERROR: rerunning mobo_agent on the {n_before} earlier designs picks a "
                         f"design {dist:.3f} away from the recorded one; this step does not "
                         f"illustrate the recorded choice")

    # 2. The same model, rebuilt, read at the recorded design.
    fit = refit(before)
    mu_o, sd_o, sdn_o, log_acq, _ = predict(fit, params)
    ref_o = (fit["y_mean"] + fit["ref_std"] * fit["y_std"]).numpy()
    scale = float(fit["y_std"][I_CNR] * fit["y_std"][I_MPXI])

    # Plot coordinates: x = MPXI, y = CNR.
    xy = [I_MPXI, I_CNR]
    mu, sd, sdn, ref = mu_o[xy], sd_o[xy], sdn_o[xy], ref_o[xy]
    P = before[[MPXI, CNR]].to_numpy(float)
    act = chosen[[MPXI, CNR]].to_numpy(float)
    f = front(P)
    check_front(P, f)
    F = P[f]

    hv0 = hypervolume(F, ref)
    cov = covered_polygon(F, ref)
    if abs(shoelace(cov) - hv0) > 1e-9:
        raise SystemExit("ERROR: the shaded area does not equal the computed hypervolume")
    segs = gain_segments(F, ref, mu)
    if not segs:
        raise SystemExit("ERROR: the predicted mean adds no area; nothing to highlight")
    gain_v = gain_polygon(segs, mu)
    Pm = np.vstack([P, mu])
    gain_mean = hypervolume(Pm[front(Pm)], ref) - hv0
    if abs(shoelace(gain_v) - gain_mean) > 1e-9:
        raise SystemExit("ERROR: the highlighted area does not equal the gain at the predicted mean")
    e_gain = expected_gain(F, ref, mu, sd)
    e_gain_mc = expected_gain_mc(F, ref, mu, sd)
    if abs(e_gain_mc - e_gain) > 0.005 * e_gain:
        raise SystemExit(f"ERROR: exact ({e_gain:.5f}) and sampled ({e_gain_mc:.5f}) "
                         f"expected gain disagree")
    if abs(e_gain - gain_mean) > 0.01 * e_gain:
        raise SystemExit(f"ERROR: expected gain {e_gain:.4f} differs from the drawn gain at the mean "
                         f"{gain_mean:.4f} by over 1%; the highlight would misstate it")
    nehvi = math.exp(log_acq) * scale
    alt = alternatives(fit, F, ref, xy, e_gain, log_acq)
    Pa = np.vstack([P, act])
    hv1 = hypervolume(Pa[front(Pa)], ref)
    gain_act = hv1 - hv0
    z = (act - mu) / sd
    if not np.all(np.abs(z) < 2):
        raise SystemExit(f"ERROR: the measured scores fall outside the drawn +/- 2 SD (z = {z})")
    if gain_act <= 0:
        raise SystemExit("ERROR: the design did not extend the best trade-off")

    # ------------------------------------------------------------ figure
    # Math as on the slides (Dr. Yao, 22 Oct): P italic, HV upright, Times New Roman.
    plt.rcParams.update({"font.family": FONT, "font.size": FS_ANN, "axes.unicode_minus": True,
                         "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                         "mathtext.it": "Times New Roman:italic",
                         "mathtext.bf": "Times New Roman:bold",
                         "mathtext.bfit": "Times New Roman:bold:italic"})
    fig = plt.figure(figsize=(FW, FH), dpi=DPI)
    fig.patch.set_facecolor("white")
    axL = fig.add_axes([LEFT_BOX[0] / FW, LEFT_BOX[1] / FH, LEFT_BOX[2] / FW, LEFT_BOX[3] / FH])
    axR = fig.add_axes([RIGHT_BOX[0] / FW, RIGHT_BOX[1] / FH, RIGHT_BOX[2] / FW, RIGHT_BOX[3] / FH])

    style_axes(axL, *LEFT_LIM, [1, 2, 3], [1, 2, 3, 4, 5], lambda t: f"{t:g}")
    style_axes(axR, *ZOOM_LIM, [1.0, 1.5, 2.0], [4.0, 4.2, 4.4, 4.6, 4.8], lambda t: f"{t:.1f}")
    for ax in (axL, axR):
        ax.set_xlabel("multiplexing (MPXI)", fontsize=FS_LABEL, color=INK, labelpad=5)
        ax.set_ylabel("CNR", fontsize=FS_LABEL, color=INK, labelpad=4)
    # The right panel is the box drawn on the left one: same frame, same line.
    for sp in axR.spines.values():
        sp.set_visible(True)
        sp.set_color(TUNGSTEN)
        sp.set_linewidth(BOX_LW)

    S_DOT_L, S_BIG_L, S_DOT_R, S_BIG_R = 20, 26, 46, 70
    # The left panel is the state before design 34 was simulated, so it carries
    # the prediction but not the result.
    draw_common(axL, P, F, ref, cov, gain_v, mu, sd, None, dot_s=S_DOT_L, big_s=S_BIG_L)
    draw_common(axR, P, F, ref, cov, gain_v, mu, sd, act, dot_s=S_DOT_R, big_s=S_BIG_R)

    # Reference point: left panel only, it lies far below the zoom window.
    axL.scatter([ref[0]], [ref[1]], marker="s", s=22, color=INK, zorder=9)
    # Where the right panel is, drawn on the left one.
    (zx0, zx1), (zy0, zy1) = ZOOM_LIM
    axL.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=TUNGSTEN,
                            lw=BOX_LW, zorder=3.5))   # under the dots

    labels = []

    def label(ax, x, y, s, group=None, **kw):
        """Annotation text. Lines of one label share a group: they may touch
        each other, never anything else."""
        kw.setdefault("fontsize", FS_ANN)
        t = ax.text(x, y, s, zorder=10, **kw)
        t.group = group
        labels.append(t)
        return t

    pct_pred = 100 * gain_mean / hv0
    pct_act = 100 * gain_act / hv0

    # Left panel: the front, the area it covers, and where that area is measured from.
    # P and HV(P) named as Dr. Yao's R7-rY slide 7 defines them (23 Oct): Pareto
    # front of nondominated designs, expected hypervolume improvement; no "trade-off".
    t_front = label(axL, 2.50, 4.58, "$\\mathbfit{P}$: Pareto front\nso far", color=GREEN, weight="bold",
                    ha="left", va="center", linespacing=1.05)
    # Leader to the horizontal run at CNR 3.60, between the third and fourth
    # front designs.
    lead = np.array([[2.58, 4.26], [2.45, F[3, 1] + 0.02]])
    if not F[2, 0] < lead[1, 0] < F[3, 0]:
        raise SystemExit("ERROR: the front label's leader no longer ends on the front")
    axL.plot(lead[:, 0], lead[:, 1], color=TUNGSTEN, lw=1.0, zorder=10)
    label(axL, 2.45, 2.25, "$\\mathrm{HV}(P)$", color=UB_BLUE, ha="center", va="center",
          fontsize=FS_ANN + 3)
    label(axL, 1.12, 1.13, "reference point", group="ref", color=INK, ha="left", va="center")
    label(axL, 1.12, 0.84, "(lowest scores \u2212 0.1 SD)", group="ref", color=TUNGSTEN,
          ha="left", va="center", fontsize=FS_NOTE)

    # Right panel: the chosen design, as predicted and as simulated.
    # QA, 24 Oct: the percentages name what they measure, the hypervolume, since
    # beside a CNR axis a bare "+2.0%" reads as a CNR change.
    label(axR, 1.0, 4.80, "predicted", group="gain", color=CORAL, weight="bold",
          ha="left", va="center")
    label(axR, 1.0, 4.725, f"$\\mathrm{{HV}}(P)$ +{pct_pred:.1f}%", group="gain", color=CORAL,
          weight="medium", ha="left", va="center")
    label(axR, mu[0] + 2 * sd[0] + 0.035, mu[1] + 1.5 * sd[1], "prediction\n$\\mathbfit{\\mu}\\ \\mathbf{\\pm}\\ \\mathbf{2}\\mathbfit{\\sigma}$",
          color=UB_BLUE, weight="bold", ha="left", va="center", linespacing=1.05)
    label(axR, act[0] + 0.045, act[1] + 0.003, "actual", group="act", color=UB_BLUE,
          weight="bold", ha="left", va="center")
    label(axR, act[0] + 0.045, act[1] - 0.058, f"$\\mathrm{{HV}}(P)$ +{pct_act:.1f}%", group="act",
          color=UB_BLUE, weight="medium", ha="left", va="center")

    t_left = fig.text(LEFT_BOX[0] / FW, TITLE_Y / FH, f"The first {n_before} designs",
                      fontsize=FS_TITLE, color=INK, weight="bold", ha="left", va="baseline")
    t_right = fig.text(RIGHT_BOX[0] / FW, TITLE_Y / FH,
                       f"Design {n_before + 1}: predicted, then simulated",
                       fontsize=FS_TITLE, color=INK, weight="bold", ha="left", va="baseline")
    note = fig.text(LEFT_BOX[0] / FW, NOTE_Y / FH,
                    f"The search\u2019s model, refit to the first {n_before} designs, "
                    f"proposes nearly the same design.",
                    fontsize=FS_NOTE, color=TUNGSTEN, style="italic", ha="left", va="baseline")
    # The right title starts where the right panel's tick labels do.
    fig.canvas.draw()
    rr = fig.canvas.get_renderer()
    x_ticks = min(t.get_window_extent(rr).x0 for t in axR.get_yticklabels())
    t_right.set_x(x_ticks / fig.bbox.width)

    # Collision and legibility checks over everything drawn.
    def in_view(ax, pts):
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        pts = np.atleast_2d(pts)
        return pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)]

    def radius(s):
        return math.sqrt(s) / 2 * DPI / 72

    bars = [np.array([[mu[0] - 2 * sd[0], mu[1]], [mu[0] + 2 * sd[0], mu[1]]]),
            np.array([[mu[0], mu[1] - 2 * sd[1]], [mu[0], mu[1] + 2 * sd[1]]])]
    box = np.array([[zx0, zy0], [zx1, zy0], [zx1, zy1], [zx0, zy1], [zx0, zy0]])
    marks = []
    for ax, s_dot, s_big, big in ((axL, S_DOT_L, S_BIG_L, [mu]),
                                  (axR, S_DOT_R, S_BIG_R, [mu, act])):
        marks.append((ax, in_view(ax, P), radius(s_dot)))
        marks.append((ax, in_view(ax, np.vstack(big)), radius(s_big)))
    marks.append((axL, np.atleast_2d(ref), radius(22)))
    lines = [(axL, stair(F), "front", None), (axR, stair(F), "front", None),
             (axL, box, "zoom box", None), (axL, lead, "leader", t_front)]
    lines += [(ax, b, "error bar", None) for ax in (axL, axR) for b in bars]
    check_layout(fig, labels, marks, lines)

    fig.savefig(a.out, dpi=DPI, facecolor="white")

    # ------------------------------------------------------------ report
    print(f"wrote {a.out}")
    print(f"step: proposal {chosen['config']}")
    print(f"  = model choice {K + 1}, design {n_before + 1} of {len(d)}, chosen from {n_before}")
    print("  parameters: " + ", ".join(f"{n}={v:.6g}" for n, v in zip(ma.PARAM_NAMES, params)))
    print("  rerun of mobo_agent.get_next_candidate: "
          + ", ".join(f"{n}={v:.6g}" for n, v in zip(ma.PARAM_NAMES, rerun))
          + f"  (normalized distance {dist:.4f}; log-acq {rerun_log_acq:.4f})")
    for name, i in (("MPXI", 0), ("CNR", 1)):
        print(f"  {name}: predicted {mu[i]:.4f} +/- {sd[i]:.4f} (SD of the fitted score; "
              f"{sdn[i]:.4f} with measurement noise), actual {act[i]:.4f}, z = {z[i]:+.2f}")
    print(f"  reference point: MPXI {ref[0]:.5f}, CNR {ref[1]:.5f} "
          f"(standardized {fit['ref_std'][I_MPXI]:.5f}, {fit['ref_std'][I_CNR]:.5f})")
    print(f"  front before ({len(F)} designs): " + "; ".join(f"({x:.4f}, {y:.4f})" for x, y in F))
    print(f"  area covered before: {hv0:.5f} raw units ({hv0 / scale:.5f} standardized)")
    print(f"  area after the actual design: {hv1:.5f} (+{gain_act:.5f}, +{100 * gain_act / hv0:.2f}%)")
    print(f"  gain at the predicted mean (drawn): {gain_mean:.5f} (+{100 * gain_mean / hv0:.2f}%)")
    print(f"  expected gain over the prediction's spread: exact {e_gain:.5f}, sampled "
          f"{e_gain_mc:.5f} (+{100 * e_gain / hv0:.2f}%)")
    print(f"  search's acquisition, exp(qLogNEHVI): {nehvi:.5f} raw "
          f"({math.exp(log_acq):.5f} standardized, +{100 * nehvi / hv0:.2f}%)")
    print(f"  against {alt['n']} random feasible designs under the same model: expected gain "
          f"max {alt['best_e']:.5f} (+{100 * alt['best_e'] / hv0:.2f}%), larger than design "
          f"{n_before + 1}'s for {alt['n_e_better']}; acquisition max {alt['best_acq'] * scale:.5f}, "
          f"larger for {alt['n_acq_better']}")
    for gp, name in zip(fit["gps"], ma.OBJ_COLUMNS):
        k = gp.covar_module
        ls = (k.base_kernel.lengthscale if hasattr(k, "base_kernel") else k.lengthscale)
        print(f"  GP[{name}]: {type(k).__name__}, lengthscales "
              + ", ".join(f"{v:.3f}" for v in ls.detach().numpy().ravel())
              + f", noise var {gp.likelihood.noise.item():.5f} (standardized)")
    return fig, labels


if __name__ == "__main__":
    main()
