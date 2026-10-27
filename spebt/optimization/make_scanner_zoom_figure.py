#!/usr/bin/env python3
"""Slide 8 (R5) scanner figure: the simulated SC-SPECT drawn to scale, with two zoom-ins.

Dr. Yao, 17 Oct 2026 meeting (24:45 to 31:58): show an aperture with a zoom-in, as in
Harsh Tripathi's SNMMI figure, and give the audience a visual aid for the physical
parameters the search moves: aperture diameter, number of apertures, the crystal counts
of rings I and II, and the inner diameters of rings II and III.

Everything drawn comes from the geometry that was simulated, which is the prototype
(Zhang et al., IEEE TMI 45(5), 2026):
  * crystals: build_sc_spect_detector_rings() from generate_mph_scanner_circularfov.py,
    called with that script's own defaults, which are read from its source and checked;
  * tungsten: the same wedge quadrilaterals the generator builds ("plate segments", the
    only collimator geometry the ray tracer reads), so an aperture is the opening between
    two walls, 0.4 mm wide at the 35 mm mid-thickness radius, through the full 2.5 mm.

The main view is to scale (S px per mm, with a 100 mm bar). Each inset is the circle
drawn on the main view, magnified M times relative to the main view.

Colour: UB blue and coral mark what the search changes (coral for the two searched inner
diameters), grey what is fixed. Text is 12 pt, 11.5 pt inside the inset circles; at the
376 dpi used here (1880 px over the 5.0 in slide box) cap heights are 45 and 43 px.

Usage:
  python make_scanner_zoom_figure.py [--out PATH] [--layers]

--layers leaves PATH alone and writes the slide-8 click-build layers beside it:
<stem>_l1_scanner.png (the drawing, ring numerals, Ø key, scale bar) and <stem>_l2_zooms.png
(zoom windows and connectors, both insets, captions, searched/fixed key). Each is RGBA,
transparent where it draws nothing. The full figure is rendered and checked in the same
run, and the layers are written only if, laid over white in that order, they equal it to
within 2 levels.
"""
import argparse
import ast
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
GEOM_DIR = HERE.parent / "geometry"
GEN_PATH = GEOM_DIR / "generate_mph_scanner_circularfov.py"
DEFAULT_OUT = HERE.parent.parent / "deck_figures" / "scanner_zoom_r5.png"

# ---------------------------------------------------------------- configuration simulated
# These must equal the generator's defaults; check_generator() fails otherwise.
D_INNER = [260.0, 390.0, 520.0, 650.0]     # rings I..IV inner diameter (mm)
N_CRYST = [480, 720, 960, 1200]            # crystals per ring
W_T, H_R, GAP_IN = 0.84, 6.0, 0.84         # crystal tangential, radial; gap inside a pair (mm)
COL_D_IN, COL_T = 67.5, 2.5                # tungsten ring inner diameter, thickness (mm)
AP_D, N_AP = 0.4, 180                      # aperture diameter (mm), number of apertures
FOV_D = 10.0                               # field of view diameter (mm)
NUMERALS = ["I", "II", "III", "IV"]

# ---------------------------------------------------------------- page and style
DPI = 376                                  # 1880 px fill the 5.0 in wide slide box
WPX, HPX = 1880, 2000
UBB, CORAL, INK, GREY = "#005BBB", "#E56D54", "#1A1A1A", "#8E8E8E"
FOVC, TUNG, LEAD = "#E3E6EA", "#6E6E73", "#4D4D4D"
BLUE_T, GREY_T = "#7BA2D5", "#AAB0B8"      # crystal fills: searched count, fixed count
FS, FSI = 12.0, 11.5                       # text size outside / inside the inset circles
CAP_PX = {12.0: 45, 11.5: 43}              # measured Helvetica Neue cap heights at 376 dpi

# ---------------------------------------------------------------- layout (pixels, y down)
S = 1.72                                   # main view scale, px per mm
CX, CY = 1112, 597                         # scanner centre
CORAL_DEG = {1: 24.0, 2: 0.0}              # direction of the ring II and ring III diameters
CALL_Y = [118, 293, 468, 643]              # where each leader leaves ring IV
# Numerals only, no diameters or crystal counts (Dr. Yao, 26 Oct: "the numbers are not
# needed at this time"); each leader starts just right of its numeral.
CALL_X_NUM, CALL_X_LEAD = 26, 112
KEY_X, KEY_Y = 30, 830
R_IN = 390                                 # inset radius, px
A_CEN, B_CEN = (590.0, 1502.0), (1478.0, 1502.0)
M_A, M_B = 80, 24                          # magnification relative to the main view
SLIDE_TEXT_CORNER = (1700, 720)            # keep x > 1700, y < 720 empty (slide text)
LAYERS = ("scanner", "zooms")              # slide-8 click-build layers, bottom first


def wedge(r0, r1, a, b):
    return [[r0 * math.cos(a), r0 * math.sin(a)], [r1 * math.cos(a), r1 * math.sin(a)],
            [r1 * math.cos(b), r1 * math.sin(b)], [r0 * math.cos(b), r0 * math.sin(b)]]


def check_generator():
    """Fail unless the constants above are the generator's defaults."""
    tree = ast.parse(GEN_PATH.read_text())
    args, assigns = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for kw in node.keywords:
                if kw.arg == "default":
                    args[node.args[0].value] = ast.literal_eval(kw.value)
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            assigns[node.targets[0].id] = node.value
    want = {"--aperture_diam": AP_D, "--n_apertures": N_AP, "--scint_radial_mm": H_R,
            "--ring_thickness": COL_T, "--n_det_ring1": N_CRYST[0],
            "--n_det_ring2": N_CRYST[1], "--d2_inner": D_INNER[1], "--d3_inner": D_INNER[2]}
    lit = {"RING_INNER_DIAM_MM": COL_D_IN, "FOV_DIAMETER_MM": FOV_D,
           "D1_INNER_MM": D_INNER[0], "D4_INNER_MM": D_INNER[3],
           "SCINT_TANGENT_MM": W_T, "INTRA_GAP_MM": GAP_IN}
    bad = [f"{k}: generator {args.get(k)} vs {v}" for k, v in want.items() if args.get(k) != v]
    bad += [f"{k}: generator {ast.literal_eval(assigns[k])} vs {v}" for k, v in lit.items()
            if ast.literal_eval(assigns[k]) != v]
    rings34 = [eval(compile(ast.Expression(e), "<gen>", "eval"), {})   # literal products
               for e in assigns["DETS_PER_RING"].elts[2:]]
    if rings34 != N_CRYST[2:]:
        bad.append(f"rings III, IV crystals: generator {rings34} vs {N_CRYST[2:]}")
    if bad:
        raise SystemExit("ERROR: figure constants differ from the generator:\n  " + "\n  ".join(bad))


def load_geometry():
    sys.dont_write_bytecode = True           # leave geometry/__pycache__ untouched
    sys.path.insert(0, str(GEOM_DIR))
    spec = importlib.util.spec_from_file_location("scspect_generator", GEN_PATH)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    units, stats = gen.build_sc_spect_detector_rings(D_INNER, N_CRYST, W_T, GAP_IN, H_R)
    units = units.double().numpy()
    off = np.cumsum([0] + N_CRYST)
    rings = [units[off[k]:off[k + 1]] for k in range(4)]
    # tungsten walls, exactly as section 2b of the generator builds them
    r_in = COL_D_IN / 2; r_out = r_in + COL_T; r_c = r_in + COL_T / 2
    d_cell = 2 * math.pi / N_AP; d_open = AP_D / r_c
    walls = []
    for i in range(N_AP):
        c = i * d_cell
        walls.append(wedge(r_in, r_out, c - 0.5 * d_cell, c - 0.5 * d_open))
        walls.append(wedge(r_in, r_out, c + 0.5 * d_open, c + 0.5 * d_cell))
    return rings, stats, np.array(walls), (r_in, r_out, r_c, d_cell, d_open)


def ang_pt(r, deg):
    t = math.radians(deg)
    return np.array([r * math.cos(t), r * math.sin(t)])


class Figure:
    def __init__(self):
        plt.rcParams["font.family"] = "Helvetica Neue"
        self.fig = plt.figure(figsize=(WPX / DPI, HPX / DPI), dpi=DPI)
        self.ax = ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, WPX); ax.set_ylim(HPX, 0); ax.axis("off")
        self.texts = []                          # (artist, inset or None)
        self.renderer = self.fig.canvas.get_renderer()
        self.layer_of, self.cur = {}, LAYERS[0]  # artist -> click-build layer

    def layer(self, name):
        """Artists added from here on belong to layer `name`; None closes the last one."""
        for a in self.fig.axes:
            for art in a.get_children():
                self.layer_of.setdefault(art, self.cur)
        self.cur = name

    # main-view coordinates: mm -> px
    @staticmethod
    def P(xy):
        xy = np.asarray(xy, float)
        return np.stack([CX + S * xy[..., 0], CY - S * xy[..., 1]], -1)

    def text(self, a, x, y, s, inset=None, **kw):
        kw.setdefault("fontsize", FS)
        t = a.text(x, y, s, **kw)
        self.texts.append((t, inset))
        return t

    def width_px(self, s, fontsize, weight):
        t = self.ax.text(0, 0, s, fontsize=fontsize, fontweight=weight)
        w = t.get_window_extent(self.renderer).width
        t.remove()
        return w

    def runs(self, a, x, y, parts, unit=1.0, inset=None, ha="center", fontsize=FS, **kw):
        """One line from [(text, colour, weight)], positioned like a single text.
        unit: data units per pixel along x for axes `a`."""
        widths = [self.width_px(s, fontsize, w) for s, _, w in parts]
        x0 = x - (sum(widths) * unit / 2 if ha == "center" else 0.0)
        for (s, col, w), wd in zip(parts, widths):
            self.text(a, x0, y, s, inset=inset, color=col, fontweight=w, ha="left",
                      fontsize=fontsize, **kw)
            x0 += wd * unit


def head(a, tip, d, color, L, Wd, z):
    d = np.asarray(d, float); d = d / np.linalg.norm(d); n = np.array([-d[1], d[0]])
    tip = np.asarray(tip, float); b = tip - d * L
    a.add_patch(Polygon([tip, b + n * Wd, b - n * Wd], closed=True, fc=color, ec="none", zorder=z))


def dim(a, p0, p1, color, lw=1.3, z=8, L=15.0, Wd=5.0, unit=1.0):
    """Dimension line with arrowheads touching p0 and p1 (L, Wd in px; unit = data per px)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    L *= unit; Wd *= unit
    a.add_line(Line2D(*np.array([p0 + d * L * .5, p1 - d * L * .5]).T, color=color, lw=lw,
                      zorder=z, solid_capstyle="butt"))
    head(a, p0, -d, color, L, Wd, z); head(a, p1, d, color, L, Wd, z)


def seg(a, p0, p1, color, lw=0.9, z=8, **kw):
    a.add_line(Line2D(*np.array([p0, p1], float).T, color=color, lw=lw, zorder=z,
                      solid_capstyle="butt", **kw))


def seg_dist(p, a, b):
    """Distance from point p to segment ab."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
    return float(np.linalg.norm(p - a - t * (b - a)))


class Inset:
    """Circle of radius R px at `cen` showing the main-view disc of radius R/M px around
    `src_mm`, magnified M times. Its axes use main-view mm, so geometry is reused as is."""

    def __init__(self, F, cen, R, M, src_mm):
        self.F, self.cen, self.R, self.M = F, np.asarray(cen, float), R, M
        self.k = M * S                                  # px per mm inside the inset
        self.win = np.asarray(src_mm, float)
        half = R / self.k
        self.ax = a = F.fig.add_axes([(cen[0] - R) / WPX, 1 - (cen[1] + R) / HPX,
                                      2 * R / WPX, 2 * R / HPX])
        a.set_xlim(self.win[0] - half, self.win[0] + half)
        a.set_ylim(self.win[1] - half, self.win[1] + half)
        a.set_aspect("equal"); a.axis("off")
        self.clip = Circle(tuple(self.win), half, fc="white", ec="none", zorder=0)
        a.add_patch(self.clip)
        sp = F.P(self.win); rs = R / M                  # the magnified disc, to scale
        F.ax.add_patch(Circle(sp, rs, fc="none", ec=INK, lw=0.9, zorder=9))
        # One connector, centre to centre. Both windows sit below the scanner centre, so
        # two-line cones crossed each other and passed a few px from the other window.
        u = (self.cen - sp) / np.linalg.norm(self.cen - sp)
        self.src, self.src_r = sp, rs
        self.link = (sp + u * rs, self.cen - u * R)
        seg(F.ax, *self.link, GREY, lw=0.85, z=7, ls=(0, (4, 3)))

    def mm(self, x_px, y_px):                           # inset px offsets (y up) -> mm
        return self.win + np.array([x_px, y_px]) / self.k

    def loc(self, p_mm):                                # mm -> inset px offsets
        return (np.asarray(p_mm, float) - self.win) * self.k

    def line(self, x, y, parts, va):
        p = self.mm(x, y)
        self.F.runs(self.ax, p[0], p[1], parts, unit=1 / self.k, inset=self, fontsize=FSI,
                    va=va, zorder=20)

    def finish(self):
        for art in self.ax.get_children():
            if art is not self.clip and not hasattr(art, "get_text") and hasattr(art, "set_clip_path"):
                art.set_clip_path(self.clip)
        self.ax.add_patch(Circle(tuple(self.win), self.R / self.k, fc="none", ec=INK, lw=1.2,
                                 zorder=50, clip_on=False))


def build(out_path, layers=False):
    check_generator()
    rings, stats, walls, (r_in_c, r_out_c, r_c_ap, d_cell, d_open) = load_geometry()
    F = Figure(); ax = F.ax; P = F.P

    # ------------------------------------------------ numbers, computed from the geometry
    cell = N_CRYST[0] // 2; d_cell1 = 2 * math.pi / cell; rc1 = D_INNER[0] / 2 + H_R / 2
    i0, i1 = 180, 181                       # ring I cells at 270 and 271.5 degrees
    plus_t = rings[0][2 * i0 + 1]; minus_t = rings[0][2 * i1]
    g_mid = ((plus_t[0] + plus_t[3]) / 2, (minus_t[1] + minus_t[2]) / 2)   # facing edges, mid-depth
    gap_mid = float(np.linalg.norm(g_mid[1] - g_mid[0]))
    gap_inner = float(np.linalg.norm(plus_t[3] - minus_t[2]))
    gap_outer = float(np.linalg.norm(plus_t[0] - minus_t[1]))
    gap_formula = math.pi * (D_INNER[0] + H_R) / cell - 2 * W_T - GAP_IN
    gap_code = stats["rings"][0]["clearance_mm"]
    assert abs(gap_formula - gap_code) < 1e-9 and abs(gap_mid - gap_formula) < 1e-3
    c0, c1 = rings[0][2 * i0], rings[0][2 * i0 + 1]
    gap_pair = float(np.linalg.norm((c0[0] + c0[3]) / 2 - (c1[1] + c1[2]) / 2))
    cw = float(np.linalg.norm(c1[0] - c1[1])); ch = float(np.linalg.norm(c1[0] - c1[3]))
    opening = 2 * r_c_ap * math.sin(d_open / 2)
    pitch_arc = 2 * math.pi * r_c_ap / N_AP
    pitch_chord = 2 * r_c_ap * math.sin(math.pi / N_AP)
    labels = {"gap_mid": f"{gap_mid:.2f} mm", "gap_pair": f"{gap_pair:.2f} mm",
              "crystal": f"{cw:.2f} × {ch:.0f} mm", "opening": f"{opening:.2f} mm",
              "pitch": f"{pitch_arc:.2f} mm"}
    assert labels == {"gap_mid": "0.96 mm", "gap_pair": "0.84 mm", "crystal": "0.84 × 6 mm",
                      "opening": "0.40 mm", "pitch": "1.22 mm"}, labels
    assert f"{pitch_chord:.2f}" == "1.22"

    # ------------------------------------------------ main view, to scale
    for k in range(4):
        ax.add_collection(PolyCollection(P(rings[k]), facecolor=BLUE_T if k < 2 else GREY_T,
                                         edgecolor="none", zorder=2))
    ax.add_collection(PolyCollection(P(walls), facecolor=TUNG, edgecolor=TUNG, lw=0.25, zorder=2))

    # searched inner diameters, end to end through the centre, arrowheads on the inner surface
    for k, deg in CORAL_DEG.items():
        r = D_INNER[k] / 2
        dim(ax, P(ang_pt(r, deg + 180)), P(ang_pt(r, deg)), CORAL, lw=1.6, z=5, L=17, Wd=6)

    # ring callouts: the numeral only; leaders radial inside the rings
    r_exit = D_INNER[3] / 2 + H_R + 2.5
    for k, y_exit in enumerate(CALL_Y):
        deg = 180 - math.degrees(math.asin((CY - y_exit) / (S * r_exit)))
        p_ring = P(ang_pt(D_INNER[k] / 2 + H_R / 2, deg)); p_exit = P(ang_pt(r_exit, deg))
        ax.add_line(Line2D([p_ring[0], p_exit[0], CALL_X_LEAD], [p_ring[1], p_exit[1], p_exit[1]],
                           color=LEAD, lw=0.8, zorder=7))
        ax.add_patch(Circle(p_ring, 6.0, fc=INK, ec="none", zorder=8))
        y = p_exit[1]
        F.text(ax, CALL_X_NUM, y, NUMERALS[k], fontweight="bold", color=INK, va="center")

    # field of view, over the two diameters so it shows where they cross; the leader
    # runs from its edge, so the label reads as the FOV's and not the collimator's
    r_fov = S * FOV_D / 2
    ax.add_patch(Circle((CX, CY), r_fov, fc=FOVC, ec=GREY, lw=1.0, zorder=6))
    seg(ax, (CX, CY - r_fov), (CX, CY - 76), GREY, lw=0.8, z=7)
    F.text(ax, CX, CY - 80, f"Ø{FOV_D:.0f} mm", color=GREY, ha="center", va="bottom")
    F.text(ax, CX, CY - 142, "FOV", color=GREY, ha="center", va="bottom")

    # key and scale bar; the searched / fixed key shows with the zooms (slide 8, click 3)
    F.layer("zooms")
    ax.add_patch(Rectangle((KEY_X, KEY_Y - 13), 26, 26, fc=UBB, ec="none"))
    ax.add_patch(Rectangle((KEY_X + 34, KEY_Y - 13), 26, 26, fc=CORAL, ec="none"))
    F.text(ax, KEY_X + 78, KEY_Y, "searched", color=INK, va="center")
    ax.add_patch(Rectangle((KEY_X, KEY_Y + 57), 26, 26, fc=GREY, ec="none"))
    F.text(ax, KEY_X + 78, KEY_Y + 70, "fixed", color=INK, va="center")
    F.layer("scanner")
    # Kept with the ring values gone: the FOV, the collimator caption and inset (b) still use Ø.
    F.text(ax, KEY_X, KEY_Y + 140, "Ø: ring inner diameter", color=INK, va="center")
    yb = KEY_Y + 225; bar = 100 * S
    seg(ax, (KEY_X, yb), (KEY_X + bar, yb), INK, lw=2.0)
    for xx in (KEY_X, KEY_X + bar):
        seg(ax, (xx, yb - 12), (xx, yb + 12), INK, lw=1.2)
    F.text(ax, KEY_X + bar + 18, yb, "100 mm", color=INK, va="center")

    # ------------------------------------------------ (a) collimator, apertures at 266..272 deg
    F.layer("zooms")                         # windows, connectors, insets and captions
    A = Inset(F, A_CEN, R_IN, M_A, ang_pt(r_c_ap, 269.0))
    A.ax.add_collection(PolyCollection(walls, facecolor=TUNG, edgecolor=TUNG, lw=0.25, zorder=2))
    hb = COL_T / 2 * A.k
    c = math.radians(266.0)                  # aperture diameter, at r = 35 mm
    pL = r_c_ap * np.array([math.cos(c - d_open / 2), math.sin(c - d_open / 2)])
    pR = r_c_ap * np.array([math.cos(c + d_open / 2), math.sin(c + d_open / 2)])
    dim(A.ax, pL, pR, UBB, unit=1 / A.k, L=14)
    mid = (pL + pR) / 2
    seg(A.ax, mid + np.array([0, 9 / A.k]), A.mm(A.loc(mid)[0], hb + 6), UBB)
    A.line(-20, hb + 10, [("aperture diameter", UBB, "normal")], va="bottom")
    A.line(-20, hb + 72, [(labels["opening"], UBB, "bold")], va="bottom")
    q0, q1 = ang_pt(r_c_ap, 268.0), ang_pt(r_c_ap, 270.0)        # neighbouring centres
    nrm = (q0 + q1) / np.linalg.norm(q0 + q1)                     # outward
    drop = (r_out_c - r_c_ap) + 20 / A.k
    for q in (q0, q1):
        seg(A.ax, q, q + nrm * (drop + 12 / A.k), UBB)
    dim(A.ax, q0 + nrm * drop, q1 + nrm * drop, UBB, unit=1 / A.k, L=14)
    A.line(0, -(hb + 30), [(labels["pitch"], UBB, "bold"), (" apart", UBB, "normal")], va="top")
    A.line(0, -(hb + 92), [(f"{N_AP} apertures", UBB, "normal")], va="top")
    A.finish()

    # ------------------------------------------------ (b) ring I, centred on a gap between pairs
    B = Inset(F, B_CEN, R_IN, M_B, ang_pt(rc1, 270.0 + math.degrees(d_cell1) / 2))
    B.ax.add_collection(PolyCollection(rings[0], facecolor=BLUE_T, edgecolor="none", zorder=2))
    hb = H_R / 2 * B.k
    dim(B.ax, g_mid[0], g_mid[1], UBB, unit=1 / B.k, L=13)      # between pairs, mid-depth
    gm = (g_mid[0] + g_mid[1]) / 2
    seg(B.ax, gm - np.array([0, 8 / B.k]), B.mm(B.loc(gm)[0], -(hb + 6)), UBB)
    ip = 179                                  # outline one pair, and dimension the gap inside it
    a_, b_ = rings[0][2 * ip], rings[0][2 * ip + 1]
    th = ip * d_cell1
    t_hat = np.array([-math.sin(th), math.cos(th)]); r_hat = np.array([math.cos(th), math.sin(th)])
    ctr = r_hat * rc1; e = 0.10
    ht, hr = (W_T + GAP_IN) / 2 + W_T / 2 + e, H_R / 2 + e
    box = [ctr + sx * ht * t_hat + sy * hr * r_hat for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    B.ax.add_patch(Polygon(box, closed=True, fc="none", ec=GREY, lw=0.9, ls=(0, (3, 2.5)), zorder=6))
    iL = (a_[0] + a_[3]) / 2; iR = (b_[1] + b_[2]) / 2
    dim(B.ax, iL, iR, GREY, unit=1 / B.k, L=13)
    im = (iL + iR) / 2
    seg(B.ax, im + np.array([0, 8 / B.k]), B.mm(B.loc(im)[0], hb + 6), GREY)
    B.line(0, hb + 10, [(f"{labels['gap_pair']} gap in a pair", GREY, "normal")], va="bottom")
    B.line(0, hb + 72, [(f"crystal {labels['crystal']}", GREY, "normal")], va="bottom")
    B.line(0, -(hb + 10), [(labels["gap_mid"], UBB, "bold"), (" between pairs", UBB, "normal")],
           va="top")
    B.line(0, -(hb + 72), [("set by crystal count", UBB, "normal")], va="top")
    B.line(0, -(hb + 134), [("and ring Ø", UBB, "normal")], va="top")
    B.finish()

    # captions with the magnification
    F.runs(ax, A_CEN[0], A_CEN[1] + R_IN + 12,
           [("(a) collimator ring, ", INK, "normal"), (f"Ø{COL_D_IN} mm", GREY, "normal"),
            (f", {M_A}×", INK, "normal")], va="top")
    F.runs(ax, B_CEN[0], B_CEN[1] + R_IN + 12,
           [("(b) detector ring I, ", INK, "normal"), (f"{M_B}×", INK, "normal")], va="top")
    F.layer(None)

    full = out_path
    if layers:                                   # leave out_path alone; check a scratch copy
        tmp = tempfile.TemporaryDirectory(); full = Path(tmp.name) / "full.png"
    F.fig.savefig(full, dpi=DPI, facecolor="white")
    from PIL import Image
    with Image.open(full) as im:                 # opaque RGB, white background
        rgb = im.convert("RGB")
    rgb.save(full, dpi=(DPI, DPI))
    problems = check_layout(F, full)
    # each connector keeps clear of the other inset's window and of the other connector
    clear = min(seg_dist(B.src, *A.link) - B.src_r, seg_dist(A.src, *B.link) - A.src_r,
                min(seg_dist(q, *B.link) for q in np.linspace(*A.link, 400)))
    if clear < 25:
        problems.append(f"zoom connectors come within {clear:.1f} px of each other or a window")
    facts = dict(gap_mid=gap_mid, gap_inner=gap_inner, gap_outer=gap_outer, gap_formula=gap_formula,
                 gap_pair=gap_pair, cw=cw, ch=ch, opening=opening, pitch_arc=pitch_arc,
                 pitch_chord=pitch_chord, r_c_ap=r_c_ap, clear=clear)
    if layers:
        facts["layers"], facts["stack"] = write_layers(F, out_path, full, (A.clip, B.clip))
        tmp.cleanup()
    return problems, facts


def write_layers(F, out_path, full_png, discs):
    """Write each layer beside out_path as straight-alpha RGBA, alpha 0 where it draws
    nothing. Drawn on a transparent canvas, the stack came out up to 4 levels light where
    many antialiased edges overlap (the main-view tungsten ring), so each layer is drawn on
    white and on black: alpha = 1 - (white - black) / 255, and the colour is the one that,
    laid over white, gives the layer drawn on white. Nothing is written unless the layers
    laid over white in order equal the full figure to within 2 levels. Returns the paths
    written and (max, mean |difference| in levels, pixels off by more than 8)."""
    import io
    from PIL import Image
    for d in discs:                  # the white under each inset covers no layer-1 ink
        d.set_facecolor("none")      # (the stack check below would show it); clip path only

    def draw(name, bg):
        for art, lay in F.layer_of.items():
            art.set_visible(lay == name)
        buf = io.BytesIO()
        F.fig.savefig(buf, format="png", dpi=DPI, facecolor=bg)
        with Image.open(buf) as im:
            return np.asarray(im.convert("RGB"), float)

    layers = []
    for name in LAYERS:
        w, b = draw(name, "white"), draw(name, "black")
        a = np.clip(np.rint(255 - (w - b).mean(axis=2)), 0, 255)[..., None]
        c = np.clip(np.rint(255 - (255 - w) * 255 / np.maximum(a, 1)), 0, 255) * (a > 0)
        layers.append(np.dstack([c, a]).astype(np.uint8))
    with Image.open(full_png) as im:
        want = np.asarray(im.convert("RGB"), float)
    got = np.full_like(want, 255.0)
    for lay in layers:
        a = lay[..., 3:] / 255
        got = lay[..., :3] * a + got * (1 - a)
    d = np.abs(np.rint(got) - want)
    stack = (d.max(), d.mean(), int((d.max(axis=2) > 8).sum()))
    if stack[0] > 2 or stack[2]:
        return [], stack
    out = Path(out_path); paths = []
    for k, (name, lay) in enumerate(zip(LAYERS, layers), 1):
        p = out.with_name(f"{out.stem}_l{k}_{name}{out.suffix}")
        Image.fromarray(lay).save(p, dpi=(DPI, DPI)); paths.append(p)
    return paths, stack


def check_layout(F, out_path):
    """Every text inside the page, inset text inside its circle, no text on drawn ink,
    nothing in the corner the slide text uses, exact pixel size."""
    from PIL import Image
    fig = F.fig
    fig.canvas.draw()
    boxes = [(t, ins, t.get_window_extent(F.renderer)) for t, ins in F.texts]
    for t, _ in F.texts:
        t.set_visible(False)
    fig.canvas.draw()
    bare = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    for t, _ in F.texts:
        t.set_visible(True)
    problems = []
    for t, ins, bb in boxes:
        name = repr(t.get_text())
        if t.get_fontsize() not in CAP_PX:
            problems.append(f"font size {t.get_fontsize()} for {name}")
        x0, x1 = int(bb.x0) - 3, int(bb.x1) + 3
        y0, y1 = HPX - int(bb.y1) + 6, HPX - int(bb.y0) - 4
        if x0 < 8 or y0 < 8 or x1 > WPX - 8 or y1 > HPX - 8:
            problems.append(f"near the page edge: {name}")
        if ins is not None:
            far = max(math.hypot(px - ins.cen[0], py - ins.cen[1])
                      for px in (bb.x0, bb.x1) for py in (HPX - bb.y0, HPX - bb.y1))
            if far > ins.R - 6:
                problems.append(f"outside its circle by {far - ins.R + 6:.0f} px: {name}")
        ink = int((bare[max(y0, 0):y1, max(x0, 0):x1].min(axis=2) < 235).sum())
        if ink:
            problems.append(f"on top of {ink} px of drawing: {name}")
    cx, cy = SLIDE_TEXT_CORNER
    with Image.open(out_path) as im:
        if im.size != (WPX, HPX):
            problems.append(f"size {im.size}, want {(WPX, HPX)}")
        page = np.asarray(im.convert("RGB")).astype(int)
    if (page[:cy, cx:].min(axis=2) < 235).any():
        problems.append("ink in the top-right corner the slide text overlaps")
    return problems


def main():
    ap = argparse.ArgumentParser(description="Build the slide-8 scanner figure")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--layers", action="store_true",
                    help="write the click-build layers beside --out instead of --out itself")
    args = ap.parse_args(); out = args.out
    problems, f = build(out, args.layers)
    if args.layers:
        mx, mean, n8 = f["stack"]
        print(f"layers laid over white vs the full figure: max {mx:.0f}, mean {mean:.4f} levels, "
              f"{n8} px off by more than 8")
        if f["layers"]:
            print("saved " + ", ".join(str(p) for p in f["layers"]))
        else:
            problems.append("the layers do not stack to the full figure; nothing written")
    else:
        print(f"saved {out}")
    print(f"ring I gap between pairs: {f['gap_mid']:.4f} mm at mid-depth "
          f"(formula {f['gap_formula']:.4f}; inner face {f['gap_inner']:.4f}, outer face {f['gap_outer']:.4f})")
    print(f"gap inside a pair {f['gap_pair']:.4f} mm; crystal {f['cw']:.4f} x {f['ch']:.4f} mm")
    print(f"aperture opening at r = {f['r_c_ap']} mm: {f['opening']:.5f} mm; "
          f"pitch {f['pitch_arc']:.4f} mm (chord {f['pitch_chord']:.4f})")
    print(f"main view {S} px/mm, 100 mm bar = {100 * S:.0f} px; insets {M_A}x and {M_B}x")
    print(f"zoom connectors: {f['clear']:.1f} px clear of each other and the windows")
    if problems:
        print("LAYOUT PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("layout checks passed")


if __name__ == "__main__":
    main()
