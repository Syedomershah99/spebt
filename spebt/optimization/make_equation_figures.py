#!/usr/bin/env python3
"""Typeset the slide equations as images (Dr. Yao, 19 Oct rehearsal: "the
equations need to be better looking ... use a better editor").

Calibri has no union sign, no subscript letters and no double-struck E, so the
argmax that slides 7 and 20 show is set here with matplotlib's mathtext in the
STIX fonts (a Times-like math face), in the deck's ink or green, on a
transparent background, at the size it has on the slide.

Usage:  python make_equation_figures.py [--outdir DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "..", "deck_figures")
INK, GREEN = "#1A1A1A", "#1E8449"
DPI = 600

# name: (mathtext, colour, size in pt as on the slide)
EQS = {
    "eq_xnext.png": (r"$x_{\mathrm{next}} = \mathrm{arg\,max}_{x}\, "
                     r"\mathrm{E}[\mathrm{HV}(P\cup\{x\}) - \mathrm{HV}(P)]$", INK, 15),
    "eq_xnext_green.png": (r"$x_{\mathrm{next}} = \mathrm{arg\,max}_{x}\, "
                           r"\mathrm{E}[\mathrm{HV}(P\cup\{x\}) - \mathrm{HV}(P)]$", GREEN, 15),
}


def render(tex, colour, size, path):
    plt.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(0.01, 0.01), dpi=DPI)
    fig.text(0, 0, tex, fontsize=size, color=colour)
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    from PIL import Image
    for name, (tex, colour, size) in EQS.items():
        path = os.path.join(a.outdir, name)
        render(tex, colour, size, path)
        with Image.open(path) as im:
            w, h = im.size
            if im.mode != "RGBA" or im.getextrema()[3][0] != 0:
                raise SystemExit(f"ERROR: {name} is not transparent")
        print(f"{name}: {w} x {h} px = {w / DPI:.2f} x {h / DPI:.2f} in at {size} pt")


if __name__ == "__main__":
    main()
