#!/usr/bin/env python3
"""Typeset the slide equations as images (Dr. Yao, 19 Oct rehearsal: "the
equations need to be better looking ... use a better editor").

Calibri has no union sign, no subscript letters and no double-struck E, so the
argmax that slides 7 and 20 show is set here with matplotlib's mathtext in the
Times New Roman (italic variables, as on the slide), in the deck's ink or green, on a
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
    # Slide 9 (Dr. Yao, 22 Oct rehearsal: "what is the mathematical definition of
    # Spearman rho?"): the correlation coefficient of the ranks r and s, which is
    # what pandas' corr(method="spearman") computes, ties at their average rank.
    "eq_spearman.png": (r"$\rho = \dfrac{\mathrm{cov}(r,\,s)}{\sigma_r\,\sigma_s}$", INK, 16),
    # Dr. Yao's own equation from his R7-rY edit of slide 7 (23 Oct), laid out as he
    # typed it: x and f(x) bold vectors, x in the design space under argmax, f(x)
    # added to the front. STIX (Times-compatible) for the chi and the double-struck
    # E. \mathrm keeps that E upright, as his is (bare \mathbb{E} is STIX's slanted
    # form), and HV is upright as in his 22 Oct rule. The design space is the Greek
    # chi, read "chi" (26 Oct: "you can insert Greek symbols"), not his script X.
    "eq_xnext_ry.png": (r"$\mathbfit{x}_{\mathrm{next}} = \underset{\mathbfit{x}\in\chi}{\mathrm{argmax}}\;"
                        r"\mathrm{\mathbb{E}}[\mathrm{HV}(P\cup\{\mathbfit{f}(\mathbfit{x})\}) - \mathrm{HV}(P)]$", INK, 15, "stix"),
    "eq_xnext_ry_green.png": (r"$\mathbfit{x}_{\mathrm{next}} = \underset{\mathbfit{x}\in\chi}{\mathrm{argmax}}\;"
                              r"\mathrm{\mathbb{E}}[\mathrm{HV}(P\cup\{\mathbfit{f}(\mathbfit{x})\}) - \mathrm{HV}(P)]$", GREEN, 15, "stix"),
    # Slide 6 (26 Oct: "put a computation there, or a formula" for PPDS). Since his
    # R9-rY (28 Oct) the formula is his, PPDS at pixel j: each detector's PPDF over
    # the summed area of its strips, summed over the detectors; the searches' objective
    # averages it over the FOV (compute_metrics._ppds_components). PPDS upright, as
    # HV and cov are; 18 pt, the size of his equation.
    "eq_ppds.png": (r"$\mathrm{PPDS}_{j} = \sum_{i}\,\dfrac{p_{ij}}{A_{i}}$", INK, 18, "stix"),
    # Slide 9 (his R9-rY comment, 28 Oct: "Can you express the weighting
    # mathematically?"): compute_fwhm_and_asci's fwhm_weighted_mean, every strip b's
    # FWHM weighted by its sensitivity s_b, the mean of its PPDF over the strip. Green
    # as the column it sits in; Sigma, not \sum, keeps b beside it.
    "eq_fwhm_w_green.png": (r"$\mathrm{FWHM}_{\mathrm{w}} = \dfrac{\Sigma_{b}\, s_{b}\, \mathrm{FWHM}_{b}}"
                            r"{\Sigma_{b}\, s_{b}}$", GREEN, 14, "stix"),
    # Backup slide 21 (QA, 24 Oct), at its 14 pt body size and in the same style:
    # vectors and matrices (x, k(x), K, y) bold italic, the kernel k(x, x') and the
    # scalars italic, the identity I, the transpose T and the operators upright.
    # Typeset, since Times has no superscript minus, subscript j, union or
    # double-struck E. The expectation is written as on slides 7 and 20.
    "eq_gp_mean.png": (r"$\mu(\mathbfit{x}) = m + \mathbfit{k}(\mathbfit{x})^{\mathrm{T}}"
                       r"(\mathbfit{K} + \eta^{2}\mathbf{I})^{-1}(\mathbfit{y} - m)$", INK, 14, "stix"),
    "eq_gp_var.png": (r"$\sigma^{2}(\mathbfit{x}) = k(\mathbfit{x}, \mathbfit{x}) - \mathbfit{k}(\mathbfit{x})^{\mathrm{T}}"
                      r"(\mathbfit{K} + \eta^{2}\mathbf{I})^{-1}\mathbfit{k}(\mathbfit{x})$", INK, 14, "stix"),
    # The slide's exp(-1/2 sum ...), written exp(-sum .../(2 l^2)) on one line;
    # Sigma, not \sum, keeps j beside it rather than under it.
    "eq_gp_kernel.png": (r"$k(\mathbfit{x}, \mathbfit{x}^{\prime}) = \exp(-\Sigma_{j}\,"
                         r"(x_{j} - x_{j}^{\prime})^{2}/(2\ell_{j}^{2}))$", INK, 14, "stix"),
    "eq_ehvi.png": (r"$\mathrm{EHVI}(\mathbfit{x}) = \mathrm{\mathbb{E}}[\mathrm{HV}(P\cup\{\mathbfit{f}"
                    r"(\mathbfit{x})\}) - \mathrm{HV}(P)]$", INK, 14, "stix"),
}


def render(tex, colour, size, path, fontset="custom"):
    # Times New Roman, italic for variables, to match the slide's x and P runs
    # (Dr. Yao, 21 Oct: one italic Times face for math symbols, text and equations)
    plt.rcParams["mathtext.fontset"] = fontset
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
    fig = plt.figure(figsize=(0.01, 0.01), dpi=DPI)
    fig.text(0, 0, tex, fontsize=size, color=colour)
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=OUTDIR)
    a = ap.parse_args()
    from PIL import Image
    for name, (tex, colour, size, *fs) in EQS.items():
        path = os.path.join(a.outdir, name)
        render(tex, colour, size, path, *fs)
        with Image.open(path) as im:
            w, h = im.size
            if im.mode != "RGBA" or im.getextrema()[3][0] != 0:
                raise SystemExit(f"ERROR: {name} is not transparent")
        print(f"{name}: {w} x {h} px = {w / DPI:.2f} x {h / DPI:.2f} in at {size} pt")


if __name__ == "__main__":
    main()
