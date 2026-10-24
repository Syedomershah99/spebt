# Where each deck figure comes from

Data plots in the MIC deck are generated, not drawn by hand. Matplotlib
scripts live in `omer/spebt/optimization/` and are run from that directory;
manim scenes live in `MIC/deck/visuals/`. Everything below is what R5
(`MIC/deck/build_deck_v4.py`, which writes R6) places on a slide.

## Data figures (matplotlib)

| Figure | Slide | Script | Data |
|---|---|---|---|
| `asci_cnr_tradeoff.png` | 11 | `make_tradeoff_figure.py`<br>Click-build layers: `asci_cnr_tradeoff_l1_frame.png` (axes, ticks, grid, x label, dashed marker, ring III note, footnote; from entry), `asci_cnr_tradeoff_l2_asci.png` (ASCI curve, its label, top title; bullet 2), `asci_cnr_tradeoff_l3_cnr.png` (CNR curve and bars, its label, bottom title; bullet 3), written by `make_tradeoff_figure.py --out ../../deck_figures/asci_cnr_tradeoff.png --layers`, which never writes the still and writes nothing unless the three laid over white in order give it to within 2 levels | ring2_000..007 sweep, 5-seed CNR; outer four points also moved ring III |
| `mpxi_optimum.png` | 10 | `make_mpxi_optimum_figure.py --data ../archive_backup/results_summary_mobo.csv --h2h ../archive_backup/h2h --band 0.20,0.45 --edges 1.5,1.8,2.1,2.5`<br>Click-build layers: `mpxi_optimum_l1_points.png` (axes, ticks, grid, both axis labels, the designs, the dot legend entry; from entry), `mpxi_optimum_l2_trend.png` (binned mean and ±1 SD bars, their legend entry, the shaded peak bin and its "best here" note, the two notes along the bottom, the title; bullet 4), written by the same command with `--out ../../deck_figures/mpxi_optimum.png --layers`, which never writes the still and exits with an error if the two laid over white in order differ from it by more than 8 levels anywhere (they give it to within 1) | aperture band 0.20 to 0.45 mm, all eleven searches pooled (main archive first, then the ten h2h files; mobo_NNNN names qualified by search, shared lhs and seed designs scored once): 796 designs, 14 of them hand-placed (ring2_*, seed6d_*); fixed bins under 1.5, 1.5 to 1.8, 1.8 to 2.1, 2.1 to 2.5, over 2.5 give 3.91 / 4.13 / 4.18 / 3.50 / 2.89. R7 plotted the main archive, 285 designs at all apertures |
| `reference_comparison.png` | 16 | `make_reference_figure.py` | 5-seed per-rod CNR, values recorded in the script |
| `gain_map.png` | 17 | `make_gain_map_figure.py` | same per-rod gains |
| `search_convergence.png` | 18 | `make_convergence_figure.py` | main archive plus the ten h2h arms; needs the `results/` and `results_h2h_*/` layout (see the script) |
| `search_process.png` | 19 | `make_search_figure.py` | `archive_backup/h2h/results_h2h_5obj_r4.csv` |
| `mobo_acquisition.png` | 20 | `make_mobo_acquisition_figure.py` | `archive_backup/h2h/results_h2h_2obj_r5.csv`, design 34 (proposal mobo_0012) with the search's model refit to the 33 designs before it; the script reruns `mobo_agent.get_next_candidate` and refuses to draw unless it lands within 0.02 |
| `scanner_zoom_r5.png` | 8 | `make_scanner_zoom_figure.py`<br>Click-build layers: `scanner_zoom_r5_l1_scanner.png` (the to-scale rings and collimator ring, the FOV and its label, the coral ring II and III diameters, numerals I to IV with their Ø and crystal labels, the Ø key, the scale bar; from entry), `scanner_zoom_r5_l2_zooms.png` (the two zoom windows on the drawing, the dashed connectors, both insets with their callouts, captions (a) and (b), the searched/fixed key; click 3), written beside the still by `make_scanner_zoom_figure.py --layers`, which never writes the still and writes nothing unless the two laid over white in order give it to within 2 levels (they give it to within 1) | the prototype geometry as simulated: crystals from `geometry/generate_mph_scanner_circularfov.py`'s own `build_sc_spect_detector_rings` and wall quads, constants checked against its defaults; to scale, insets 80x and 24x |
| `recon_comparison_2panel.png` | (source of the two panels) | `make_recon_figure.py` | needs recon `.npz` on CCR; one noise realization, common display window |
| `recon_reference_panel.png`, `recon_optimized_panel.png` | 12 | cropped from `recon_comparison_2panel.png` without its matplotlib titles | |
| `eq_xnext_ry.png`, `eq_xnext_ry_green.png` | 7, 20 | `make_equation_figures.py` (mathtext, STIX): Dr. Yao's R7-rY equation as he laid it out, x in the design space under argmax, the double-struck E upright (`\mathrm{\mathbb{E}}`; bare `\mathbb{E}` is STIX's slanted form); the builder reads the size from the file | |
| `eq_spearman.png` | 9 | `make_equation_figures.py`: the correlation coefficient of the ranks, as pandas computes it | |
| `eq_gp_mean.png`, `eq_gp_var.png`, `eq_gp_kernel.png`, `eq_ehvi.png` | 21 | `make_equation_figures.py`: the Gaussian process and the expected hypervolume improvement, at the slide's 14 pt | |
| `eq_xnext.png`, `eq_xnext_green.png` | (not in R8) | `make_equation_figures.py`; R7's slides 7 and 20 | |
| `hotrod_phantom.png` | 6 | rendered from the phantom tensor | rod radii 0.100 to 0.225 mm |

## Diagrams and animations (manim)

Render stills with `-s --format=png` and videos with `--format=mp4`; every
scene asserts that nothing leaves the frame. Videos are H.264 yuv420p and end
on the same frame as the still, which the deck uses as the poster frame.

| Figure | Slide | Scene |
|---|---|---|
| `sc_intro.png`, `sc_intro_c_1..4.mp4` (+ `.png` first frames) | 3 | `sc_intro_manim.py SCIntro` (still), `SCIntroAnim` rendered with `SC_CUTS=cuts.json` (1960x700, 30 fps), then `cut_clips.py <video> cuts.json sc_intro_c`: one clip per click, each starting on the frame the one before ends on, the zoom onto the front detectors in clips 2 and 3; passage computed exactly by `sc_geometry.py`; the still covers the clips on the slide and in the PDF |
| `sc_challenge.png`, `sc_challenge_c_1..7.mp4` (+ `.png` first frames) | 4 | `sc_challenge_manim.py SCChallenge` (still `-s -r 2400,1800`; video `-r 1600,1200` with `SC_CUTS=cuts.json`), then `cut_clips.py <video> cuts.json sc_challenge_c`: the build plays as the slide opens, then one clip per click (gap, ring, metal ring, aperture diameter, number of apertures, all marks) |
| `loop_r4.png`, `loop_r4.mp4` | 5 | `loop_r4_manim.py LoopStill`, `LoopAnim` (five cards in a cycle, per Dr. Yao's 19 Oct review: geometry, PPDF and image on top; MO and BO, the contributions, green in the bottom row; one detector's PPDF computed through the collimator as drawn; arrows from the PPDF onto MO's four PPDF bars and from the image onto CNR; bars in slide 6's order) |
| `ppdf_cut.png` | 6 | `ppdf_cut_manim.py PPDFCut` (strips, not beams; dashed lines to the FOV and strips only inside it; ASCI as strips through one pixel with its formula; the FWHM profile beside its cut)<br>Click-build layers, one per bullet: `ppdf_cut_l1_ppdf.png`, `ppdf_cut_l2_mpxi.png`, `ppdf_cut_l3_fwhm.png`, `ppdf_cut_l4_asci.png`, each rendered with `PPDF_LAYER=ppdf /Users/Omer/Desktop/RA/omer/.venv/bin/python -m manim render ppdf_cut_manim.py PPDFCut -s --format=png -t -o ppdf_cut_l1_ppdf` (then `mpxi` l2, `fwhm` l3, `asci` l4) and copied from `media/images/ppdf_cut_manim/`; `-t` is required, the full scene is built and checked every time, and the four laid over white in that order give `ppdf_cut.png` to within 1 level |
| `mobo_search.png`, `mobo_search.mp4` | 7 | `mobo_search_manim.py MOBOSearch`, real data from `results_h2h_2obj_r5.csv`; adds the area the best trade-off covers (HV) against design number |

## Retired

`scanner_case_study.png` (R4 slide 8, not to scale; replaced by the to-scale
`scanner_zoom_r5.png` Dr. Yao asked for), `collimation_compare.png` (R3 slide 3: two of its strips started in gaps
between crystals), `loop_mini.png`, `ppdf_beams.png`, `scanner_only.png` and
the August figures from the earlier 4D study are no longer used by the deck.
