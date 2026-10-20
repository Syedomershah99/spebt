# Where each deck figure comes from

Data plots in the MIC deck are generated, not drawn by hand. Matplotlib
scripts live in `omer/spebt/optimization/` and are run from that directory;
manim scenes live in `MIC/deck/visuals/`. Everything below is what R5
(`MIC/deck/build_deck_v4.py`, which writes R6) places on a slide.

## Data figures (matplotlib)

| Figure | Slide | Script | Data |
|---|---|---|---|
| `asci_cnr_tradeoff.png` | 11 | `make_tradeoff_figure.py` | ring2_000..007 sweep, 5-seed CNR; outer four points also moved ring III |
| `mpxi_optimum.png` | 12 | `make_mpxi_optimum_figure.py --data ../archive_backup/results_summary_mobo.csv` | main campaign, n = 285 |
| `reference_comparison.png` | 16 | `make_reference_figure.py` | 5-seed per-rod CNR, values recorded in the script |
| `gain_map.png` | 17 | `make_gain_map_figure.py` | same per-rod gains |
| `search_convergence.png` | 18 | `make_convergence_figure.py` | main archive plus the ten h2h arms; needs the `results/` and `results_h2h_*/` layout (see the script) |
| `search_process.png` | 19 | `make_search_figure.py` | `archive_backup/h2h/results_h2h_5obj_r4.csv` |
| `mobo_acquisition.png` | 20 | `make_mobo_acquisition_figure.py` | `archive_backup/h2h/results_h2h_2obj_r5.csv`, design 34 (proposal mobo_0012) with the search's model refit to the 33 designs before it; the script reruns `mobo_agent.get_next_candidate` and refuses to draw unless it lands within 0.02 |
| `scanner_zoom_r5.png` | 8 | `make_scanner_zoom_figure.py` | the prototype geometry as simulated: crystals from `geometry/generate_mph_scanner_circularfov.py`'s own `build_sc_spect_detector_rings` and wall quads, constants checked against its defaults; to scale, insets 80x and 24x |
| `recon_comparison_2panel.png` | (source of the two panels) | `make_recon_figure.py` | needs recon `.npz` on CCR; one noise realization, common display window |
| `recon_reference_panel.png`, `recon_optimized_panel.png` | 9 | cropped from `recon_comparison_2panel.png` without its matplotlib titles | |
| `eq_xnext.png` | 7 | `make_equation_figures.py` (mathtext, STIX; Calibri has no union sign) | |
| `hotrod_phantom.png` | 6 | rendered from the phantom tensor | rod radii 0.100 to 0.225 mm |

## Diagrams and animations (manim)

Render stills with `-s --format=png` and videos with `--format=mp4`; every
scene asserts that nothing leaves the frame. Videos are H.264 yuv420p and end
on the same frame as the still, which the deck uses as the poster frame.

| Figure | Slide | Scene |
|---|---|---|
| `sc_intro.png`, `sc_intro.mp4` | 3 | `sc_intro_manim.py SCIntro`, `SCIntroAnim` (1960x700, 30 fps, 43.4 s, poster `sc_intro.png`; passage computed exactly by `sc_geometry.py`; one wide opening in the metal ring under the layer-I pair, which is outlined in blue; clip timed to the slide 3 text of `talk_script_v4.txt`) |
| `sc_challenge.png`, `sc_challenge.mp4` | 4 | `sc_challenge_manim.py SCChallenge` |
| `loop_r4.png`, `loop_r4.mp4` | 5 | `loop_r4_manim.py LoopStill`, `LoopAnim` (five cards in a cycle, per Dr. Yao's 19 Oct review: geometry, PPDF and image on top; MO and BO, the contributions, green in the bottom row; one detector's PPDF computed through the collimator as drawn; arrows from the PPDF onto MO's four PPDF bars and from the image onto CNR; bars in slide 6's order) |
| `ppdf_cut.png` | 6 | `ppdf_cut_manim.py PPDFCut` (strips, not beams; dashed lines to the FOV and strips only inside it; ASCI as strips through one pixel with its formula; the FWHM profile beside its cut) |
| `mobo_search.png`, `mobo_search.mp4` | 7 | `mobo_search_manim.py MOBOSearch`, real data from `results_h2h_2obj_r5.csv`; adds the area the best trade-off covers (HV) against design number |

## Retired

`scanner_case_study.png` (R4 slide 8, not to scale; replaced by the to-scale
`scanner_zoom_r5.png` Dr. Yao asked for), `collimation_compare.png` (R3 slide 3: two of its strips started in gaps
between crystals), `loop_mini.png`, `ppdf_beams.png`, `scanner_only.png` and
the August figures from the earlier 4D study are no longer used by the deck.
