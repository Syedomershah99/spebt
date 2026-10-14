# Where each deck figure comes from

Data plots in the MIC deck are generated, not drawn by hand. Scripts live in
`omer/spebt/optimization/` and are run from that directory.

| Figure | Script | Runs on |
|---|---|---|
| `reference_comparison.png` | `make_reference_figure.py --out ...` | laptop |
| `asci_cnr_tradeoff.png` | `make_tradeoff_figure.py` | laptop |
| `mpxi_optimum.png` | `make_mpxi_optimum_figure.py` | laptop |
| `search_convergence.png` | `make_convergence_figure.py` | laptop |
| `search_process.png` | `make_search_figure.py` | laptop |
| `gain_map.png` | `make_gain_map_figure.py` | laptop |
| `recon_comparison.png` | `make_recon_figure.py` | CCR (needs recon .npz) |
| `ppdf_beam.png` | `make_ppdf_figure.py` | CCR (needs raw PPDF .hdf5) |

The two CCR figures read files that exist only on the cluster under
`/vscratch`; both exit with a clear message if run on the laptop. Copy the
resulting .png back into this directory.

`selfcollimation_v3.png` and `metrics_diagram.png` are manim scenes, built
from `MIC/deck/visuals/selfcollimation_v3.py` and `metrics_manim.py`:

    python -m manim render -s -q h --format=png -o <name> <script>.py <Scene>

Both scenes assert that nothing runs outside the frame, because manim crops
silently and still logs "Rendered".

`sai_photo.png` is Fig. 2 of the SAI TMI paper (see
`docs/24-2025-TMI-SCSPECT_SAI_TMI_20250908.pdf`), cropped. Cite it on any
slide that uses it.

`scanner_geometry.png`, `conventional_collimation.png` and
`selfcollimation_concept.png` are older schematics with no generator kept in
the repo. Only `conventional_collimation.png` is still used, on slide 3.

Figures dated August (`headtohead.png`, `hypervolume_convergence.png`,
`metric_scatter_combined.png`, `pareto_expansion.png`) are from the earlier
4D study and are not used by the current deck.
