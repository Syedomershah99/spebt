# Where each deck figure comes from

Data plots in the MIC deck are generated, not drawn by hand. Scripts live in
`omer/spebt/optimization/` and are run from that directory.

| Figure | Script | Runs on |
|---|---|---|
| `reference_comparison.png` | `make_reference_figure.py --out ...` | laptop |
| `asci_cnr_tradeoff.png` | `make_tradeoff_figure.py` | laptop |
| `mpxi_optimum.png` | `make_mpxi_optimum_figure.py` | laptop |
| `search_convergence.png` | `make_convergence_figure.py` | laptop |
| `recon_comparison.png` | `make_recon_figure.py` | CCR (needs recon .npz) |
| `ppdf_beam.png` | `make_ppdf_figure.py` | CCR (needs raw PPDF .hdf5) |

The two CCR figures read files that exist only on the cluster under
`/vscratch`; both exit with a clear message if run on the laptop. Copy the
resulting .png back into this directory.

`scanner_geometry.png`, `conventional_collimation.png` and
`selfcollimation_concept.png` are schematics rather than data plots, and no
generator for them is kept in the repo — they were drawn once in throwaway
scripts. Changing them means redrawing them.

Figures dated August (`headtohead.png`, `hypervolume_convergence.png`,
`metric_scatter_combined.png`, `pareto_expansion.png`) are from the earlier
4D study and are not used by the current deck.
