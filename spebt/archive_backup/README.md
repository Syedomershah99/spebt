# Archive backup

Every evaluated-design result, as CSV. 1.1 MB.

## Why this exists

The results live under `optimization/results/` on /vscratch, which CCR purges.
The purge has already taken most of the raw intermediates: of 391 config
directories, 64 still have PPDFs and 67 still have beam files.

That loss is survivable, because PPDFs and beam files regenerate from the six
design parameters. We verified this directly: rebuilding a design in a fresh
directory weeks later reproduced its per-seed CNR to six decimal places.

What does NOT regenerate cheaply is the measurement record itself. These CSVs
are the outcome of roughly 1200 full pipeline evaluations. Losing them means
re-running all of it. They were untracked until 2026-10-13, sitting on the
filesystem doing the purging.

So: back up the 1.1 MB that is irreplaceable, not the 600 GB that is merely
expensive.

## What is here

    results_summary_mobo.csv   the main archive, 337 designs
    cnr_repeats/               5-seed CNR measurements
    misc/cnr_*/                later re-measurement batches
    h2h/                       the ten head-to-head arm archives
    asci_binsize.csv           ASCI vs angular bin width
    ppds_window.csv            PPDS vs FWHM window
    mpxi_variants*.csv         MPXI definition variants
    mobo_manifest.csv          iteration manifest
    fwhm_total_weight.csv      weighted FWHM with each strip's whole PPDF as weight (col 7 x
                               col 9), 330 designs (compute_fwhm_total_weight.py, 30 Oct 2026);
                               n64 counts strips with FWHM = 64 mm (width not found), and the
                               *_no64 columns are the plain and mean-weighted FWHM without them
    ppds_radial_recheck.csv    per-ring PPDS as searched (_bug) and with the strip-length fix
                               (_fix), the 316 designs whose PPDFs and strip files remain
                               (recheck_ppds_radial.py, 30 Oct 2026); ppds_recheck/ holds its
                               design list, CCR job scripts and the analysis output
    design_maps/               benchmark (tmi_reference_000) and MOBO-optimized (nd2fine_960)
                               maps and strips for R11 slides 13-15 (compute_design_maps.py)
    fwhm_filter.csv            FWHM with the 0.45 mm strip filter, 330 designs with beam
                               files (optimization/compute_fwhm_filter.py, 28 Oct 2026)

## Refreshing it

    bash archive_backup/refresh.sh && git add archive_backup && git commit

Do this after any batch of evaluations finishes.

## R12 checks (30 Oct 2026)

    r12_checks/rod_position_cnr/   Dr. Yao's slide 14 question: does the CNR gain come from rods
                                   away from the centre? The 5 saved reconstructions per design,
                                   the pipeline's CNR, rods split at the median distance 2.57 mm:
                                   inner +40% (ASCI 44 vs 64%), outer +23% (step2_results.json)
    r12_checks/sensitivity/        his slide 15 question on the colour-bar magnitudes: the maps
                                   summed T8's 8 bed positions (now their mean); 2-D in-plane
                                   probabilities checked by an independent ray cast; 3-D pinhole
                                   estimates against the SAI paper (analysis.json, raycast_report.json)
