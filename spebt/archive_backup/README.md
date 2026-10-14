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

## Refreshing it

    bash archive_backup/refresh.sh && git add archive_backup && git commit

Do this after any batch of evaluations finishes.
