#!/bin/bash
# Re-collect the result CSVs into archive_backup/ for committing.
set -euo pipefail
cd "$(dirname "$0")/../optimization"
B=../archive_backup
mkdir -p "$B/cnr_repeats" "$B/h2h" "$B/misc"
cp results/results_summary_mobo.csv "$B/" 2>/dev/null || true
cp results/cnr_repeats/*.csv "$B/cnr_repeats/" 2>/dev/null || true
for d in results/cnr_*/; do
  n=$(basename "$d"); mkdir -p "$B/misc/$n"
  cp "$d"*.csv "$B/misc/$n/" 2>/dev/null || true
done
for f in results_h2h_*/results_summary_mobo.csv; do
  [ -f "$f" ] && cp "$f" "$B/h2h/$(dirname "$f").csv"
done
for f in mpxi_variants mpxi_variants_seed6d asci_binsize ppds_window mobo_manifest; do
  cp "results/$f.csv" "$B/" 2>/dev/null || true
done
echo "backed up $(find "$B" -name "*.csv" | wc -l | tr -d " ") CSVs, $(du -sh "$B" | cut -f1)"
