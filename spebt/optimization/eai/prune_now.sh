#!/bin/bash
# Prune one node's searches by hand with a 30-minute recency guard instead of
# the launcher's 2 hours (no measured design is re-run any more), same keep set.
#   srun --jobid=JOB --overlap -N1 -n1 bash prune_now.sh ROOT
source ~/venvs/spebt-arm64/bin/activate
L=/projects/su/ryao1/develop/oshah/logs/manual_prune.log
for S in "$1"/*/; do
  [ -f "$S/results_summary_mobo.csv" ] || continue
  { echo "=== $(date) $S"
    python /projects/su/ryao1/develop/oshah/spebt/optimization/prune_ppdfs.py --results_dir "$S" \
      --keep_top 5 --mpxi_window 1.5:2.5 --masks --delete --min_age_hours 0.5 2>&1 | grep -E "Keeping|Deleted|reclaimed|warn"
  } >> "$L"
done
echo "$1 done"
