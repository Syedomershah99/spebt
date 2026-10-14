#!/usr/bin/env bash
# Status of the ASCI fine-binning job on CCR.
#
# The job prints progress to a block-buffered stdout, so a running job often
# shows nothing in its .out file. Absence of progress lines is NOT a hang.
# Run with --watch to poll every 5 minutes until it leaves the queue.
set -uo pipefail

JOB="${ASCI_JOB:-26493634}"
REMOTE=/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization
LOGDIR="$REMOTE/results/slurm_logs"

status_once() {
  ssh ccr "
    printf '%s  job %s  ' \"\$(date '+%Y-%m-%d %H:%M')\" '$JOB'
    sacct -j '$JOB' --format=State,Elapsed -X -n 2>/dev/null | head -1 | tr -s ' '
    echo '  --- last stdout ---'
    tail -3 '$LOGDIR/out/asci_fine_$JOB.out' 2>/dev/null || echo '  (no stdout yet)'
    err=\$(tail -5 '$LOGDIR/err/asci_fine_$JOB.err' 2>/dev/null)
    if [ -n \"\$err\" ]; then echo '  --- stderr ---'; echo \"\$err\"; fi
    echo '  --- result csv ---'
    ls -l '$REMOTE/results/asci_finebins.csv' 2>/dev/null || echo '  not written yet'
  "
}

if [ "${1:-}" = "--watch" ]; then
  while true; do
    out=$(status_once)
    echo "$out"
    case "$out" in
      *COMPLETED*|*FAILED*|*CANCELLED*|*TIMEOUT*) echo "=== job left the queue ==="; break ;;
    esac
    sleep 300
  done
else
  status_once
fi
