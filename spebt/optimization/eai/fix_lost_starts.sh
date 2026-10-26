#!/bin/bash
# Requeue starting designs lost to the 16:10 code sync (stale NFS handle), by
# appending each one's own line to its search's lhs_pending.csv, which the
# running eai-v1 launcher is still reading. Run inside the node's job:
#   srun --jobid=JOB --overlap -N1 -n1 bash fix_lost_starts.sh ROOT search:design ...
R=$1; shift
for x in "$@"; do
  S=$R/${x%%:*} n=${x#*:} f=$R/${x%%:*}/lhs_pending.csv
  if [ -s "$S/mobo_manifest.csv" ]; then echo "$S: optimizer already started, left alone"; continue; fi
  if grep -q "$n" "$S/results_summary_mobo.csv"; then echo "$S: $n has a row, left alone"; continue; fi
  k=$(grep -c "^$n," "$f")
  if [ "$k" -ne 1 ]; then echo "$S: $n listed $k times, left alone"; continue; fi
  l=$(grep "^$n," "$f")
  printf '%s\n' "$l" >> "$f"
  echo "$S: appended $n (now $(wc -l < "$f") lines)"
done
