#!/bin/bash -l
# One Empire AI Grace node hosting several independent q=1 searches.
#
# Each search is a directory holding its own results_summary_mobo.csv (warm
# start: the 21 evaluated starting designs) or an lhs_pending.csv (fresh start:
# 21 designs still to evaluate). For each search, in its own slot:
#   1. evaluate every lhs_pending.csv design that has no results row yet
#   2. run its controller, run_mobo_loop.py, with MOBO_EXECUTOR=local, so each
#      proposed design runs on this node as a child process
# Every PRUNE_EVERY seconds the PPDFs of designs no longer worth re-measuring
# are deleted (the top KEEP_TOP by CNR overall and inside the MPXI window are
# kept), and the CSVs, manifests and logs are mirrored to project space.
# The job ends when every search has ended, so no node sits idle on the bill.
#
# Settings come from the environment (sbatch --export=ALL,...):
#   SEARCH_ROOT     directory holding this node's searches (required); scratch
#                   is visible only from Grace nodes, so the job itself creates
#                   the searches there on its first run, from SETUP
#   SETUP           make_searches.py arguments for that first run, e.g.
#                   "--warm 3 --fresh 3 --first_seed 10"; ignored once
#                   SEARCH_ROOT exists, so a resubmitted job resumes its searches
#   MAX_ITERS       optimizer iterations per search (default 100)
#   MOBO_OBJECTIVES, MOBO_CONSTRAINTS, MOBO_RING2_MAX  as in mobo_agent.py
#                   (defaults: CNR alone, MPXI 1.5-2.5, ring 2 up to 960)
#
# Production (one job per node, each with its own SEARCH_ROOT):
#   sbatch --qos=long --time=6-00:00:00 \
#     --export=ALL,SEARCH_ROOT=/ddn/scratch/suny/oshah/node1,SETUP="--warm 3 --fresh 3 --first_seed 10" \
#     run_node_searches.sh
#
#SBATCH --job-name=eai_searches
#SBATCH --account=su_ryao1_develop
#SBATCH --partition=grace
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/projects/su/ryao1/develop/oshah/logs/searches_%j.out

set -uo pipefail
P=${P:-/projects/su/ryao1/develop/oshah}
JOB=${SLURM_JOB_ID:-local}
CODE=${CODE:-$P/spebt}
SEARCH_ROOT=${SEARCH_ROOT:?set SEARCH_ROOT to the search directory for this node}
MAX_ITERS=${MAX_ITERS:-100}
KEEP_TOP=${KEEP_TOP:-5}
PRUNE_EVERY=${PRUNE_EVERY:-1200}
MIRROR=${MIRROR:-$P/mirror}
export SPEBT_VENV=${SPEBT_VENV:-$HOME/venvs/spebt-arm64} MOBO_CODE_DIR=$CODE CODE_DIR=$CODE
export PHANTOM_PATH=$CODE/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt
export MOBO_EXECUTOR=local CNR_CACHE_PPDFS=1
export MOBO_OBJECTIVES=${MOBO_OBJECTIVES:-cnr_sector_mean}
export MOBO_CONSTRAINTS=${MOBO_CONSTRAINTS:-mpxi_windowed_active_mean:1.5:2.5}
export MOBO_RING2_MAX=${MOBO_RING2_MAX:-960}
export CUDA_VISIBLE_DEVICES="" HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source "$SPEBT_VENV/bin/activate"

if [ ! -d "$SEARCH_ROOT" ]; then
  echo "creating searches in $SEARCH_ROOT: make_searches.py ${SETUP:?set SETUP for the first run}"
  (cd "$CODE/optimization" && python eai/make_searches.py --root "$SEARCH_ROOT" $SETUP) || exit 1
fi
SEARCHES=$(find "$SEARCH_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | tr '\n' ' ')
echo "searches job $JOB on $(hostname) at $(date), code $(cat $CODE/COMMIT)"
echo "searches: $SEARCHES"
echo "objectives $MOBO_OBJECTIVES | constraints $MOBO_CONSTRAINTS | ring 2 <= $MOBO_RING2_MAX | $MAX_ITERS iterations"
mkdir -p "$P/logs"
PROV=$P/logs/searches_${JOB}_provenance.txt
{ cat "$CODE/COMMIT"; hostname; lscpu | grep -E "Model name|^CPU\(s\)"; env | grep -E "^MOBO_|^CNR_|^SPEBT"; pip freeze; } > "$PROV"

evaluate_pending() {   # the fresh start's designs, through the same pipeline
  local S=$1 name diam nap nd1 nd2 d2 d3 state
  [ -f "$S/lhs_pending.csv" ] || return 0
  mkdir -p "$S/slurm_logs/out" "$S/slurm_logs/err"
  while IFS=, read -r name diam nap nd1 nd2 d2 d3 || [ -n "$name" ]; do
    [ "$name" = config ] && continue
    # The design's state in the results CSV. Read it with pandas: config is not
    # the first column (eai-v1 tested "^name," and so re-ran every design).
    state=$(python - "$S/results_summary_mobo.csv" "$name" <<'EOF'
import os, sys
import pandas as pd
p, n = sys.argv[1], sys.argv[2]
d = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["config"])
r = d[d["config"].astype(str) == n]
print("none" if r.empty else
      "done" if "cnr_sector_mean" in r and r["cnr_sector_mean"].notna().any() else "no_cnr")
EOF
)
    case "$state" in
      done) continue ;;
      no_cnr)   # metrics row but no CNR (stopped in step 4): only the CNR step
        python "$CODE/optimization/compute_cnr.py" --work_dir "$S/$name" --phantom_path "$PHANTOM_PATH" \
          --out_csv "$S/results_summary_mobo.csv" --config_name "$name" --iterations 150 \
          --seed_from_config --cache_ppdfs >> "$S/slurm_logs/out/${name}_local.out" 2>&1 ;;
      *)
        CONFIG_NAME=$name WORK_DIR=$S/$name APERTURE_DIAM=$diam N_APERTURES=$nap \
        N_DET_RING1=$nd1 N_DET_RING2=$nd2 D2_INNER=$d2 D3_INNER=$d3 A_MM=0.2 B_MM=0.2 \
        RESULTS_CSV=$S/results_summary_mobo.csv \
          bash "$CODE/optimization/run_sai_pipeline.sh" \
          > "$S/slurm_logs/out/${name}_local.out" 2> "$S/slurm_logs/err/${name}_local.err" ;;
    esac
  done < "$S/lhs_pending.csv"
}

slot() {
  local S=$1
  echo "[$(date +%H:%M)] $(basename $S): starting"
  evaluate_pending "$S"
  echo "=== job $JOB on $(hostname), $(date)" >> "$S/controller.log"
  MOBO_RESULTS_DIR=$S OMP_NUM_THREADS=4 python "$CODE/optimization/run_mobo_loop.py" \
    --max_iters "$MAX_ITERS" >> "$S/controller.log" 2>&1
  echo "[$(date +%H:%M)] $(basename $S): controller exited $?"
}

housekeeping() {
  local S n h
  for S in $SEARCHES; do
    { echo "=== $(date)"; python "$CODE/optimization/prune_ppdfs.py" --results_dir "$S" \
        --keep_top "$KEEP_TOP" --mpxi_window 1.5:2.5 --masks --delete; } >> "$S/prune.log" 2>&1
    h="$MIRROR/$(basename "$S")"
    mkdir -p "$h/history"
    rsync -a --include='*/' --include='*.csv' --include='*.log' --include='*.out' --include='*.err' \
      --exclude='*' "$S/" "$h/" 2>> "$MIRROR/rsync_errors_$JOB.log"
    # Timestamped copies whenever they change, so a damaged file can be traced.
    for f in results_summary_mobo.csv mobo_manifest.csv; do
      [ -f "$S/$f" ] || continue
      last=$(ls -t "$h/history/${f%.csv}_"*.csv 2>/dev/null | head -1)
      { [ -n "$last" ] && cmp -s "$S/$f" "$last"; } || cp "$S/$f" "$h/history/${f%.csv}_$(date +%m%d_%H%M).csv"
    done
    n=0
    [ -f "$S/results_summary_mobo.csv" ] && n=$(($(wc -l < "$S/results_summary_mobo.csv") - 1))
    echo "[$(date +%H:%M)] $(basename "$S"): $n rows"
  done
  echo "[$(date +%H:%M)] scratch: $(lfs quota -h -u "$USER" /ddn/scratch 2>/dev/null | awk 'NR==3 {print $2 " of " $4}')"
}

PIDS=()
for S in $SEARCHES; do
  slot "$S" &
  PIDS+=($!)
done
( while true; do sleep "$PRUNE_EVERY"; housekeeping; done ) &
HK=$!
wait "${PIDS[@]}"
kill $HK 2>/dev/null
housekeeping
echo "all searches ended at $(date)"
if [ "$JOB" != local ]; then
  sacct -j "$JOB" -X -n --format=JobID,Partition,QOS,AllocTRES%60,Elapsed
fi
