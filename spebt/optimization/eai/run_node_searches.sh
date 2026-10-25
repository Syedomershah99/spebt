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
#   SEARCHES        space-separated search directories (required)
#   MAX_ITERS       optimizer iterations per search (default 100)
#   MOBO_OBJECTIVES, MOBO_CONSTRAINTS, MOBO_RING2_MAX  as in mobo_agent.py
#                   (defaults: CNR alone, MPXI 1.5-2.5, ring 2 up to 960)
#
# Production: sbatch --qos=long --time=6-00:00:00 --export=ALL,SEARCHES="..." run_node_searches.sh
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
P=/projects/su/ryao1/develop/oshah
CODE=${CODE:-$P/spebt}
SEARCHES=${SEARCHES:?set SEARCHES to the search directories}
MAX_ITERS=${MAX_ITERS:-100}
KEEP_TOP=${KEEP_TOP:-5}
PRUNE_EVERY=${PRUNE_EVERY:-1200}
MIRROR=${MIRROR:-$P/mirror}
export SPEBT_VENV=$HOME/venvs/spebt-arm64 MOBO_CODE_DIR=$CODE CODE_DIR=$CODE
export PHANTOM_PATH=$CODE/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt
export MOBO_EXECUTOR=local CNR_CACHE_PPDFS=1
export MOBO_OBJECTIVES=${MOBO_OBJECTIVES:-cnr_sector_mean}
export MOBO_CONSTRAINTS=${MOBO_CONSTRAINTS:-mpxi_windowed_active_mean:1.5:2.5}
export MOBO_RING2_MAX=${MOBO_RING2_MAX:-960}
export CUDA_VISIBLE_DEVICES="" HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source "$SPEBT_VENV/bin/activate"

echo "searches job $SLURM_JOB_ID on $(hostname) at $(date), code $(cat $CODE/COMMIT)"
echo "objectives $MOBO_OBJECTIVES | constraints $MOBO_CONSTRAINTS | ring 2 <= $MOBO_RING2_MAX | $MAX_ITERS iterations"
PROV=$P/logs/searches_${SLURM_JOB_ID}_provenance.txt
{ cat "$CODE/COMMIT"; hostname; lscpu | grep -E "Model name|^CPU\(s\)"; env | grep -E "^MOBO_|^CNR_|^SPEBT"; pip freeze; } > "$PROV"

evaluate_pending() {   # the fresh start's designs, through the same pipeline
  local S=$1 name diam nap nd1 nd2 d2 d3
  [ -f "$S/lhs_pending.csv" ] || return 0
  mkdir -p "$S/slurm_logs/out" "$S/slurm_logs/err"
  while IFS=, read -r name diam nap nd1 nd2 d2 d3; do
    [ "$name" = config ] && continue
    grep -q "^$name," "$S/results_summary_mobo.csv" 2>/dev/null && continue
    CONFIG_NAME=$name WORK_DIR=$S/$name APERTURE_DIAM=$diam N_APERTURES=$nap \
    N_DET_RING1=$nd1 N_DET_RING2=$nd2 D2_INNER=$d2 D3_INNER=$d3 A_MM=0.2 B_MM=0.2 \
    RESULTS_CSV=$S/results_summary_mobo.csv \
      bash "$CODE/optimization/run_sai_pipeline.sh" \
      > "$S/slurm_logs/out/${name}_local.out" 2> "$S/slurm_logs/err/${name}_local.err"
  done < "$S/lhs_pending.csv"
}

slot() {
  local S=$1
  echo "[$(date +%H:%M)] $(basename $S): starting"
  evaluate_pending "$S"
  MOBO_RESULTS_DIR=$S OMP_NUM_THREADS=4 python "$CODE/optimization/run_mobo_loop.py" \
    --max_iters "$MAX_ITERS" > "$S/controller.log" 2>&1
  echo "[$(date +%H:%M)] $(basename $S): controller exited $?"
}

housekeeping() {
  for S in $SEARCHES; do
    python "$CODE/optimization/prune_ppdfs.py" --results_dir "$S" --keep_top "$KEEP_TOP" \
      --mpxi_window 1.5:2.5 --delete > "$S/prune.log" 2>&1
    mkdir -p "$MIRROR/$(basename $S)"
    rsync -a --include='*/' --include='*.csv' --include='*.log' --include='*.out' \
      --exclude='*' "$S/" "$MIRROR/$(basename $S)/" 2>/dev/null
    n=$(($(wc -l < "$S/results_summary_mobo.csv" 2>/dev/null || echo 1) - 1))
    echo "[$(date +%H:%M)] $(basename $S): $n rows"
  done
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
sacct -j $SLURM_JOB_ID -X -n --format=JobID,Partition,QOS,AllocTRES%60,Elapsed
