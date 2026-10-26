#!/bin/bash -l
# Rebuild the re-measurement designs that lack their PPDFs, on one Grace node,
# and write remeasure.sh's list (rebuild.py). CANDIDATES holds analyze_campaign.py's
# candidates.csv and ccr_candidates.csv, copied to /projects, separated by a space.
# CODE is required and may not be the live tree $P/spebt.
#
#   sbatch --qos=standard --time=12:00:00 --export=ALL,CODE=$P/code_<commit>,\
#   CANDIDATES="$P/remeasure_inputs/candidates.csv $P/remeasure_inputs/ccr_candidates.csv" rebuild.sh
#
#SBATCH --job-name=eai_rebuild
#SBATCH --account=su_ryao1_develop
#SBATCH --partition=grace
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/projects/su/ryao1/develop/oshah/logs/rebuild_%j.out

set -uo pipefail
P=/projects/su/ryao1/develop/oshah
CODE=${CODE:?set CODE to a code tree archived for this job, not $P/spebt}
[ "$(cd "$CODE" && pwd -P)" != "$(cd "$P/spebt" 2>/dev/null && pwd -P)" ] \
  || { echo "ERROR: CODE is the live tree $P/spebt"; exit 1; }
COMMIT=$(cat "$CODE/COMMIT") || { echo "ERROR: no $CODE/COMMIT"; exit 1; }
SCR=${SCR:-/ddn/scratch/suny/oshah}
CANDIDATES=${CANDIDATES:?set CANDIDATES to the candidate CSV files}
LIST=${LIST:-$P/remeasure_inputs/designs.txt}
export SPEBT_VENV=${SPEBT_VENV:-$HOME/venvs/spebt-arm64} CODE_DIR=$CODE
export PHANTOM_PATH=$CODE/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt
export CNR_CACHE_PPDFS=1 CUDA_VISIBLE_DEVICES="" HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source "$SPEBT_VENV/bin/activate"
echo "rebuild job ${SLURM_JOB_ID:-local} on $(hostname) at $(date), code $COMMIT ($CODE)"
mkdir -p "$(dirname "$LIST")"
python "$CODE/optimization/eai/rebuild.py" --root "$SCR/rebuild" --list "$LIST" \
  --logs "$(dirname "$LIST")/rebuild_logs" --code "$CODE" $CANDIDATES
rc=$?
echo "rebuild.py exit $rc at $(date)"
[ -z "${SLURM_JOB_ID:-}" ] || sacct -j "$SLURM_JOB_ID" -X -n --format=JobID,Partition,QOS,AllocTRES%60,Elapsed
exit $rc
