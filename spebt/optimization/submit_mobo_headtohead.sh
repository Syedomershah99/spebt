#!/bin/bash
# One arm of the two-objective vs five-objective head-to-head.
#
# Both arms start from the SAME fresh 6D LHS seeds and differ only in what they
# optimize, which is what the replay could not establish: the replay's candidate
# pool was chosen by the five-objective campaign, so a subset winning there was
# always searching a space something else mapped.
#
# Usage:
#   sbatch submit_mobo_headtohead.sh 2obj
#   sbatch submit_mobo_headtohead.sh 5obj
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --job-name=mobo_h2h
#SBATCH --output=results/slurm_logs/out/h2h_%j.out
#SBATCH --error=results/slurm_logs/err/h2h_%j.err

set -euo pipefail

ARM="${1:-}"
if [[ "$ARM" != "2obj" && "$ARM" != "5obj" ]]; then
    echo "ERROR: pass an arm: sbatch submit_mobo_headtohead.sh {2obj|5obj}" >&2
    exit 2
fi

BASE=/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt
source /vscratch/grp-rutaoyao/Omer/.venv/bin/activate
cd "$BASE/optimization"

# Each arm gets its own results directory: manifest, results CSV, singleton lock
# and logs. Sharing one would make the two arms claim the same manifest indices
# and interleave rows into a single CSV, which is unrecoverable after the fact.
export MOBO_RESULTS_DIR="$BASE/optimization/results_h2h_${ARM}"

if [[ "$ARM" == "2obj" ]]; then
    export MOBO_OBJECTIVES="cnr_sector_mean,mpxi_windowed_active_mean"
else
    # Unset means all five. Set explicitly so the log records what ran rather
    # than leaving it to whatever the default was on the day.
    export MOBO_OBJECTIVES="fwhm_weighted_mean,asci_pct_fwhm0p45,ppds_ring1,mpxi_windowed_active_mean,cnr_sector_mean"
fi

mkdir -p "$MOBO_RESULTS_DIR/slurm_logs/out" "$MOBO_RESULTS_DIR/slurm_logs/err"

echo "arm         : $ARM"
echo "results dir : $MOBO_RESULTS_DIR"
echo "objectives  : $MOBO_OBJECTIVES"

# 80 optimized iterations on top of the 21 shared LHS seeds. The seeds live in
# the results CSV as training data but are NOT manifest rows, so --max_iters
# counts optimized iterations only.
#
# Both arms get the same budget. The comparison is best CNR against iteration
# number, so an unequal budget would make the curves incomparable past the
# shorter one.
python run_mobo_loop.py --max_iters 80
