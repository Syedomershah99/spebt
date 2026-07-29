#!/bin/bash
# Submit the sequential MOBO controller for the SAI SC-SPECT design search.
#
# The controller (run_mobo_loop.py) reads the manifest to figure out how many
# iterations have already been completed and picks up from there — this script
# is safe to re-submit any time; it will only propose the delta up to
# --max_iters.
#
# Committed to git so it survives vscratch auto-cleanups.
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --job-name=mobo_loop
#SBATCH --output=results/slurm_logs/out/mobo_%j.out
#SBATCH --error=results/slurm_logs/err/mobo_%j.err

set -euo pipefail

source /vscratch/grp-rutaoyao/Omer/.venv/bin/activate
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization

mkdir -p results/slurm_logs/out results/slurm_logs/err

# Bump this ceiling as needed. Controller no-ops once the manifest already has
# --max_iters rows, so it's safe to set higher than we plan to run in one go.
# 180 closed out the 4D campaign; 260 gives 80 iterations of 6D search on the
# revised objective set.
python run_mobo_loop.py --max_iters 260
