#!/bin/bash
# Replay the design archive under smaller objective subsets.
#
# Read-only against the archive — safe to run while the MOBO controller is live.
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=4
# Far below the controller's 160G: the replay refits on at most ~200 points and
# never holds a full-archive baseline.
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=replay_subsets
#SBATCH --output=results/slurm_logs/out/replay_%j.out
#SBATCH --error=results/slurm_logs/err/replay_%j.err

set -euo pipefail

source /vscratch/grp-rutaoyao/Omer/.venv/bin/activate
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization

mkdir -p results/slurm_logs/out results/slurm_logs/err

python replay_objective_subsets.py \
    --results_csv results/results_summary_mobo.csv \
    --n_repeats 10 \
    --n_steps 80
