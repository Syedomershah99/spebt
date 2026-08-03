#!/bin/bash
# Replay the design archive under smaller objective subsets.
#
# Read-only against the archive — safe to run while the MOBO controller is live.
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=4
# 32G was OOM-killed after 4.5 h. The cost is not the point count, as that first
# estimate assumed — it is qLogNEHVI's hypervolume partition, which grows with
# the size of the Pareto front and the objective count. The script now chunks
# candidate scoring and uses approximate partitioning at m=5, which measured
# 0.83 GB peak on a synthetic archive of the same shape.
#
# Sized at 128G anyway, because that measurement is a floor rather than a
# prediction: the synthetic archive had a 12% Pareto front and the real one has
# 44%, and I could not reproduce 44% closely enough to trust an extrapolation.
# The job is short, so over-requesting costs little; another OOM costs hours.
#SBATCH --mem=128G
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
