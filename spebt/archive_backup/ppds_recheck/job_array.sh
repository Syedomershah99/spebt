#!/bin/bash
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-31
#SBATCH --job-name=ppds_fix
#SBATCH --output=logs/ppds_%A_%a.out
#SBATCH --error=logs/ppds_%A_%a.err
set -euo pipefail
source $HOME/spebt-venv/bin/activate
cd $HOME/r11_ppds
export PYTHONPATH=/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization:${PYTHONPATH:-}
SECONDS=0
python recheck_ppds_radial.py designs.csv /projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization parts/part_${SLURM_ARRAY_TASK_ID}.csv --task ${SLURM_ARRAY_TASK_ID} --ntasks 32
echo took ${SECONDS}s
