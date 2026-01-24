#!/bin/bash
#SBATCH --job-name=ppdf_t8
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=nih
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/out/ppdf_%A_%a.out
#SBATCH --error=slurm_logs/err/ppdf_%A_%a.err
#SBATCH --mail-user=syedomer@buffalo.edu
#SBATCH --mail-type=FAIL,END

set -euo pipefail

mkdir -p slurm_logs/out slurm_logs/err

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Start: $(date)"
echo "=========================================================="

# --- Activate venv ---
source /vscratch/grp-rutaoyao/Omer/.venv/bin/activate

# --- Make scanner_modeling importable ---
export PYTHONPATH=/vscratch/grp-rutaoyao/Omer/spebt/pymatcal:$PYTHONPATH

LAYOUT_FILE="/vscratch/grp-rutaoyao/Omer/spebt/geometry/scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"

echo "Running T8 for layout=${SLURM_ARRAY_TASK_ID}"
python /vscratch/grp-rutaoyao/Omer/spebt/pymatcal/arg_ppdf_t8.py \
  ${SLURM_ARRAY_TASK_ID} \
  --layout_file "${LAYOUT_FILE}" \
  --a_mm 0.8 --b_mm 0.8

echo "=========================================================="
echo "End: $(date)"
echo "Exit code: $?"
echo "=========================================================="