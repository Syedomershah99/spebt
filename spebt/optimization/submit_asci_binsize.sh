#!/bin/bash
# Recompute ASCI at several angular bin widths to test for saturation.
#
# Per Dr. Yao, 13 Oct 2026: ASCI at 1-degree bins sits near 100% for almost
# every design, so it cannot rank them. Finer bins should separate designs
# again. This reuses the surviving beam property and mask files, so no ray
# tracing is needed.
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=4
# A boolean coverage grid at 3600 bins is 40000 x 3600 = 144 MB, and two are
# live at once while the next config loads. 48G is generous but this job is
# short and queues quickly at that size.
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=asci_bins
#SBATCH --mail-user=syedomer@buffalo.edu
#SBATCH --mail-type=FAIL,TIMEOUT
#SBATCH --output=results/slurm_logs/out/asci_bins_%j.out
#SBATCH --error=results/slurm_logs/err/asci_bins_%j.err

set -euo pipefail
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization
mkdir -p results/slurm_logs/out results/slurm_logs/err

python analyze_asci_binsize.py \
  --bins 360 720 1800 3600 \
  --limit "${ASCI_LIMIT:-40}" \
  --out results/asci_binsize.csv
