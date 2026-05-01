#!/bin/bash
#SBATCH --job-name=recon_compare
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/recon_compare_%j.out
#SBATCH --error=slurm_logs/recon_compare_%j.err
#SBATCH --mail-user=syedomer@buffalo.edu
#SBATCH --mail-type=FAIL,END

# ============================================================
# Reconstruction comparison: Baseline vs BO-Optimized
#
# Runs ML-EM on both configs and generates CNR comparison.
#
# Usage:
#   sbatch submit_recon_comparison.sh
# ============================================================

set -euo pipefail

source /vscratch/grp-rutaoyao/Omer/.venv/bin/activate

CODE_DIR="/vscratch/grp-rutaoyao/Omer/spebt/spebt"
RECON_SCRIPT="${CODE_DIR}/recon/run_recon_comparison.py"
RESULTS_BASE="${CODE_DIR}/recon/recon_results"
PHANTOM="/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"

# Paths to PPDF directories
BASELINE_PPDF="/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm"
BO_PPDF="/vscratch/grp-rutaoyao/Omer/spebt/spebt/optimization/results/bo_0013_ap0.5300_nap232"

mkdir -p "${RESULTS_BASE}"
mkdir -p "$(dirname "$0")/slurm_logs"

echo "=================================================="
echo "Reconstruction Comparison | $(date)"
echo "=================================================="

# Step 1: Baseline reconstruction
echo ""
echo "=== BASELINE CONFIG (d=0.4mm, n=180) ==="
python "${RECON_SCRIPT}" \
  --config baseline \
  --ppdf_dir "${BASELINE_PPDF}" \
  --phantom_path "${PHANTOM}" \
  --output_dir "${RESULTS_BASE}/baseline"

# Step 2: BO-optimized reconstruction
echo ""
echo "=== BO-OPTIMIZED CONFIG (d=0.53mm, n=232) ==="
python "${RECON_SCRIPT}" \
  --config bo_optimized \
  --ppdf_dir "${BO_PPDF}" \
  --phantom_path "${PHANTOM}" \
  --output_dir "${RESULTS_BASE}/bo_optimized"

# Step 3: Comparison
echo ""
echo "=== GENERATING COMPARISON ==="
python "${RECON_SCRIPT}" \
  --compare \
  --baseline_dir "${RESULTS_BASE}/baseline" \
  --bo_dir "${RESULTS_BASE}/bo_optimized" \
  --output_dir "${RESULTS_BASE}/comparison"

echo ""
echo "=================================================="
echo "ALL DONE | $(date)"
echo "=================================================="
