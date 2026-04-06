#!/bin/bash
#SBATCH --job-name=sai_pipeline
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/out/pipeline_%A_%a.out
#SBATCH --error=slurm_logs/err/pipeline_%A_%a.err
#SBATCH --mail-user=syedomer@buffalo.edu
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# ============================================================
# SAI SC-SPECT per-config pipeline
# Called by run_bo_loop.py for each BO-proposed configuration.
#
# Environment variables (passed via --export):
#   WORK_DIR       - output directory for this config
#   APERTURE_DIAM  - aperture diameter in mm
#   N_APERTURES    - number of apertures
#   A_MM           - T8 ellipse semi-axis X (default 0.2)
#   B_MM           - T8 ellipse semi-axis Y (default 0.2)
#   CODE_DIR       - base code directory
#   RESULTS_CSV    - path to append JI results
#   CONFIG_NAME    - identifier for this config
# ============================================================

source /vscratch/grp-rutaoyao/Omer/.venv/bin/activate
export PYTHONPATH="${CODE_DIR}/pymatcal:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING=FALSE

# Defaults for T8 (fixed during BO)
A_MM="${A_MM:-0.2}"
B_MM="${B_MM:-0.2}"

mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

echo "=================================================="
echo "SAI Pipeline | $(date)"
echo "Config: ${CONFIG_NAME}"
echo "  aperture_diam = ${APERTURE_DIAM} mm"
echo "  n_apertures   = ${N_APERTURES}"
echo "  a_mm=${A_MM}  b_mm=${B_MM}"
echo "  work_dir = ${WORK_DIR}"
echo "  CPUs = ${SLURM_CPUS_PER_TASK}"
echo "=================================================="

# -------------------------------------------------------
# Step 0: Generate geometry
# -------------------------------------------------------
echo "[0/3] Generating scanner geometry..."
python "${CODE_DIR}/geometry/generate_mph_scanner_circularfov.py" \
  --aperture_diam "${APERTURE_DIAM}" \
  --n_apertures "${N_APERTURES}" \
  --output_dir "${WORK_DIR}"

# Find the generated .tensor file
shopt -s nullglob
TENSORS=("${WORK_DIR}"/*.tensor)
if [ ${#TENSORS[@]} -eq 0 ]; then
  echo "[ERROR] No .tensor file generated in ${WORK_DIR}"
  exit 1
fi
TENSOR_FILE="${TENSORS[0]}"
echo "  Tensor file: ${TENSOR_FILE}"

# -------------------------------------------------------
# Step 1: PPDF computation (2 layouts × 8 T8 poses = 16 files)
# -------------------------------------------------------
echo "[1/3] Computing PPDFs (2 layouts × 8 T8 poses)..."
for layout_idx in 0 1; do
  echo "  Layout ${layout_idx}..."
  python "${CODE_DIR}/pymatcal/arg_ppdf_t8.py" \
    "${layout_idx}" \
    --layout_file "${TENSOR_FILE}" \
    --output_dir "${WORK_DIR}" \
    --a_mm "${A_MM}" \
    --b_mm "${B_MM}"
done

# Verify 16 HDF5 files
N_HDF5=$(ls "${WORK_DIR}"/position_*_ppdfs_t8_*.hdf5 2>/dev/null | wc -l)
echo "  Generated ${N_HDF5} PPDF files (expected 16)"
if [ "${N_HDF5}" -lt 16 ]; then
  echo "[ERROR] Expected 16 PPDF files, got ${N_HDF5}"
  exit 1
fi

# -------------------------------------------------------
# Step 2: Beam analysis (masks, properties, ASCI)
# Runs per-layout: extract masks → extract properties → ASCI histogram
# Uses T8-aggregated PPDFs (sums 8 poses per layout)
# -------------------------------------------------------
echo "[2/3] Beam analysis (masks → properties → ASCI)..."
export PYTHONPATH="${CODE_DIR}/pymatana/ppdf-analysis/beam-analysis:${PYTHONPATH:-}"

for layout_idx in 0 1; do
  echo "  Layout ${layout_idx}: extracting masks..."
  python "${CODE_DIR}/optimization/sai_extract_masks.py" \
    --layout_idx "${layout_idx}" \
    --work_dir "${WORK_DIR}" \
    --tensor_file "${TENSOR_FILE}"

  echo "  Layout ${layout_idx}: extracting properties..."
  python "${CODE_DIR}/optimization/sai_extract_props.py" \
    --layout_idx "${layout_idx}" \
    --work_dir "${WORK_DIR}" \
    --tensor_file "${TENSOR_FILE}"

  echo "  Layout ${layout_idx}: computing ASCI histogram..."
  python "${CODE_DIR}/optimization/sai_analyze_asci.py" \
    --layout_idx "${layout_idx}" \
    --work_dir "${WORK_DIR}"
done

# -------------------------------------------------------
# Step 3: Compute JI and append to results CSV
# -------------------------------------------------------
echo "[3/3] Computing JI..."
python "${CODE_DIR}/optimization/compute_ji.py" \
  --work_dir "${WORK_DIR}" \
  --out_csv "${RESULTS_CSV}" \
  --config_name "${CONFIG_NAME}" \
  --aperture_diam_mm "${APERTURE_DIAM}" \
  --n_apertures "${N_APERTURES}"

echo "=================================================="
echo "PIPELINE COMPLETE | $(date)"
echo "=================================================="
