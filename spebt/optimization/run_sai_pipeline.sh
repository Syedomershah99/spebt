#!/bin/bash
#SBATCH --job-name=sai_pipeline
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/out/pipeline_%A_%a.out
#SBATCH --error=slurm_logs/err/pipeline_%A_%a.err
#SBATCH --mail-user=syedomer@buffalo.edu
#SBATCH --mail-type=FAIL,END

set -euo pipefail

# ============================================================
# SAI SC-SPECT per-config pipeline
# Called by run_bo_loop.py or submit_lhs_sweep.sh.
#
# Key optimization: 16 PPDF poses run in parallel (not sequential).
# With 25 CPUs, runs ~8 poses concurrently (each is ~single-threaded).
# Resume-safe: skips already-computed HDF5 files.
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

# Limit PyTorch threads per process (we run many in parallel)
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

# Defaults for T8 (fixed during BO)
A_MM="${A_MM:-0.2}"
B_MM="${B_MM:-0.2}"

# Max parallel PPDF processes (leave a few CPUs for OS/IO)
MAX_PARALLEL=12

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
echo "  Max parallel = ${MAX_PARALLEL}"
echo "=================================================="

# -------------------------------------------------------
# Step 0: Generate geometry (skip if .tensor already exists)
# -------------------------------------------------------
shopt -s nullglob
TENSORS=("${WORK_DIR}"/*.tensor)
if [ ${#TENSORS[@]} -gt 0 ]; then
  TENSOR_FILE="${TENSORS[0]}"
  echo "[0/3] Geometry already exists: ${TENSOR_FILE}"
else
  echo "[0/3] Generating scanner geometry..."
  python "${CODE_DIR}/geometry/generate_mph_scanner_circularfov.py" \
    --aperture_diam "${APERTURE_DIAM}" \
    --n_apertures "${N_APERTURES}" \
    --output_dir "${WORK_DIR}"

  TENSORS=("${WORK_DIR}"/*.tensor)
  if [ ${#TENSORS[@]} -eq 0 ]; then
    echo "[ERROR] No .tensor file generated in ${WORK_DIR}"
    exit 1
  fi
  TENSOR_FILE="${TENSORS[0]}"
fi
echo "  Tensor file: ${TENSOR_FILE}"

# -------------------------------------------------------
# Step 1: PPDF computation (2 layouts × 8 T8 poses = 16 files)
#   Parallelized: each pose runs as a separate background process.
#   Skips poses whose HDF5 output already exists (resume support).
# -------------------------------------------------------
echo "[1/3] Computing PPDFs (2 layouts × 8 T8 poses, parallel)..."

n_running=0
n_skipped=0
n_launched=0
PIDS=()

for layout_idx in 0 1; do
  for pose_idx in $(seq 0 7); do
    OUT_FILE="${WORK_DIR}/position_$(printf '%03d' ${layout_idx})_ppdfs_t8_$(printf '%02d' ${pose_idx}).hdf5"

    if [ -f "${OUT_FILE}" ]; then
      n_skipped=$((n_skipped + 1))
      continue
    fi

    # Throttle: wait if at max parallel
    while [ ${n_running} -ge ${MAX_PARALLEL} ]; do
      # Wait for any one child to finish
      wait -n 2>/dev/null || true
      n_running=$((n_running - 1))
    done

    echo "  Launching layout=${layout_idx} pose=${pose_idx}..."
    python "${CODE_DIR}/pymatcal/arg_ppdf_t8.py" \
      "${layout_idx}" \
      --layout_file "${TENSOR_FILE}" \
      --output_dir "${WORK_DIR}" \
      --a_mm "${A_MM}" \
      --b_mm "${B_MM}" \
      --pose_idx "${pose_idx}" &

    PIDS+=($!)
    n_running=$((n_running + 1))
    n_launched=$((n_launched + 1))
  done
done

echo "  Launched ${n_launched} poses, skipped ${n_skipped} (already exist)"
echo "  Waiting for all PPDF processes to finish..."

# Wait for all background jobs, track failures
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || FAIL=$((FAIL + 1))
done

if [ ${FAIL} -gt 0 ]; then
  echo "[ERROR] ${FAIL} PPDF process(es) failed"
  exit 1
fi

# Verify 16 HDF5 files
N_HDF5=$(ls "${WORK_DIR}"/position_*_ppdfs_t8_*.hdf5 2>/dev/null | wc -l)
echo "  Total PPDF files: ${N_HDF5} (expected 16)"
if [ "${N_HDF5}" -lt 16 ]; then
  echo "[ERROR] Expected 16 PPDF files, got ${N_HDF5}"
  exit 1
fi

echo "  Step 1 complete at $(date)"

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
