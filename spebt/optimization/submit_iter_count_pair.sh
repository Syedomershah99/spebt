#!/bin/bash
# CNR of the published benchmark and the MOBO-optimized design against the
# ML-EM iteration count (ARM=iter) and the count level (ARM=count).
#
# The critique Dr. Yao forwarded on 23 Oct 2026 asked how sensitive the CNR
# improvement is to the iteration count and the noise level. Every design in
# the campaign was reconstructed at 150 iterations and one count level
# (T_sec 10). The July sweep (submit_iteration_sweep.sh) checked the ranking of
# three other designs at one seed; this one measures the pair the talk compares,
# over the five seeds of the repeat study.
#
# Designs: tmi_reference_000 (the benchmark) and nd2fine_960, a rebuild of
# mobo_0296 from its parameters (mobo_0296's own PPDFs were pruned; its one
# surviving file is byte-identical to nd2fine_960's, and the two designs'
# stored per-seed CNRs agree to within 6.4e-7, float32 rounding).
#
# Every task runs 400 iterations through run_iter_count_pair.py, which keeps
# and scores every iterate, so one run gives CNR at each count from 1 to 400:
#   ARM=iter   T_sec 10 (the campaign's), seeds 0-4: 5 tasks per design
#   ARM=count  T_sec 2.5 5 20 40, seeds 0-4: 20 tasks per design
# A seed fixes the Poisson draw only for one design at one T_sec, so all
# iteration counts of a run share the same data, but the two designs, and two
# count levels, share only the seed index: their noise draws are independent.
#
# The T_sec 10 runs also re-create the stored repeats (cnr_repeat_seed<S>),
# which collect_iter_count_pair.py requires: the same projections byte for byte
# and the same CNR at iteration 150. The Poisson draw depends on float32
# rounding in the forward projection, which can differ between CPU types, so
# each design runs on the CPU its stored repeats used, set by --constraint and
# checked below (on those CPUs all ten re-created runs matched bit for bit):
#   tmi_reference_000  Xeon Gold 6330 (Ice Lake), job 25911155
#   nd2fine_960        Xeon Gold 6448Y (Sapphire Rapids), job 26251666
#
# Each task writes <work_dir>/iterpair_N400_T<T>_seed<S>/. Reads and writes only
# the project directory: /vscratch is read-only from 27 Oct 2026.
#
# Usage (from anywhere; the five-task iter runs first, then check the gate):
#   sbatch --array=0-4 --constraint=CPU-Gold-6330 --export=ALL,ARM=iter,DESIGN=tmi_reference_000 submit_iter_count_pair.sh
#   sbatch --array=0-4 --constraint=CPU-Gold-6448Y --export=ALL,ARM=iter,DESIGN=nd2fine_960 submit_iter_count_pair.sh
#   python3 -B collect_iter_count_pair.py
#   sbatch --array=0-19 --constraint=CPU-Gold-6330 --export=ALL,ARM=count,DESIGN=tmi_reference_000 submit_iter_count_pair.sh
#   sbatch --array=0-19 --constraint=CPU-Gold-6448Y --export=ALL,ARM=count,DESIGN=nd2fine_960 submit_iter_count_pair.sh
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=rutaoyao
# Two cores as in the stored repeats. The cached PPDFs take 8.6 GB (benchmark)
# or 9.5 GB (nd2fine_960). An iteration is then a few seconds of arithmetic,
# so 400 iterations should take well under an hour.
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=iterpair
# Absolute, so the logs cannot land on /vscratch whatever directory sbatch runs in.
#SBATCH --output=/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization/results/slurm_logs/out/iterpair_%A_%a.out
#SBATCH --error=/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization/results/slurm_logs/err/iterpair_%A_%a.err

set -euo pipefail

VENV_PY="${SPEBT_VENV:-$HOME/spebt-venv}/bin/python3"
CODE_DIR="$(readlink -f "${CODE_DIR:-/projects/academic/rutaoyao/Omer/spebt/spebt/spebt}")"
case "${CODE_DIR}" in
  /vscratch/*) echo "ERROR: CODE_DIR is on /vscratch, which is read-only from 27 Oct 2026." >&2; exit 2 ;;
esac
cd "${CODE_DIR}/optimization"
RESULTS_DIR="${CODE_DIR}/optimization/results"
PHANTOM_PATH="${CODE_DIR}/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"
# No .pyc rewrites in the shared tree, and no GPU: the stored repeats ran on CPU.
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
# Unbuffered, so a task killed at its time limit still leaves its log.
export PYTHONUNBUFFERED=1

case "${DESIGN:-}" in
  tmi_reference_000) CPU="Gold 6330" ;;
  nd2fine_960)       CPU="Gold 6448Y" ;;
  *) echo "ERROR: set DESIGN=tmi_reference_000 or DESIGN=nd2fine_960." >&2; exit 2 ;;
esac
model="$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2-)"
if [[ "${model}" != *"${CPU}"* ]]; then
  echo "ERROR: ${DESIGN} must run on Xeon ${CPU} like its stored repeats, but this node has${model}." >&2
  echo "       Submit it with --constraint=CPU-${CPU// /-}." >&2
  exit 3
fi

SEEDS=(0 1 2 3 4)
case "${ARM:-}" in
  iter)  LEVELS=(10) ;;
  count) LEVELS=(2.5 5 20 40) ;;
  *) echo "ERROR: set ARM=iter or ARM=count (sbatch --export=ALL,ARM=...)." >&2; exit 2 ;;
esac
n_tasks=$(( ${#LEVELS[@]} * ${#SEEDS[@]} ))
t=${SLURM_ARRAY_TASK_ID}
if (( t >= n_tasks )); then
  echo "ERROR: task ${t} is past this arm's ${n_tasks} tasks; use --array=0-$(( n_tasks - 1 ))." >&2
  exit 2
fi
T=${LEVELS[$(( t / ${#SEEDS[@]} ))]}
seed=${SEEDS[$(( t % ${#SEEDS[@]} ))]}
N=400
tag="N${N}_T${T/./p}_seed${seed}"

echo "[${ARM} ${t}] ${DESIGN} ${tag} on $(hostname) (${model# }) at $(date)"
"${VENV_PY}" run_iter_count_pair.py \
    --work_dir "${RESULTS_DIR}/${DESIGN}" \
    --phantom_path "${PHANTOM_PATH}" \
    --out_subdir "iterpair_${tag}" \
    --seed "${seed}" \
    --iterations "${N}" \
    --T_sec "${T}"
echo "[${ARM} ${t}] done at $(date)"
