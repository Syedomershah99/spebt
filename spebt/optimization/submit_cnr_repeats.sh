#!/bin/bash
# Repeat-run CNR measurement for the top designs, to establish the noise floor.
#
# The forward projection applies unseeded Poisson noise, so a single CNR value
# is one draw from a distribution. Two runs of mobo_0069 gave 4.4405 and 4.5688
# (spread 0.128) while the gap between mobo_0069 and mobo_0177 is 0.017 --
# i.e. the ranking among the top designs is not resolvable at n=1.
#
# This array runs N_SEEDS seeded repeats of each top config so we can report
# CNR as mean +/- std instead of a bare point estimate.
#
# Usage:  sbatch submit_cnr_repeats.sh
# Then:   python3 analyze_cnr_repeats.py
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=cnr_repeat
#SBATCH --array=0-14
#SBATCH --output=results/slurm_logs/out/cnr_repeat_%A_%a.out
#SBATCH --error=results/slurm_logs/err/cnr_repeat_%A_%a.err

set -euo pipefail

source /vscratch/grp-rutaoyao/Omer/.venv/bin/activate
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization

PHANTOM_PATH="/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"

# One CSV per array task: _write_cnr_to_csv does a read-modify-write, so 15
# concurrent tasks sharing one file would silently drop rows. The analysis
# script globs these back together (and can also read the per-seed npz files).
mkdir -p results/cnr_repeats
OUT_CSV="results/cnr_repeats/task_${SLURM_ARRAY_TASK_ID}.csv"

# Top designs from the 180-iteration campaign (all within ~0.17 CNR of each other)
CONFIGS=(
  "mobo_0069_ap0.3138_nap124_nd1_612_nd2_230"
  "mobo_0177_ap0.3512_nap97_nd1_604_nd2_584"
  "mobo_0173_ap0.3500_nap117_nd1_446_nd2_236"
)
N_SEEDS=5

cfg_idx=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
seed=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
config="${CONFIGS[$cfg_idx]}"

echo "[array ${SLURM_ARRAY_TASK_ID}] config=${config} seed=${seed}"

python3 compute_cnr.py \
    --work_dir "results/${config}" \
    --phantom_path "${PHANTOM_PATH}" \
    --out_csv "${OUT_CSV}" \
    --config_name "${config}__seed${seed}" \
    --out_subdir "cnr_repeat_seed${seed}" \
    --seed "${seed}" \
    --iterations 150
