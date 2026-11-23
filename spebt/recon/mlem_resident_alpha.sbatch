#!/bin/bash -l
#SBATCH --job-name=mlem_gpu
#SBATCH --account=su_ryao1_develop
#SBATCH --partition=alpha
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=mlem_gpu_%j.out
# One ML-EM reconstruction on one Empire AI GPU. Alpha sends every 1-GPU job to an RTX PRO 6000 (sm_120), so the
# environment needs a CUDA 12.8 build of torch; the Python/3.10.15 module cannot start on GPU nodes (libcrypt.so.2).
# Usage: CODE=/path/to/spebt/recon sbatch mlem_resident_alpha.sbatch PPDF_DIR PROJECTIONS.npy OUT.npz
set -e
source ~/venvs/gpu312cu128/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
python "${CODE:?set CODE to the spebt/recon folder}/mlem_resident.py" --ppdf_dir "$1" --projs "$2" --out "$3"
