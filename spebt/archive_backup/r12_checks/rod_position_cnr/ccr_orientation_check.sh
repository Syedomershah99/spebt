#!/bin/bash
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=r12_bias_orient
#SBATCH --output=/user/syedomer/r12_bias/logs/orient_%j.out
#SBATCH --error=/user/syedomer/r12_bias/logs/orient_%j.err
set -euo pipefail
source ~/spebt-venv/bin/activate
export PYTHONPATH=/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization:${PYTHONPATH:-}
cd ~/r12_bias
python3 -B ccr_orientation_check.py
