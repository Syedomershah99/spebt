#!/bin/bash
#SBATCH --job-name=r12_sens_ray
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/user/syedomer/r12_sensitivity/logs/%x_%j.out
source ~/spebt-venv/bin/activate
export PYTHONPATH=/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization:$PYTHONPATH
cd ~/r12_sensitivity
python raycast_check.py ~/r12_sensitivity/out
python recompute_rows.py ~/r12_sensitivity/out
