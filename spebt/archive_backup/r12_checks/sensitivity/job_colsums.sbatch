#!/bin/bash
#SBATCH --job-name=r12_sens_col
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --time=00:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/user/syedomer/r12_sensitivity/logs/%x_%j.out
source ~/spebt-venv/bin/activate
export PYTHONPATH=/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization:$PYTHONPATH
cd ~/r12_sensitivity
python colsums.py ~/r12_sensitivity/out
