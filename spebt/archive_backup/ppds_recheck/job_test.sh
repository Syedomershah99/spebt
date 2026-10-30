#!/bin/bash
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=ppds_test
#SBATCH --output=ppds_test_%j.out
#SBATCH --error=ppds_test_%j.err
set -euo pipefail
source $HOME/spebt-venv/bin/activate
cd $HOME/r11_ppds
export PYTHONPATH=/projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization:${PYTHONPATH:-}
SECONDS=0
python recheck_ppds_radial.py designs.csv /projects/academic/rutaoyao/Omer/spebt/spebt/spebt/optimization test.csv --task 0 --ntasks 158
echo took ${SECONDS}s
