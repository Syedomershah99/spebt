#!/bin/bash
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:58:00
#SBATCH --job-name=brain_maps
#SBATCH --output=/projects/academic/rutaoyao/Omer/brain_maps_check/logs/brain_maps_%j.out
#SBATCH --error=/projects/academic/rutaoyao/Omer/brain_maps_check/logs/brain_maps_%j.err
set -euo pipefail
source $HOME/spebt-venv/bin/activate
cd /projects/academic/rutaoyao/Omer/brain_maps_check/code/beam-analysis
SECONDS=0
for L in 0 1; do
  python arg_extract_beam_masks.py $L --config /projects/academic/rutaoyao/Omer/brain_maps_check/brain.yml
  python arg_extract_beam_properties.py $L --config /projects/academic/rutaoyao/Omer/brain_maps_check/brain.yml
done
echo extraction took ${SECONDS}s
python arg_design_maps.py --config /projects/academic/rutaoyao/Omer/brain_maps_check/brain.yml --layouts 0 1 --out /projects/academic/rutaoyao/Omer/brain_maps_check/out/brain_maps.hdf5
python arg_design_maps.py --config /projects/academic/rutaoyao/Omer/brain_maps_check/brain_nofilter.yml --layouts 0 1 --out /projects/academic/rutaoyao/Omer/brain_maps_check/out/brain_maps_nofilter.hdf5
python plot_design_maps.py /projects/academic/rutaoyao/Omer/brain_maps_check/out/brain_maps.hdf5 --labels "Brain template" --out /projects/academic/rutaoyao/Omer/brain_maps_check/out/brain_maps.png
echo total ${SECONDS}s
