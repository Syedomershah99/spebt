#!/bin/bash
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:58:00
#SBATCH --job-name=brain_tests
#SBATCH --output=/projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/tests_%j.out
#SBATCH --error=/projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/tests_%j.err
set -uo pipefail
source $HOME/spebt-venv/bin/activate
T=/projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct
echo "=== TEST 1: corners zeroed"
python $T/make_zeroed.py
sed "s|data_output_dir: .*|data_output_dir: \"$T/corners/data\"|" /projects/academic/rutaoyao/Omer/brain_maps_check/brain.yml > $T/corners/corners.yml
sed "s|fwhm_max_mm: 10$|fwhm_max_mm: 1.0e9|" $T/corners/corners.yml > $T/corners/corners_nofilter.yml
cd /projects/academic/rutaoyao/Omer/brain_maps_check/code/beam-analysis
for L in 0 1; do
  python arg_extract_beam_masks.py $L --config $T/corners/corners.yml > /dev/null
  python arg_extract_beam_properties.py $L --config $T/corners/corners.yml > /dev/null
done
python arg_design_maps.py --config $T/corners/corners.yml --layouts 0 1 --out $T/corners/corners_maps.hdf5 > /dev/null
python arg_design_maps.py --config $T/corners/corners_nofilter.yml --layouts 0 1 --out $T/corners/corners_maps_nofilter.hdf5 > /dev/null
python $T/compare_corners.py
echo "=== TEST 2: 64 mm line"
python $T/fwhm_long.py
