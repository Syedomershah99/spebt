#!/bin/bash
#SBATCH --partition=debug
#SBATCH --qos=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --job-name=readme_walk
#SBATCH --output=/projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme_%j.out
#SBATCH --error=/projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme_%j.err
set -u
source $HOME/spebt-venv/bin/activate
T=/projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme
TD=/vscratch/grp-rutaoyao/Tridev/brain_spect
rm -rf $T && mkdir -p $T/out && cd $T
git clone -q -b design-maps https://github.com/Syedomershah99/pymatana.git && cd pymatana/beam-analysis
git log --oneline -1
echo "=== a student's edits to configs/base_config.yml, from the README"
python - <<PY
import yaml
p = "configs/base_config.yml"; c = yaml.safe_load(open(p))
c["paths"]["scanner_layouts_dir"] = "$TD/geometry"
c["paths"]["scanner_layouts_filename"] = "brain_spect_layouts_e90446e80e2b0ae1da75fd55d98c8d37_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
c["paths"]["data_output_dir"] = "$T/out"
c["fov"]["n_pixels"] = [200, 200]; c["fov"]["mm_per_pixel"] = [1.0, 1.0]
c["analysis"]["fwhm_max_mm"] = 10
c["maps"]["ppdf_files"] = "$TD/data/brain_spect_ppdfs_layout{layout:03d}_all.hdf5"
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
print("maps:", c["maps"])
PY
echo "=== README step 1 as written (layout 0)"
python arg_extract_beam_masks.py 0 --config configs/base_config.yml 2>&1 | tail -2
python arg_extract_beam_properties.py 0 --config configs/base_config.yml 2>&1 | tail -2
ls $T/out
echo "=== README step 2 as written"
python arg_design_maps.py --config configs/base_config.yml --layouts 0 1 --out $T/brain.hdf5 2>&1 | tail -3
echo "=== rerun of step 1 after linking the PPDFs as position_NNN_ppdfs.hdf5"
ln -s $TD/data/brain_spect_ppdfs_layout000_all.hdf5 $T/out/position_000_ppdfs.hdf5
ln -s $TD/data/brain_spect_ppdfs_layout001_all.hdf5 $T/out/position_001_ppdfs.hdf5
python arg_extract_beam_masks.py 0 --config configs/base_config.yml 2>&1 | tail -1
python arg_extract_beam_properties.py 0 --config configs/base_config.yml 2>&1 | tail -2
echo "=== after removing the stale properties file; layout 1 too, which the README example omits"
rm -f $T/out/beams_properties_configuration_00.hdf5
python arg_extract_beam_properties.py 0 --config configs/base_config.yml 2>&1 | tail -1
python arg_extract_beam_masks.py 1 --config configs/base_config.yml 2>&1 | tail -1
python arg_extract_beam_properties.py 1 --config configs/base_config.yml 2>&1 | tail -1
echo "=== README step 2 again"
python arg_design_maps.py --config configs/base_config.yml --layouts 0 1 --out $T/brain.hdf5 2>&1 | tail -6
python plot_design_maps.py $T/brain.hdf5 --labels "Brain template" --out $T/brain.png 2>&1 | tail -1
python tests/test_design_maps.py
python - <<PY
import h5py, numpy as np
with h5py.File("$T/brain.hdf5") as a, h5py.File("/projects/academic/rutaoyao/Omer/brain_maps_check/out/brain_maps.hdf5") as b:
    for k in ("wfwhm", "asci", "sensitivity", "ppds"):
        x, y = a[k][:], b[k][:]
        print(k, "identical to the first run:", np.array_equal(x, y, equal_nan=True))
PY
