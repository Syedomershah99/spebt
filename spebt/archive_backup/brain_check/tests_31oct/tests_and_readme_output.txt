##### TESTS 1-2
=== TEST 1: corners zeroed
layout 0: zeroed 8572 pixels holding 17.25% of the PPDF
layout 1: zeroed 8572 pixels holding 17.30% of the PPDF
window 10 mm wfwhm        FOV mean 23.11 -> 23.38 (+1.18%); max abs change 1.08
window 10 mm asci         FOV mean 10.04 -> 36.69 (+265.35%); max abs change 46.4
window 10 mm sensitivity  FOV mean 0.3751 -> 0.3751 (+0.00%); max abs change 0
window 10 mm ppds         FOV mean 0.8936 -> 0.8865 (-0.79%); max abs change 0.184
no window    asci         FOV mean 50.45 -> 74.08 (+46.84%); max abs change 43.1
layout 0 baseline: strips with pixels 5312, at 64 mm 62, median FWHM 5.08 mm, crystals without strips 20
layout 0 zeroed  : strips with pixels 4338, at 64 mm 6, median FWHM 6.78 mm, crystals without strips 20
layout 1 baseline: strips with pixels 5281, at 64 mm 52, median FWHM 5.09 mm, crystals without strips 20
layout 1 zeroed  : strips with pixels 4338, at 64 mm 2, median FWHM 6.76 mm, crystals without strips 20
=== TEST 2: 64 mm line
layout 0: 5312 strips with pixels; 64 mm line vs stored FWHM: max abs diff 0 mm, 100.00% within 0.001 mm
  stored 64 mm strips: 62; on the 300 mm line: median 300.0 mm, 5/95%: [300. 300.]; still at 300: 62
  other strips whose FWHM changes by > 0.01 mm: 439 of 5250; largest change 0.61 mm
  share of strip PPDF in the stored 64 mm strips: 0.018%
layout 1: 5281 strips with pixels; 64 mm line vs stored FWHM: max abs diff 0 mm, 100.00% within 0.001 mm
  stored 64 mm strips: 52; on the 300 mm line: median 300.0 mm, 5/95%: [300. 300.]; still at 300: 52
  other strips whose FWHM changes by > 0.01 mm: 492 of 5229; largest change 0.68 mm
  share of strip PPDF in the stored 64 mm strips: 0.021%
maps with 300 mm widths, in the FOV: wfwhm mean 23.12 (64 mm line: 23.11); max abs change 0.0541
maps with 300 mm widths, in the FOV: asci mean 10.02 (64 mm line: 10.04); max abs change 0.278
maps with 300 mm widths, in the FOV: ppds mean 0.8732 (64 mm line: 0.8936); max abs change 0.0296
--- err
##### TEST 3
d338069 Add design maps: wFWHM, ASCI, sensitivity, PPDS
=== a student's edits to configs/base_config.yml, from the README
maps: {'ppdf_files': '/vscratch/grp-rutaoyao/Tridev/brain_spect/data/brain_spect_ppdfs_layout{layout:03d}_all.hdf5', 'properties_file': 'beams_properties_configuration_{layout:02d}.hdf5', 'masks_file': 'beams_masks_configuration_{layout:02d}.hdf5'}
=== README step 1 as written (layout 0)
--- Starting Beam Mask Extraction for Layout Index: 0 ---
Error: PPDF file /projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme/out/position_000_ppdfs.hdf5 not found.
  beam_properties_dataset = out_hdf5_file.create_dataset(
Error: PPDF file /projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme/out/position_000_ppdfs.hdf5 not found.
beams_masks_configuration_00.hdf5
beams_properties_configuration_00.hdf5
=== README step 2 as written
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5f.pyx", line 106, in h5py.h5f.open
FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme/out/beams_properties_configuration_01.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
=== rerun of step 1 after linking the PPDFs as position_NNN_ppdfs.hdf5
--- Finished Mask Extraction for Layout Index: 0 ---
    raise FileExistsError(
FileExistsError: Output file beams_properties_configuration_00.hdf5 already exists in /projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme/out.
=== after removing the stale properties file; layout 1 too, which the README example omits
--- Beam properties saved for layout 0 ---
--- Finished Mask Extraction for Layout Index: 1 ---
--- Beam properties saved for layout 1 ---
=== README step 2 again
  layout 1: 1 PPDF file(s), 5516 strips
  wfwhm: mean 22.8797 mm, max 26.5255
  asci: mean 9.10538 %, max 19.4444
  sensitivity: mean 0.356215 , max 1.11931
  ppds: mean 0.825952 mm^-2, max 0.98371
--- Design maps saved to /projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme/brain.hdf5 ---
Saved /projects/academic/rutaoyao/Omer/brain_maps_check/tests_31oct/readme/brain.png
ok
wfwhm identical to the first run: True
asci identical to the first run: True
sensitivity identical to the first run: True
ppds identical to the first run: True
--- err

[exited with code 0]
