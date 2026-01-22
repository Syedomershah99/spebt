# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# import os

# def load_ppdfs_for_layout(ppdf_dir: str, layout_idx: int, n_xtals_to_load: int = -1):
#     """Loads PPDFs from an HDF5 file for a specific layout."""
#     ppdf_filename = os.path.join(ppdf_dir, f"position_{layout_idx:03d}_ppdfs.hdf5")
#     if not os.path.exists(ppdf_filename):
#         print(f"PPDF file {ppdf_filename} does not exist.")
#         return None
    
#     with h5py.File(ppdf_filename, "r") as f:
#         # Selects all crystals if n_xtals_to_load is -1, otherwise slices the array
#         ppdfs_data = f["ppdfs"][:n_xtals_to_load] if n_xtals_to_load != -1 else f["ppdfs"][:]
#     return ppdfs_data

# def create_and_plot_sensitivity_histogram(sensitivity_map: np.ndarray, output_dir: str, file_suffix: str, title_suffix: str):
#     """Generates and displays a histogram for a given sensitivity map."""
#     # --- 1. Prepare Data: Filter for values within the FOV ---
#     fov_values = sensitivity_map[sensitivity_map > 0].flatten()
#     if fov_values.size == 0:
#         print("Warning: The sensitivity map contains no non-zero values.")
#         return

#     # --- 2. Create and Plot the Histogram ---
#     plt.figure(figsize=(10, 6))
#     plt.hist(fov_values, bins=100, color='royalblue', alpha=0.75)
#     plt.title(f'Histogram of Sensitivity Values - {title_suffix}')
#     plt.xlabel('Sensitivity')
#     plt.ylabel('Number of Pixels (Frequency)')
#     plt.grid(True, linestyle='--', alpha=0.6)
    
#     mean_sensitivity = np.mean(fov_values)
#     plt.axvline(mean_sensitivity, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sensitivity:.4f}')
#     plt.legend()
#     plt.tight_layout()

#     # --- 3. Save the Histogram Plot ---
#     hist_filename = os.path.join(output_dir, f"sensitivity_histogram_{file_suffix}.png")
#     plt.savefig(hist_filename)
#     print(f"Saved sensitivity histogram to: {hist_filename}")
#     plt.close()


# if __name__ == "__main__":
#     # --- Configuration ---
#     ppdf_files_base_dir = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/"
#     num_layouts_to_load = 1
#     layout_indices = list(range(num_layouts_to_load))

#     FOV_PIXELS_X, FOV_PIXELS_Y = 200, 200
#     MM_PER_PIXEL_X, MM_PER_PIXEL_Y = 0.05, 0.05
#     num_crystals_to_sum = -1
#     # --- End Configuration ---

#     if not os.path.isdir(ppdf_files_base_dir):
#         print(f"Error: PPDF directory '{ppdf_files_base_dir}' not found.")
#         exit()

#     # --- Loop through layouts, load, and aggregate ---
#     print(f"Loading and aggregating PPDFs for layouts: {layout_indices}...")
#     aggregated_ppdfs = None
#     successful_loads = 0 # NEW: Counter for successfully loaded layouts

#     for idx in layout_indices:
#         print(f"  - Loading layout {idx:03d}...")
#         ppdfs_for_layout = load_ppdfs_for_layout(ppdf_files_base_dir, idx, num_crystals_to_sum)
        
#         if ppdfs_for_layout is None:
#             print(f"Warning: Could not load data for layout {idx}. Skipping.")
#             continue
            
#         if aggregated_ppdfs is None:
#             aggregated_ppdfs = ppdfs_for_layout.astype(np.float32)
#         else:
#             aggregated_ppdfs += ppdfs_for_layout
        
#         successful_loads += 1 # NEW: Increment the counter on a successful load

#     if aggregated_ppdfs is None:
#         print("Error: No PPDF data was successfully loaded. Exiting.")
#         exit()
    
#     print(f"\nAggregation complete. Successfully loaded {successful_loads} layouts.")

#     # --- Generate and NORMALIZE the sensitivity map ---
#     if successful_loads > 0:
#         # UPDATED: Divide by the number of successful loads to get the average
#         sensitivity_map_1d = np.sum(aggregated_ppdfs, axis=0) / successful_loads
#     else:
#         # This case is already handled by the exit above, but it's good practice
#         sensitivity_map_1d = np.zeros(FOV_PIXELS_X * FOV_PIXELS_Y)

#     sensitivity_map_2d = sensitivity_map_1d.reshape((FOV_PIXELS_Y, FOV_PIXELS_X))
            
#     # Determine plot extent
#     extent = [-(MM_PER_PIXEL_X * FOV_PIXELS_X / 2), (MM_PER_PIXEL_X * FOV_PIXELS_X / 2),
#               -(MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2), (MM_PER_PIXEL_Y * FOV_PIXELS_Y / 2)]

#     # --- Create descriptive names for aggregated output files and titles ---
#     layouts_str_fname = f"aggregated_{successful_loads}layouts_normalized"
#     layouts_str_title = f"Normalized Sensitivity Map ({successful_loads} Layouts)"

#     # Plot and Save the Aggregated and Normalized Sensitivity Map Image
#     plt.figure(figsize=(8, 7))
#     plt.imshow(sensitivity_map_2d, cmap='viridis', origin='lower', extent=extent)
#     # UPDATED: Changed colorbar label to reflect normalization
#     plt.colorbar(label='Average Sensitivity (Normalized)')
#     plt.title(layouts_str_title)
#     plt.xlabel('X (mm)'); plt.ylabel('Y (mm)')
#     plt.axhline(0, color='white', linestyle=':', lw=0.5); plt.axvline(0, color='white', linestyle=':', lw=0.5)
#     plt.tight_layout()
#     map_filename = os.path.join(ppdf_files_base_dir, f"sensitivity_map_{layouts_str_fname}.png")
#     plt.savefig(map_filename)
#     print(f"\nSaved normalized sensitivity map to: {map_filename}")
#     plt.close()
    
#     # NEW: Also create a histogram for the normalized data
#     create_and_plot_sensitivity_histogram(sensitivity_map_2d, ppdf_files_base_dir, layouts_str_fname, layouts_str_title)

#test_t4

#!/usr/bin/env python3
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

def load_ppdfs(ppdf_dir: str, filename: str, n_xtals_to_load: int = -1):
    path = os.path.join(ppdf_dir, filename)
    if not os.path.exists(path):
        print(f"[MISS] {path}")
        return None
    with h5py.File(path, "r") as f:
        data = f["ppdfs"][:n_xtals_to_load] if n_xtals_to_load != -1 else f["ppdfs"][:]
    return data.astype(np.float32)

def sensitivity_from_ppdfs(ppdfs: np.ndarray, fov_pixels: int):
    # sum over crystals -> (fov_pixels,)
    sens_1d = np.sum(ppdfs, axis=0)
    assert sens_1d.size == fov_pixels
    return sens_1d

if __name__ == "__main__":
    # ---------------- CONFIG ----------------
    base_dir = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/"
    layouts = [0, 1]
    poses = [0, 1, 2, 3]

    FOV_X, FOV_Y = 200, 200
    MM_PER_PX = 0.05
    fov_pixels = FOV_X * FOV_Y

    n_xtals_to_load = -1  # -1 = all crystals
    # ----------------------------------------

    t4_files = [f"position_{li:03d}_ppdfs_t4_{pi:02d}.hdf5" for li in layouts for pi in poses]

    sens_sum = None
    loaded = 0

    print("Loading files:")
    for fn in t4_files:
        ppdfs = load_ppdfs(base_dir, fn, n_xtals_to_load)
        if ppdfs is None:
            continue

        # OLD-STYLE: sum crystals for this dataset
        sens_1d = sensitivity_from_ppdfs(ppdfs, fov_pixels)

        if sens_sum is None:
            sens_sum = sens_1d
        else:
            sens_sum += sens_1d

        loaded += 1
        print(f"  ✓ {fn}  (loaded={loaded})")

    if loaded == 0:
        raise RuntimeError("No PPDF files loaded.")

    # OLD-STYLE: average over loaded datasets (layouts×poses)
    sens_mean_1d = sens_sum / float(loaded)
    sens_mean_2d = sens_mean_1d.reshape(FOV_Y, FOV_X)

    # ----------- RESOLUTION OF THE “>100%” ISSUE -----------
    # This is the key: make it explicitly RELATIVE sensitivity (0..1) or (% of max).
    eps = 1e-12
    sens_rel = sens_mean_2d / (sens_mean_2d.max() + eps)        # 0..1
    sens_rel_pct = 100.0 * sens_rel                              # 0..100 %
    # -------------------------------------------------------

    extent = [
        -(MM_PER_PX * FOV_X / 2.0), +(MM_PER_PX * FOV_X / 2.0),
        -(MM_PER_PX * FOV_Y / 2.0), +(MM_PER_PX * FOV_Y / 2.0),
    ]

    # Save raw mean sensitivity (arbitrary units)
    plt.figure(figsize=(8, 7))
    plt.imshow(sens_mean_2d, cmap="viridis", origin="lower", extent=extent)
    plt.colorbar(label=f"Mean sensitivity (a.u.) over {loaded} datasets")
    plt.title(f"Sensitivity (mean over {len(layouts)} layouts × {len(poses)} poses)")
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.tight_layout()
    out_raw = os.path.join(base_dir, f"sensitivity_mean_raw_{loaded}datasets.png")
    plt.savefig(out_raw, dpi=300); plt.close()
    print(f"\nSaved RAW mean sensitivity (a.u.): {out_raw}")

    # Save relative % sensitivity (fixed scale)
    plt.figure(figsize=(8, 7))
    plt.imshow(sens_rel_pct, cmap="viridis", origin="lower", extent=extent, vmin=0, vmax=100)
    plt.colorbar(label="Relative sensitivity (% of max)")
    plt.title(f"Relative Sensitivity (% of max) (mean over {loaded} datasets)")
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.tight_layout()
    out_pct = os.path.join(base_dir, f"sensitivity_mean_relative_pct_{loaded}datasets.png")
    plt.savefig(out_pct, dpi=300); plt.close()
    print(f"Saved RELATIVE sensitivity (%): {out_pct}")

    # Histogram for relative (%) inside FOV (nonzero)
    vals = sens_rel_pct[sens_mean_2d > 0].flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(vals, bins=100, alpha=0.75)
    plt.title("Histogram of Relative Sensitivity (% of max) inside FOV")
    plt.xlabel("Relative sensitivity (%)")
    plt.ylabel("Pixel count")
    plt.grid(True, linestyle="--", alpha=0.5)
    out_hist = os.path.join(base_dir, f"sensitivity_hist_relative_pct_{loaded}datasets.png")
    plt.tight_layout()
    plt.savefig(out_hist, dpi=300); plt.close()
    print(f"Saved histogram: {out_hist}")