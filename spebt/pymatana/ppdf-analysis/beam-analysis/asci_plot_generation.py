import os, h5py, torch, matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -------------------------------------------------------------------
INPUT_DIR  = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/"
PLOT_DIR   = "/vscratch/grp-rutaoyao/Omer/spebt/plots/sai_10mm/"
LAYOUT_SEQ = range(2)            # range(24) after you have all histograms
N_BINS     = 360
FOV_SIDE   = 10                  # mm  (±16 mm)
# -------------------------------------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

asci_hist = torch.zeros(200*200, N_BINS, dtype=torch.int32)

for idx in LAYOUT_SEQ:
    with h5py.File(os.path.join(INPUT_DIR,
                                f"asci_histogram_{idx:02d}_t4_agg.hdf5"), "r") as f:
        asci_hist += torch.from_numpy(f["asci_histogram"][...])

asci_map = torch.count_nonzero(asci_hist, dim=1) / N_BINS   # 0‒1

# ---- plot -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8,7), layout="constrained")

im = ax.imshow(asci_map.view(200,200).T,
               extent=(-FOV_SIDE/2, FOV_SIDE/2, -FOV_SIDE/2, FOV_SIDE/2),
               origin="lower", cmap="viridis")
cbar = fig.colorbar(im, ax=ax, label="ASCI")
cbar.formatter = PercentFormatter(xmax=1.0, decimals=1); cbar.update_ticks()

ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
ax.set_title(f"max {asci_map.max():.2%}, min {asci_map.min():.2%}")

fig.savefig(os.path.join(PLOT_DIR, "asci_map.png"), dpi=300)
plt.close(fig)

# asci t4

# import os, h5py, torch, matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter

# # -------------------------------------------------------------------
# INPUT_DIR  = "/vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm/"
# PLOT_DIR   = "/vscratch/grp-rutaoyao/Omer/spebt/plots/sai_10mm/"
# LAYOUT_SEQ = range(2)            # range(24) after you have all histograms
# N_BINS     = 360
# FOV_SIDE   = 10                  # mm (=> extent is ±5 mm)
# IMG_NX, IMG_NY = 200, 200
# # -------------------------------------------------------------------

# os.makedirs(PLOT_DIR, exist_ok=True)

# asci_hist = torch.zeros(IMG_NX * IMG_NY, N_BINS, dtype=torch.int32)

# successful = 0
# for idx in LAYOUT_SEQ:
#     # UPDATED: use the aggregated T4 histogram file
#     h5_path = os.path.join(INPUT_DIR, f"asci_histogram_{idx:02d}_t4_agg.hdf5")

#     if not os.path.exists(h5_path):
#         print(f"[WARN] Missing: {h5_path} (skipping)")
#         continue

#     with h5py.File(h5_path, "r") as f:
#         asci_hist += torch.from_numpy(f["asci_histogram"][...])
#     successful += 1

# if successful == 0:
#     raise RuntimeError("No T4 aggregated ASCI histogram files were found. Nothing to plot.")

# # fraction of angle bins that are nonzero per pixel (0..1)
# asci_map = torch.count_nonzero(asci_hist, dim=1) / float(N_BINS)

# # ---- plot -----------------------------------------------------------
# fig, ax = plt.subplots(figsize=(8,7), layout="constrained")

# im = ax.imshow(
#     asci_map.view(IMG_NX, IMG_NY).T,
#     extent=(-FOV_SIDE/2, FOV_SIDE/2, -FOV_SIDE/2, FOV_SIDE/2),
#     origin="lower",
#     cmap="viridis"
# )

# cbar = fig.colorbar(im, ax=ax, label="ASCI (fraction of angle bins)")
# cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
# cbar.update_ticks()

# ax.set_xlabel("X (mm)")
# ax.set_ylabel("Y (mm)")
# ax.set_title(
#     f"T4-aggregated ASCI map | layouts loaded={successful} | "
#     f"max={asci_map.max():.2%}, min={asci_map.min():.2%}"
# )

# out_path = os.path.join(PLOT_DIR, "asci_map_t4_agg.png")
# fig.savefig(out_path, dpi=300)
# plt.close(fig)

# print(f"[DONE] Saved: {out_path}")