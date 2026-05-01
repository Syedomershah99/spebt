#!/usr/bin/env python3
"""
Reconstruction comparison: Baseline vs BO-Optimized config.

Runs the full pipeline for a given config:
  1. Generate file list (dataset_flist.csv) from PPDF HDF5 files
  2. Forward project phantom through system matrices
  3. ML-EM reconstruction with Gaussian post-filter
  4. Compute CNR
  5. Save results and comparison plots

Usage:
  # Baseline (uses existing PPDFs in data/sai_10mm)
  python run_recon_comparison.py --config baseline \
    --ppdf_dir /vscratch/grp-rutaoyao/Omer/spebt/data/sai_10mm \
    --output_dir recon_results/baseline

  # BO-optimized (uses PPDFs from BO config work dir)
  python run_recon_comparison.py --config bo_optimized \
    --ppdf_dir /vscratch/grp-rutaoyao/Omer/spebt/spebt/optimization/results/bo_0013_ap0.5300_nap232 \
    --output_dir recon_results/bo_optimized

  # Compare both
  python run_recon_comparison.py --compare \
    --baseline_dir recon_results/baseline \
    --bo_dir recon_results/bo_optimized \
    --output_dir recon_results/comparison
"""
import os
import sys
import glob
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =====================
# Constants
# =====================
IMG_DIM = 200
SFOV = IMG_DIM * IMG_DIM
SPROJ = 3360
N_ITERATIONS = 150
SAVE_EVERY = 5
CONVERGENCE_TOL = 1e-4
GAUSS_FWHM_MM = 0.16
MM_PER_PX = 0.05


# =====================
# Helpers (from mlem_torch_gpf_nonmpi.py)
# =====================
def gaussian_kernel_1d(sigma_px, device, dtype=torch.float32):
    if sigma_px <= 0:
        return torch.tensor([1.0], device=device, dtype=dtype)
    radius = int(np.ceil(3.0 * sigma_px))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma_px * sigma_px))
    return k / torch.sum(k)


def gaussian_filter_2d(img_2d, fwhm_mm, mm_per_px):
    device = img_2d.device
    dtype = img_2d.dtype
    sigma_px = (fwhm_mm / 2.355) / mm_per_px
    k1d = gaussian_kernel_1d(sigma_px, device=device, dtype=dtype)
    x = img_2d.unsqueeze(0).unsqueeze(0)
    kx = k1d.view(1, 1, 1, -1)
    pad_x = (kx.shape[-1] // 2, kx.shape[-1] // 2, 0, 0)
    x = F.pad(x, pad_x, mode="replicate")
    x = F.conv2d(x, kx)
    ky = k1d.view(1, 1, -1, 1)
    pad_y = (0, 0, ky.shape[-2] // 2, ky.shape[-2] // 2)
    x = F.pad(x, pad_y, mode="replicate")
    x = F.conv2d(x, ky)
    return x.squeeze(0).squeeze(0)


# =====================
# Step 1: Generate file list
# =====================
def generate_flist(ppdf_dir, output_dir):
    """Find all PPDF HDF5 files and write file list."""
    # Pattern: position_NNN_ppdfs_t8_PP.hdf5
    pattern = os.path.join(ppdf_dir, "position_*_ppdfs_t8_*.hdf5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PPDF files found matching {pattern}")

    flist_path = os.path.join(output_dir, "dataset_flist.csv")
    with open(flist_path, "w") as f:
        for fname in files:
            f.write(fname + "\n")

    print(f"[Step 1] Generated file list: {len(files)} files -> {flist_path}")
    return flist_path, files


# =====================
# Step 2: Forward projection
# =====================
def forward_project(flist, phantom_path, output_dir, device):
    """Project phantom through system matrices."""
    phantom_data = torch.load(phantom_path, map_location="cpu", weights_only=False)
    phantom_tensor = phantom_data["Phantom tensor"]

    h, w = phantom_tensor.shape
    pad_h = (IMG_DIM - h) // 2
    pad_w = (IMG_DIM - w) // 2
    phantom_padded = F.pad(phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
    phantom_flat = phantom_padded.view(-1).to(device)

    print(f"[Step 2] Phantom: {h}x{w} -> padded {IMG_DIM}x{IMG_DIM}")

    all_projs = []
    for i, fname in enumerate(flist):
        with h5py.File(fname, "r") as h5f:
            m = torch.from_numpy(h5f["ppdfs"][:]).to(device=device, dtype=torch.float32)
            m = m.view(1, SPROJ, SFOV)
            proj = torch.matmul(m, phantom_flat)
            all_projs.append(proj.cpu())
        if (i + 1) % 4 == 0:
            print(f"  Projected {i + 1}/{len(flist)} files")

    projs = torch.cat(all_projs, dim=0)
    projs_path = os.path.join(output_dir, "projections_T8.npy")
    np.save(projs_path, projs.numpy())
    print(f"[Step 2] Projections shape: {projs.shape} -> {projs_path}")
    return projs_path


# =====================
# Step 3: ML-EM Reconstruction
# =====================
def run_mlem(flist, projs_path, output_dir, device):
    """Run ML-EM with Gaussian post-filter."""
    pdata = torch.from_numpy(np.load(projs_path)).to(device=device, dtype=torch.float32)
    estimate = torch.ones(SFOV, device=device, dtype=torch.float32)

    estimates_history = []
    estimates_gauss = []
    diffs = []

    print(f"[Step 3] Starting ML-EM ({N_ITERATIONS} iterations)...")
    t_start = time.time()

    for it in range(N_ITERATIONS):
        t0 = time.time()
        estimate_prev = estimate.clone()
        back_projection = torch.zeros(SFOV, device=device, dtype=torch.float32)
        sensitivity_map = torch.zeros(SFOV, device=device, dtype=torch.float32)

        for i, fname in enumerate(flist):
            with h5py.File(fname, "r") as h5f:
                m = torch.from_numpy(h5f["ppdfs"][:]).to(device=device, dtype=torch.float32)
                m = m.view(1, SPROJ, SFOV)

            p = pdata[i].view(1, SPROJ)
            y = torch.clamp(torch.matmul(m, estimate), min=1e-12)
            r = p / y
            back_projection += torch.matmul(m.transpose(1, 2), r.unsqueeze(-1)).squeeze()
            sensitivity_map += torch.sum(m, dim=1).squeeze()

        sensitivity_map = torch.clamp(sensitivity_map, min=1e-12)
        estimate = estimate * (back_projection / sensitivity_map)

        diff = torch.norm(estimate - estimate_prev) / torch.norm(estimate_prev)
        diffs.append(float(diff.item()))

        if it % SAVE_EVERY == 0:
            est2d = estimate.view(IMG_DIM, IMG_DIM).detach().cpu()
            estimates_history.append(est2d.numpy())
            est2d_f = gaussian_filter_2d(est2d.to(device), fwhm_mm=GAUSS_FWHM_MM, mm_per_px=MM_PER_PX)
            estimates_gauss.append(est2d_f.detach().cpu().numpy())

        dt = time.time() - t0
        if (it + 1) % 10 == 0:
            print(f"  Iter {it + 1}/{N_ITERATIONS}  diff={diff:.2e}  ({dt:.1f}s)")

        if diff < CONVERGENCE_TOL:
            print(f"  Converged at iter {it + 1}")
            break

    total_time = time.time() - t_start
    print(f"  Total reconstruction time: {total_time / 60:.1f} min")

    # Final with post-filter
    final = estimate.view(IMG_DIM, IMG_DIM).detach()
    final_gauss = gaussian_filter_2d(final, fwhm_mm=GAUSS_FWHM_MM, mm_per_px=MM_PER_PX)

    recon_path = os.path.join(output_dir, "recon_mlem_T8_gauss.npz")
    np.savez_compressed(
        recon_path,
        estimates=np.array(estimates_history),
        estimates_gauss=np.array(estimates_gauss),
        final=final.cpu().numpy(),
        final_gauss=final_gauss.cpu().numpy(),
        diffs=np.array(diffs),
    )
    print(f"[Step 3] Saved reconstruction: {recon_path}")
    return recon_path


# =====================
# Step 4: CNR Computation
# =====================
def compute_cnr(recon_path, phantom_path, output_dir, use_gauss=True):
    """Compute CNR per rod sector."""
    nz = np.load(recon_path)
    if use_gauss and "final_gauss" in nz:
        recon = torch.from_numpy(nz["final_gauss"])
    else:
        recon = torch.from_numpy(nz["final"])
    H, W = recon.shape

    phantom_data = torch.load(phantom_path, map_location="cpu", weights_only=False)
    meta = phantom_data["Metadata"]
    mm_per_px = meta["mm per pixel"][0]

    # Rod radii from phantom metadata
    rod_radii_mm = meta["rods radii in mm"]

    # Create background mask (circular FOV, exclude rods)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cx, cy = (H - 1) / 2.0, (W - 1) / 2.0
    rr2 = (yy - cx) ** 2 + (xx - cy) ** 2

    # Background: annular region (inner 20% to outer 90% of FOV radius)
    fov_radius_px = H / 2.0
    bg_inner = fov_radius_px * 0.2
    bg_outer = fov_radius_px * 0.9
    bg_mask = (rr2 >= bg_inner ** 2) & (rr2 <= bg_outer ** 2)

    # Exclude regions where phantom has hot rods
    phantom_tensor = phantom_data["Phantom tensor"]
    h, w = phantom_tensor.shape
    pad_h = (IMG_DIM - h) // 2
    pad_w = (IMG_DIM - w) // 2
    phantom_padded = F.pad(phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0)
    hot_mask = phantom_padded > 0.1
    bg_mask &= ~hot_mask

    bg_vals = recon[bg_mask]
    mu_bg = bg_vals.mean()
    sd_bg = bg_vals.std(unbiased=False) + 1e-12

    # Compute CNR per rod sector using phantom as mask
    # Group rods by radius (6 sectors, each with different rod radius)
    print(f"\n[Step 4] CNR Computation")
    print(f"  Background pixels: {int(bg_mask.sum())}")
    print(f"  Background mean: {mu_bg:.6f}")
    print(f"  Background std:  {sd_bg:.6f}")

    # Use thresholded phantom to identify rod regions per sector
    # For each rod radius, create an approximate mask based on the phantom
    # Simple approach: threshold the reconstruction where phantom is hot
    overall_hot_vals = recon[hot_mask]
    overall_cnr = float((overall_hot_vals.mean() - mu_bg) / sd_bg)

    print(f"\n  Overall CNR (all rods vs background): {overall_cnr:.2f}")
    print(f"  Hot region mean:  {overall_hot_vals.mean():.6f}")
    print(f"  Hot region pixels: {int(hot_mask.sum())}")

    # Per-sector CNR (approximate: divide phantom into angular sectors)
    sector_cnrs = []
    angles_deg = [30, 90, 150, 210, 270, 330]  # sector centers
    for i, (angle, radius_mm) in enumerate(zip(angles_deg, rod_radii_mm)):
        # Angular mask for this sector (±30 degrees)
        angle_rad = np.deg2rad(angle)
        px_angles = torch.atan2(yy - cx, xx - cy)
        angle_diff = torch.abs(px_angles - angle_rad)
        angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)
        sector_mask = (angle_diff < np.deg2rad(30)) & hot_mask

        if sector_mask.sum() < 5:
            sector_cnrs.append(float('nan'))
            continue

        hot_vals = recon[sector_mask]
        cnr = float((hot_vals.mean() - mu_bg) / sd_bg)
        sector_cnrs.append(cnr)
        print(f"  Sector {i} (r={radius_mm:.3f}mm, {angle}°): CNR={cnr:.2f}  ({int(sector_mask.sum())} px)")

    # Save results
    results = {
        "overall_cnr": overall_cnr,
        "sector_cnrs": sector_cnrs,
        "rod_radii_mm": rod_radii_mm,
        "bg_mean": float(mu_bg),
        "bg_std": float(sd_bg),
        "hot_mean": float(overall_hot_vals.mean()),
    }
    results_path = os.path.join(output_dir, "cnr_results.npz")
    np.savez(results_path, **{k: np.array(v) for k, v in results.items()})
    print(f"\n  Saved CNR results: {results_path}")
    return results


# =====================
# Step 5: Comparison Plot
# =====================
def plot_comparison(baseline_dir, bo_dir, output_dir):
    """Side-by-side comparison of baseline vs BO-optimized reconstruction."""
    os.makedirs(output_dir, exist_ok=True)

    # Load reconstructions
    base_nz = np.load(os.path.join(baseline_dir, "recon_mlem_T8_gauss.npz"))
    bo_nz = np.load(os.path.join(bo_dir, "recon_mlem_T8_gauss.npz"))

    base_img = base_nz["final_gauss"]
    bo_img = bo_nz["final_gauss"]

    # Load CNR results
    base_cnr = np.load(os.path.join(baseline_dir, "cnr_results.npz"), allow_pickle=True)
    bo_cnr = np.load(os.path.join(bo_dir, "cnr_results.npz"), allow_pickle=True)

    vmax = max(base_img.max(), bo_img.max())
    extent = [-5, 5, -5, 5]  # 10mm FOV

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.8])

    # Baseline
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(base_img.T, cmap="hot", extent=extent, origin="lower", vmin=0, vmax=vmax)
    ax1.set_title(f"Baseline (d=0.4, n=180)\nCNR={float(base_cnr['overall_cnr']):.2f}", fontsize=12)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")

    # BO-optimized
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(bo_img.T, cmap="hot", extent=extent, origin="lower", vmin=0, vmax=vmax)
    ax2.set_title(f"BO-Optimized (d=0.53, n=232)\nCNR={float(bo_cnr['overall_cnr']):.2f}", fontsize=12)
    ax2.set_xlabel("X (mm)")

    # CNR bar chart
    ax3 = fig.add_subplot(gs[2])
    rod_radii = base_cnr["rod_radii_mm"]
    x = np.arange(len(rod_radii))
    width = 0.35
    base_sector = base_cnr["sector_cnrs"]
    bo_sector = bo_cnr["sector_cnrs"]

    ax3.barh(x - width/2, base_sector, width, label="Baseline", color="steelblue")
    ax3.barh(x + width/2, bo_sector, width, label="BO-Optimized", color="coral")
    ax3.set_yticks(x)
    ax3.set_yticklabels([f"{r:.3f}mm" for r in rod_radii])
    ax3.set_xlabel("CNR")
    ax3.set_title("CNR per Rod Sector")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "recon_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Compare] Saved comparison plot: {plot_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline  CNR: {float(base_cnr['overall_cnr']):.2f}")
    print(f"  BO-Opt    CNR: {float(bo_cnr['overall_cnr']):.2f}")
    cnr_change = (float(bo_cnr['overall_cnr']) - float(base_cnr['overall_cnr'])) / float(base_cnr['overall_cnr']) * 100
    print(f"  Change:        {cnr_change:+.1f}%")
    print(f"\n  If CNR improved -> JI is a valid proxy metric")
    print(f"  If CNR didn't improve -> 83% JI gain is reward hacking")
    print(f"{'='*60}")


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser(description="Reconstruction Comparison Pipeline")
    parser.add_argument("--config", type=str, choices=["baseline", "bo_optimized"],
                        help="Which config to reconstruct")
    parser.add_argument("--ppdf_dir", type=str,
                        help="Directory containing PPDF HDF5 files")
    parser.add_argument("--phantom_path", type=str,
                        default=None,
                        help="Path to hot rod phantom .pt file")
    parser.add_argument("--output_dir", type=str, default="recon_results",
                        help="Output directory for results")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison mode (requires --baseline_dir and --bo_dir)")
    parser.add_argument("--baseline_dir", type=str,
                        help="Baseline results directory (for comparison)")
    parser.add_argument("--bo_dir", type=str,
                        help="BO-optimized results directory (for comparison)")
    args = parser.parse_args()

    if args.compare:
        if not args.baseline_dir or not args.bo_dir:
            parser.error("--compare requires --baseline_dir and --bo_dir")
        plot_comparison(args.baseline_dir, args.bo_dir, args.output_dir)
        return

    if not args.ppdf_dir:
        parser.error("--ppdf_dir is required when not in --compare mode")

    # Find phantom
    phantom_path = args.phantom_path
    if phantom_path is None:
        # Try common locations
        for candidate in [
            os.path.join(args.ppdf_dir, "hot_rods_phantom_10.0_mm_x_10.0_mm.pt"),
            "/vscratch/grp-rutaoyao/Omer/spebt/spebt/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt",
        ]:
            if os.path.exists(candidate):
                phantom_path = candidate
                break
        if phantom_path is None:
            parser.error("Could not find phantom .pt file. Specify --phantom_path")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"PPDF dir: {args.ppdf_dir}")
    print(f"Phantom: {phantom_path}")
    print(f"Output: {args.output_dir}")
    print()

    # Step 1: Generate file list
    flist_path, flist = generate_flist(args.ppdf_dir, args.output_dir)

    # Step 2: Forward projection
    projs_path = forward_project(flist, phantom_path, args.output_dir, device)

    # Step 3: ML-EM reconstruction
    recon_path = run_mlem(flist, projs_path, args.output_dir, device)

    # Step 4: CNR
    compute_cnr(recon_path, phantom_path, args.output_dir)

    print(f"\nDone! Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
