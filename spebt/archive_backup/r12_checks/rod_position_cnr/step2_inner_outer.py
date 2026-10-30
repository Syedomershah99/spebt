"""Step 2: is the MOBO-optimized design's CNR gain confined to rods away from the centre?

Same reconstructions, same masks, same background and same CNR definition as the archived
sector-mean CNR (step1_reproduce.py reproduces it to 2e-6). The only change: within each of the
six rod-size sectors, the hot pixels are pooled separately for the rods whose centre lies nearer
the field centre than a cut T ("inner") and for the rest ("outer"):

  CNR_k(S) = (mean recon over the hot pixels of the rods of set S in sector k - mu_bg) / sd_bg
  CNR(S)   = mean over the six sectors of CNR_k(S)      (equal weight per rod size, as the objective)

mu_bg, sd_bg: the pipeline's background, 1.0 to 4.5 mm annulus minus hot pixels, ddof 0.
Cuts: 2.50, 2.55 and 2.60 mm (the ASCI crossover) and the median rod-centre distance.
Also: pooled (area-weighted) CNR, per rod size, per rod, a contrast/noise decomposition,
a regional-background variant, and ASCI over the inner and outer rod pixels.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from rods import C, IMG, MM, geometry, load_phantom, masks, rods  # noqa: E402

DESIGNS = [("benchmark", "tmi_reference_000", "bench"), ("MOBO", "nd2fine_960", "mobo")]
MAPS = "/Users/Omer/Desktop/RA/omer/spebt/archive_backup/design_maps/{}.npz"
SEEDS = range(5)

ph, radii = load_phantom()
hot, bg, sectors = masks(ph)
R, lab = rods(ph, radii)
dist = np.array([r["dist_mm"] for r in R])
sec_of = np.array([r["sector"] for r in R])
median = float(np.median(dist))
CUTS = [("2.50 mm", 2.50), ("2.55 mm", 2.55), ("2.60 mm", 2.60), ("2.65 mm", 2.65),
        (f"median {median:.3f} mm", median)]
ii, jj, rr2, ang = geometry()
rmm = np.sqrt(rr2) * MM

recon = {name: [np.load(os.path.join(W, "data", d, f"cnr_repeat_seed{s}", "recon_mlem_T8.npz"))["final"]
                .astype(np.float64) for s in SEEDS] for name, d, _ in DESIGNS}
asci = {name: np.load(MAPS.format(m))["asci_win"].reshape(IMG, IMG) for name, _, m in DESIGNS}

# ------------------------------------------------------------------ orientation (recon vs phantom)
x = recon["benchmark"][0]
print("orientation: mean recon over hot mask as-is %.2f, transposed %.2f, background %.2f"
      % (x[hot].mean(), x[hot.T].mean(), x[bg].mean()))


def bgstats(img, region=None):
    m = bg if region is None else bg & region
    v = img[m]
    return v.mean(), v.std() + 1e-12


def set_cnr(img, sel_rods, region=None):
    """Sector-mean and pooled CNR over the rods in sel_rods (boolean over R)."""
    mu, sd = bgstats(img, region)
    per = np.full(6, np.nan)
    allpix = np.zeros_like(hot)
    for k in range(6):
        pix = np.zeros_like(hot)
        for r, use in zip(R, sel_rods):
            if use and r["sector"] == k:
                pix |= r["mask"]
        allpix |= pix
        if pix.any():
            per[k] = (img[pix].mean() - mu) / sd
    pooled = (img[allpix].mean() - mu) / sd
    return per, np.nanmean(per), pooled, mu, sd


def msd(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1)


def ratio_se(ma, sa, mb, sb, n=5):
    g = ma / mb
    return g, g * np.sqrt((sa / ma) ** 2 / n + (sb / mb) ** 2 / n)


out = {"rods": [dict(sector=int(r["sector"]), radius_mm=r["radius_mm"], dist_mm=r["dist_mm"],
                     npix=r["npix"]) for r in R], "median_dist_mm": median, "cuts": {}}

# ------------------------------------------------------------------ check: all rods = archived value
for name, _, _ in DESIGNS:
    v = [set_cnr(img, np.ones(len(R), bool))[1] for img in recon[name]]
    print(f"check, all rods together: {name} {np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}")

# ------------------------------------------------------------------ inner / outer by cut
for label, T in CUTS:
    inner = dist < T
    n_in = [int((inner & (sec_of == k)).sum()) for k in range(6)]
    n_out = [int((~inner & (sec_of == k)).sum()) for k in range(6)]
    print(f"\n=== cut {label}: inner rods per size {n_in} (total {inner.sum()}), outer {n_out} "
          f"(total {(~inner).sum()})")
    res = {}
    for name, _, _ in DESIGNS:
        rows = {"in": [], "out": [], "in_pool": [], "out_pool": [], "in_sec": [], "out_sec": [],
                "in_loc": [], "out_loc": [], "mu": [], "sd": [], "hot_in": [], "hot_out": []}
        for img in recon[name]:
            pi, si, qi, mu, sd = set_cnr(img, inner)
            po, so, qo, _, _ = set_cnr(img, ~inner)
            rows["in"].append(si); rows["out"].append(so)
            rows["in_pool"].append(qi); rows["out_pool"].append(qo)
            rows["in_sec"].append(pi); rows["out_sec"].append(po)
            rows["mu"].append(mu); rows["sd"].append(sd)
            # regional background: inner rods against 1.0 mm <= r < T, outer against T <= r <= 4.5 mm
            rows["in_loc"].append(set_cnr(img, inner, region=rmm < T)[1])
            rows["out_loc"].append(set_cnr(img, ~inner, region=rmm >= T)[1])
            # contrast recovery, sector-mean: (hot/bg - 1) / (10/2 - 1)
            rows["hot_in"].append(np.mean([(img[np.logical_or.reduce([r["mask"] for r, u in zip(R, inner) if u and r["sector"] == k])].mean() / mu - 1) / 4 for k in range(6)]))
            rows["hot_out"].append(np.mean([(img[np.logical_or.reduce([r["mask"] for r, u in zip(R, ~inner) if u and r["sector"] == k])].mean() / mu - 1) / 4 for k in range(6)]))
        res[name] = {k: np.array(v) for k, v in rows.items()}

    b, m = res["benchmark"], res["MOBO"]
    entry = {}
    for reg in ("in", "out", "in_pool", "out_pool", "in_loc", "out_loc", "hot_in", "hot_out"):
        mb, sb = msd(b[reg]); mm_, sm = msd(m[reg])
        g, ge = ratio_se(mm_, sm, mb, sb)
        t = stats.ttest_ind(m[reg], b[reg], equal_var=False)
        entry[reg] = dict(bench=[mb, sb], mobo=[mm_, sm], gain=[g, ge], welch_p=float(t.pvalue),
                          bench_seeds=b[reg].tolist(), mobo_seeds=m[reg].tolist())
    # interaction: per seed, inner/outer ratio within a design (shared background), compared across designs
    rb, rm = b["in"] / b["out"], m["in"] / m["out"]
    ti = stats.ttest_ind(rm, rb, equal_var=False)
    entry["interaction"] = dict(bench_in_over_out=list(msd(rb)), mobo_in_over_out=list(msd(rm)),
                                gain_in_over_gain_out=float(rm.mean() / rb.mean()), welch_p=float(ti.pvalue))
    # per rod size
    per_size = []
    for k in range(6):
        d = {}
        for reg, key in (("inner", "in_sec"), ("outer", "out_sec")):
            vb, vm = b[key][:, k], m[key][:, k]
            mb, sb = msd(vb); mm_, sm = msd(vm)
            d[reg] = dict(bench=[mb, sb], mobo=[mm_, sm], gain=list(ratio_se(mm_, sm, mb, sb)))
        per_size.append(dict(radius_mm=float(radii[k]), n_inner=n_in[k], n_outer=n_out[k], **d))
    entry["per_size"] = per_size
    entry["n_inner"] = int(inner.sum()); entry["n_outer"] = int((~inner).sum())
    out["cuts"][label] = entry

    def line(tag, reg):
        e = entry[reg]
        return (f"  {tag:<34} bench {e['bench'][0]:.3f} +/- {e['bench'][1]:.3f}   MOBO {e['mobo'][0]:.3f} +/- "
                f"{e['mobo'][1]:.3f}   gain x{e['gain'][0]:.3f} +/- {e['gain'][1]:.3f} "
                f"({100 * (e['gain'][0] - 1):+.1f}%)  Welch p {e['welch_p']:.1e}")
    print(line("INNER rods, sector-mean CNR", "in"))
    print(line("OUTER rods, sector-mean CNR", "out"))
    print(line("inner, pooled (area-weighted)", "in_pool"))
    print(line("outer, pooled (area-weighted)", "out_pool"))
    print(line("inner, regional background", "in_loc"))
    print(line("outer, regional background", "out_loc"))
    print(line("inner, contrast recovery", "hot_in"))
    print(line("outer, contrast recovery", "hot_out"))
    for reg in ("in", "out"):
        print(f"    per realization, {reg:>3}: bench {np.round(entry[reg]['bench_seeds'], 3).tolist()}  "
              f"MOBO {np.round(entry[reg]['mobo_seeds'], 3).tolist()}")
    print(f"    inner rod centres {dist[inner].min():.3f}-{dist[inner].max():.3f} mm, "
          f"outer {dist[~inner].min():.3f}-{dist[~inner].max():.3f} mm")
    it = entry["interaction"]
    print(f"  inner/outer CNR ratio: bench {it['bench_in_over_out'][0]:.3f} +/- {it['bench_in_over_out'][1]:.3f}, "
          f"MOBO {it['mobo_in_over_out'][0]:.3f} +/- {it['mobo_in_over_out'][1]:.3f}; "
          f"gain_inner / gain_outer {it['gain_in_over_gain_out']:.3f}, Welch p {it['welch_p']:.2f}")
    print("  per rod size (radius: n in/out | inner bench, MOBO, gain | outer bench, MOBO, gain):")
    for d in per_size:
        a, o = d["inner"], d["outer"]
        print(f"    {d['radius_mm']:.3f} mm: {d['n_inner']:2d}/{d['n_outer']:<2d} | "
              f"{a['bench'][0]:.2f}+/-{a['bench'][1]:.2f}  {a['mobo'][0]:.2f}+/-{a['mobo'][1]:.2f}  "
              f"x{a['gain'][0]:.2f}+/-{a['gain'][1]:.2f} | "
              f"{o['bench'][0]:.2f}+/-{o['bench'][1]:.2f}  {o['mobo'][0]:.2f}+/-{o['mobo'][1]:.2f}  "
              f"x{o['gain'][0]:.2f}+/-{o['gain'][1]:.2f}")

# ------------------------------------------------------------------ background noise and mean
for name, _, _ in DESIGNS:
    mus = [bgstats(img)[0] for img in recon[name]]; sds = [bgstats(img)[1] for img in recon[name]]
    print(f"\nbackground {name}: mean {np.mean(mus):.3f}, SD {np.mean(sds):.3f}, CV {np.mean(sds) / np.mean(mus):.4f}")

# ------------------------------------------------------------------ per rod
per_rod = {}
for name, _, _ in DESIGNS:
    v = np.zeros((5, len(R)))
    for s, img in enumerate(recon[name]):
        mu, sd = bgstats(img)
        for n, r in enumerate(R):
            v[s, n] = (img[r["mask"]].mean() - mu) / sd
    per_rod[name] = v
gb, gm = per_rod["benchmark"].mean(0), per_rod["MOBO"].mean(0)
gain = gm / gb
print("\nper rod (58 rods, CNR averaged over the 5 realizations), gain = MOBO / benchmark:")
rho, p = stats.spearmanr(dist, gain)
print(f"  Spearman(rod distance, gain) over all rods: rho {rho:+.3f}, p {p:.2f}")
# within-size: regress log gain on distance with a separate intercept per size
X = np.column_stack([dist] + [(sec_of == k).astype(float) for k in range(6)])
y = np.log(gain)
beta, res_, rank, _ = np.linalg.lstsq(X, y, rcond=None)
dof = len(y) - X.shape[1]
s2 = float(((y - X @ beta) ** 2).sum() / dof)
cov = s2 * np.linalg.inv(X.T @ X)
slope, se = beta[0], np.sqrt(cov[0, 0])
print(f"  log(gain) ~ distance + size: slope {slope:+.4f} +/- {se:.4f} per mm "
      f"({100 * (np.exp(slope) - 1):+.1f}% per mm, t {slope / se:+.2f}, p {2 * stats.t.sf(abs(slope / se), dof):.2f})")
order = np.argsort(dist)
rows = []
for k in range(6):
    idx = [n for n in order if sec_of[n] == k]
    rows.append(f"    {radii[k]:.3f} mm: " + "  ".join(f"{dist[n]:.2f}:x{gain[n]:.2f}" for n in idx))
print("  per rod, by size, nearest first (distance mm : gain):")
print("\n".join(rows))
inn = [min(np.flatnonzero(sec_of == k), key=lambda n: dist[n]) for k in range(6)]
print("  innermost rod of each size (%s mm): gain %s; mean %.3f"
      % (", ".join(f"{dist[n]:.2f}" for n in inn), ", ".join(f"x{gain[n]:.2f}" for n in inn), np.mean(gain[inn])))
for name in per_rod:
    v = per_rod[name][:, inn].mean(1)
    print(f"    {name} innermost-rod sector-mean CNR {v.mean():.3f} +/- {v.std(ddof=1):.3f}")
out["per_rod"] = dict(dist_mm=dist.tolist(), sector=sec_of.tolist(), bench=gb.tolist(), mobo=gm.tolist(),
                      gain=gain.tolist(), spearman=[float(rho), float(p)], slope_log_gain_per_mm=[float(slope), float(se)])

# ------------------------------------------------------------------ ASCI over rod pixels, radial profile
print("\nASCI (asci_win, % of 180 1-degree bins covered by strips with FWHM <= 0.45 mm)")
for name in asci:
    print(f"  {name}: field mean {asci[name].mean():.2f}")
edges = np.arange(0.0, 5.01, 0.25)
prof = {}
for name in asci:
    prof[name] = [asci[name][(rmm >= a) & (rmm < b_)].mean() for a, b_ in zip(edges[:-1], edges[1:])]
print("  radial profile (annulus: bench, MOBO, MOBO - bench):")
for a, b_, pb, pm in zip(edges[:-1], edges[1:], prof["benchmark"], prof["MOBO"]):
    print(f"    {a:.2f}-{b_:.2f} mm: {pb:6.2f} {pm:6.2f} {pm - pb:+6.2f}")
diff = asci["MOBO"] - asci["benchmark"]
fine = np.arange(0.0, 5.0, 0.05)
dprof = np.array([diff[(rmm >= a) & (rmm < a + 0.05)].mean() for a in fine])
cross = [float(fine[n] + 0.05) for n in range(len(fine) - 1) if dprof[n] < 0 <= dprof[n + 1]]
print(f"  MOBO - bench, 0.05 mm annuli: changes sign from - to + at r = {cross} mm")
out["asci_profile"] = dict(edges=edges.tolist(), bench=prof["benchmark"], mobo=prof["MOBO"], crossings_mm=cross)
out["asci_rods"] = {}
for label, T in CUTS:
    inner = dist < T
    print(f"  cut {label}:")
    e = {}
    for tagT, A in (("as-is", asci), ("transposed", {k: v.T for k, v in asci.items()})):
        for reg, sel in (("inner", inner), ("outer", ~inner)):
            pix = np.logical_or.reduce([r["mask"] for r, u in zip(R, sel) if u])
            secmean = {n: np.mean([A[n][np.logical_or.reduce([r["mask"] for r, u in zip(R, sel) if u and r["sector"] == k])].mean()
                                   for k in range(6)]) for n in A}
            vb, vm = A["benchmark"][pix].mean(), A["MOBO"][pix].mean()
            e[f"{reg}_{tagT}"] = dict(bench=float(vb), mobo=float(vm), bench_secmean=float(secmean["benchmark"]),
                                      mobo_secmean=float(secmean["MOBO"]))
            print(f"    {reg:5s} rod pixels ({tagT:10s}): bench {vb:5.1f}  MOBO {vm:5.1f}  ({vm - vb:+5.1f});"
                  f" size-weighted: bench {secmean['benchmark']:5.1f}  MOBO {secmean['MOBO']:5.1f}")
    out["asci_rods"][label] = e
# ASCI averaged over the background annulus used by CNR
print(f"  background annulus (1.0-4.5 mm minus rods): bench {asci['benchmark'][bg].mean():.1f}, "
      f"MOBO {asci['MOBO'][bg].mean():.1f}")
# innermost rods
pix = np.logical_or.reduce([R[n]["mask"] for n in inn])
print(f"  innermost rod of each size: bench {asci['benchmark'][pix].mean():.1f}, MOBO {asci['MOBO'][pix].mean():.1f}")


def conv(o):
    if isinstance(o, dict):
        return {k: conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


with open(os.path.join(W, "step2_results.json"), "w") as f:
    json.dump(conv(out), f, indent=1)
print(f"\nwrote {os.path.join(W, 'step2_results.json')}")
