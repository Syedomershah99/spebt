import glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raycast_check as rc
W = os.path.dirname(os.path.abspath(__file__))
for tag in ("bench", "mobo"):
    f = glob.glob(os.path.join(W, "layouts", tag, "*.tensor"))[0]
    L = torch.load(f, weights_only=False)["layouts"]
    print(tag, sorted(L.keys()))
    for key in sorted(L.keys()):
        P = L[key]["plate segments"].double().numpy(); C = L[key]["detector units"].double().numpy()
        g = rc.geometry_report(P, C)
        gi, go = rc.aperture_widths(P)
        n_ap = P.shape[0] // 2
        g["gap_in_pct"] = np.percentile(gi, [0, 25, 50, 75, 100]).round(4).tolist()
        g["gap_out_pct"] = np.percentile(go, [0, 25, 50, 75, 100]).round(4).tolist()
        g["largest_half_in_out"] = [float(np.sort(gi)[-n_ap:].mean()), float(np.sort(go)[-n_ap:].mean())]
        g["open_frac_origin"] = rc.open_fraction(P, np.zeros(2), n=1_000_000)
        print(key, json.dumps(g))
    P = L["position 000"]["plate segments"].double().numpy(); C = L["position 000"]["detector units"].double().numpy()
    t0 = time.time()
    per, op = rc.raycast(P, C, np.array([0.2 - 0.025, -0.025]), n_rays=200_000)
    print(tag, "raycast total", per.sum(), "open", op, "nonzero crystals", (per > 0).sum(), f"{time.time()-t0:.1f}s")
