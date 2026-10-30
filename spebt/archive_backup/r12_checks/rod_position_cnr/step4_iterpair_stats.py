"""Step 4: statistics on ccr_iterpair_split.py's per-seed table (iterpair_split.csv).

For each scan time and iterate: inner and outer gain (ratio of 5-seed means, +/- SE by the delta
method), and the interaction test: per seed, CNR_inner / CNR_outer within a design (same
background), Welch t-test MOBO against benchmark.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(W, "iterpair_split.csv"))
cut = sys.argv[1] if len(sys.argv) > 1 else "median"


def gain(m, b):
    g = m.mean() / b.mean()
    return g, g * np.sqrt(m.var(ddof=1) / m.mean() ** 2 / len(m) + b.var(ddof=1) / b.mean() ** 2 / len(b))


print(f"cut {cut}: gain = MOBO / benchmark (ratio of 5-seed means +/- SE); "
      f"interaction = Welch t on per-seed inner/outer ratios")
for T in ("T2p5", "T5", "T10", "T20", "T40"):
    for it in (50, 150, 400):
        q = df[(df["T"] == T) & (df.iteration == it)]
        b = q[q.design == "tmi_reference_000"]; m = q[q.design == "nd2fine_960"]
        gi, gie = gain(m[f"in_{cut}"].values, b[f"in_{cut}"].values)
        go, goe = gain(m[f"out_{cut}"].values, b[f"out_{cut}"].values)
        rb = b[f"in_{cut}"].values / b[f"out_{cut}"].values
        rm = m[f"in_{cut}"].values / m[f"out_{cut}"].values
        t = stats.ttest_ind(rm, rb, equal_var=False)
        print(f"  {T:>4} it {it:3d}: inner x{gi:.3f} +/- {gie:.3f}   outer x{go:.3f} +/- {goe:.3f}   "
              f"inner - outer {gi - go:+.3f}   in/out ratio bench {rb.mean():.3f} MOBO {rm.mean():.3f}  "
              f"t {t.statistic:+.2f} p {t.pvalue:.3f}")
