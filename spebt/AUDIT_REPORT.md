# SAI SC-SPECT MOBO Pipeline — Audit Report

Focused rigor pass on the MIC-critical pipeline before scaling the CNR-in-loop
runs. Covers: correctness bugs, dead/bloated code, missing error handling,
consistency risks. All findings below are either fixed in this pass or flagged
for a decision.

## Test suite added

`tests/test_pipeline.py` — 25 tests, all passing (~5 s runtime).

| Coverage area | Tests |
|---|---|
| `backfill_cnr._find_cnr_npz` matcher (3 fallback layers) | 5 |
| `mobo_agent.is_feasible` aperture-overlap math | 4 |
| `compute_metrics._per_beam_radial_fwhm` projection math | 5 |
| `compute_metrics.compute_ppds` end-to-end on synthetic HDF5 | 4 |
| `compute_cnr._write_cnr_to_csv` row/column update semantics | 5 |
| `compute_metrics.main()` CSV column alignment on append | 2 |

Run with `python3 -m pytest tests/test_pipeline.py -v` from the repo root.

---

## Fixes applied (ready to push)

### HIGH — correctness

- **`run_mobo_loop.py::patch_manifest_status()` now anchors on `idx`.**
  Previous behaviour patched the last non-empty row regardless of the `idx`
  argument. Worked only because the caller always appended immediately before
  patching, so any interruption between append and patch (e.g. controller
  wall-time kill) could leave the wrong row updated later. Fixed by scanning
  the manifest in reverse for the row whose first column equals `idx`; falls
  back to previous behaviour with a warning if no match.

- **`mobo_agent.py` docstring corrected.** The design vector was still listed
  as `(aperture_diam_mm, n_apertures, scint_radial_thickness_mm, ring_thickness_mm)`
  from an older 3D formulation; actual is `(aperture_diam_mm, n_apertures,
  n_det_ring1, n_det_ring2)`. Also fixed the "all 4 objectives" line to be
  generic (now references `OBJ_COLUMNS`) so future flips don't require another
  docstring edit.

- **`mobo_agent.py` unused import removed.**
  `from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood`
  had no callers.

- **`mobo_agent.py` CLI help updated** to list all five objective columns
  including `cnr_mean`.

### MED — clarity / correctness of comments

- **`compute_metrics.py` docstring updated** to list `PPDS` alongside the other
  metrics, with a note that CNR is added by `compute_cnr.py` in a separate
  step (so future readers don't wonder where CNR comes from).

- **`compute_metrics.compute_fwhm_and_asci()` inline comment fixed.** The old
  comment described a wrong 5-column schema for `beam_properties`; the real
  schema (defined in `pymatana/scanner_modeling/beam_property_io.py`) is 11
  columns and column 4 is `FWHM (mm)`. New comment lists all 11 columns.

- **`run_recon_comparison.py` — dead `SPROJ = None` module variable** replaced
  with a comment explaining that `sproj` is auto-detected per PPDF file inside
  `forward_project` / `run_mlem` and cannot be a constant (varies across
  configs when `n_det_ring1/2` differ).

- **`run_recon_comparison.py` — stale phantom fallback path.** The
  auto-detection list only had the old 2-spebt phantom location, which is
  gone after the vscratch recovery. Added the 3-spebt path as the primary
  fallback (kept the 2-spebt entry as a low-priority backup for legacy
  ad-hoc invocations).

- **`run_recon_comparison.py` — "reward hacking" debug print removed.**
  Legacy tag from the JI-vs-CNR sanity-check era; the current message wasn't
  useful in the compare-mode output. Replaced with a cleaner summary using
  the user-supplied labels.

---

## Flagged for decision (NOT fixed in this pass)

These are not correctness bugs and don't affect the running MOBO loop. Bring
them up if / when time permits.

### `run_mobo_loop.py`

- **`is_job_running()` uses substring match on job IDs.** With 8-digit CCR
  IDs the odds of a false positive are effectively zero, but a strict
  word-boundary check would be more defensible.
- **Sequential q=1 loop wastes wall-time waiting.** By design (each iteration
  is a separate SLURM job). Only worth changing if we want batch acquisition,
  which is a much bigger change.
- **No timeout on `is_job_running` polling loop.** In practice the SLURM
  wall-time kills the child job, which clears squeue. But a hung squeue
  command would freeze the controller silently.

### `mobo_agent.py`

- **Penalty calculation** (`col.min() * 0.5` / `col.max() * 2.0`) is
  sign-fragile. Works because all metrics are strictly positive today.
  Documenting this constraint would be safer than changing the code.
- **`warnings.filterwarnings("ignore", category=UserWarning)`** globally
  silences ALL UserWarnings from botorch/gpytorch. Some of those flag
  numerical issues in GP fitting. Currently intentional (log noise reduction)
  but worth revisiting if a GP fit ever gives obviously-wrong lengthscales.

### `run_sai_pipeline.sh`

- **`set -uo pipefail` without `-e`** is intentional (the script wants to
  handle failures per-step), but silent-failure risk is real: any Python
  crash in step 3 or step 4 is invisible unless the user tails the `.err`
  file. `write_zero_ji` only covers KNOWN infeasibility, not arbitrary crashes.
- The legacy `write_zero_ji` function name (from the JI compound-metric era)
  is still accurate about what it does but confusing to a new reader.

### `run_recon_comparison.py`

- The `--config {baseline,bo_optimized}` CLI argument is accepted but never
  branches on. Dead arg, safe to remove — but external ad-hoc callers might
  still pass it.

---

## Legacy files (candidates for archive)

These are on the disk but not on the MIC-critical path. Consider moving to a
`legacy/` subfolder to reduce cognitive load in the `optimization/` directory:

- `bo_agent.py` — old 2D BO agent
- `run_bo_loop.py` — old 2D BO controller
- `analyze_bo_convergence.py` — 2D BO analysis
- `backfill_mpxi.py` — one-off MPXI backfill (all rows now have MPXI)
- `create_4d_csv.py` — one-off migration from 2D to 4D CSV schema
- `cleanup_lhs_results.py` — one-off cleanup
- `generate_configs.py`, `generate_configs_3d.py` — LHS generators, ran once
  at project start

None of these are imported by the live pipeline; removing or archiving them
is purely cosmetic.

---

## What did NOT need changes

- **`compute_cnr.py`** — cleanly written, correctly reuses `run_recon_comparison`
  functions, monkey-patches `N_ITERATIONS` in a try/finally block, handles
  missing PPDFs with an explicit `FileNotFoundError` catch.
- **`backfill_cnr.py`** — the three-layer matcher (exact → prefix-glob →
  design-signature) is exactly right for our naming variance across LHS vs
  MOBO configs; tests confirm all three layers work as advertised.
- **`backfill_ppds.py`** — simple, correct, idempotent. Checkpoints every 10
  rows so a crash doesn't lose work.
- **`submit_mobo.sh`** — small, correct, committed to git so it survives
  vscratch cleanup.
- **`compute_metrics.compute_ppds()`** — verified end-to-end with synthetic
  HDF5 tests. Note the PPDS metric is not currently used as a MOBO objective
  (per Dr. Yao's earlier decision); the computation is kept in place for
  potential future reformulation.
- **`sai_extract_masks.py`, `sai_extract_props.py`, `sai_analyze_asci.py`** —
  thin adapters over the existing `pymatana/ppdf-analysis/beam-analysis`
  library. Small, single-purpose, no issues found.
- **`generate_mph_scanner_circularfov.py`** — geometry generator was already
  reviewed during the SNMMI slide-15 visualization work; nothing critical
  has changed since.

---

## Verification

All fixes verified against the test suite:

```
$ python3 -m pytest tests/test_pipeline.py -v
...
============================== 25 passed in 4.64s ==============================
```

Same 25 tests pass both before AND after the fixes; no behavioral regressions.

---

## Recommendation for the CNR runs

The pipeline is in a good state to keep the current 4-objective loop running
overnight. When the CNR row count reaches ~35-45 (currently 37, growing at
~1/hour), we can push the 5-objective flip and the reverted `mobo_agent.py` /
`run_mobo_loop.py` edits together with all the audit fixes as one clean commit.
The test suite catches the most likely regression paths (matcher, column
alignment, PPDS math), so future edits have a safety net.
