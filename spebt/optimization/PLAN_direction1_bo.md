# Direction 1: BO/MOBO for SC-SPECT SAI Configuration Optimization

---

# Part I: Presentation — Mathematical Foundations & Approach

## 1. Why Bayesian Optimization?

**Problem**: Evaluating a single SC-SPECT configuration requires ~30 min of PPDF ray-tracing on HPC. We want to find the optimal (aperture_diam, a, b) from a continuous 3D design space — exhaustive grid search is infeasible (e.g., 20³ = 8000 configs × 30 min = 167 days).

**Bayesian Optimization (BO)** solves this by building a cheap-to-evaluate *surrogate model* of the objective function and using it to decide which configuration to evaluate next. It is designed for:
- **Expensive** black-box functions (each evaluation costs significant compute)
- **Low-dimensional** continuous spaces (our 3D is ideal)
- **Derivative-free** optimization (no gradient of JI w.r.t. parameters available)

**BO Algorithm:**

```
Input: design space X ⊂ ℝ³, objective f(x) = JI(aperture_diam, a, b)
1. Evaluate f at N₀ initial points (Latin Hypercube Sampling, N₀ = 150)
2. Build surrogate model M(x) that approximates f(x)  (Gaussian Process)
3. For t = 1 to T:
   a. Select x_new = argmax α(x; M)   (acquisition function)
   b. Evaluate f(x_new) on HPC        (PPDF ray-tracing + beam analysis)
   c. Update dataset: D ← D ∪ {(x_new, f(x_new))}
   d. Refit surrogate M on D
4. Return: x* = argmax f(x) over all evaluated points
```

**Key advantage**: BO typically finds near-optimal solutions in 50–100 evaluations, vs. thousands for grid search. With batch BO (q=8 parallel evaluations), we achieve ~100 evaluations in ~6.5 hours of wall time.

---

## 2. Gaussian Process Surrogate — Mathematical Foundation

### 2.1 What is a Gaussian Process?

A Gaussian Process (GP) is a distribution over functions. For any finite set of inputs {x₁, ..., xₙ}, the corresponding outputs {f(x₁), ..., f(xₙ)} follow a multivariate Gaussian distribution:

```
f(x) ~ GP(m(x), k(x, x'))
```

where:
- **m(x)** = mean function (typically set to 0 after normalizing data)
- **k(x, x')** = kernel/covariance function (encodes assumptions about smoothness)

### 2.2 GP Prediction

Given training data D = {(xᵢ, yᵢ)}ᵢ₌₁ⁿ, the GP predicts at a new point x* as:

```
μ(x*) = k(x*, X) [K + σ²ₙI]⁻¹ y        (posterior mean)
σ²(x*) = k(x*, x*) - k(x*, X) [K + σ²ₙI]⁻¹ k(X, x*)  (posterior variance)
```

where:
- K = kernel matrix with Kᵢⱼ = k(xᵢ, xⱼ)
- σ²ₙ = observation noise variance (learned from data)
- k(x*, X) = vector of kernel evaluations between x* and training points

**Intuition**: The GP gives us both a *prediction* μ(x*) AND an *uncertainty* σ(x*) at every point. This uncertainty is crucial — it tells us where the model is confident and where it's uncertain, guiding the search.

### 2.3 Why GP for Our Problem?

| Property | Relevance to our problem |
|----------|------------------------|
| Works well in low dimensions (d ≤ 10) | Our design space is 3D |
| Natural uncertainty quantification | Needed for acquisition function (EI) |
| Data-efficient (works with 100–300 points) | Each evaluation costs 30 min on HPC |
| Smooth interpolation | Physical systems produce smooth response surfaces |
| Proven for SPECT optimization | Kirtiraj's `bo_agent.py` uses same `SingleTaskGP` from BoTorch |

---

## 3. Matérn 5/2 Kernel — Why This Choice

### 3.1 Kernel Definition

The kernel k(x, x') controls the GP's assumptions about the function's smoothness. We use the **Matérn 5/2** kernel:

```
k_Matérn5/2(x, x') = σ² (1 + √5·r + 5r²/3) exp(-√5·r)

where r = √(Σᵢ (xᵢ - x'ᵢ)² / lᵢ²)
```

Parameters:
- **σ²** (output scale): controls the amplitude of the function's variation
- **lᵢ** (lengthscale per dimension): controls how quickly the function varies along dimension i
- Using **ARD** (Automatic Relevance Determination): a separate lengthscale per dimension, so the GP learns which parameters (aperture_diam, a, b) matter most

### 3.2 Why Matérn 5/2 and Not Other Kernels?

| Kernel | Smoothness | Why (not) for us |
|--------|-----------|-----------------|
| **RBF (Squared Exponential)** | C∞ (infinitely differentiable) | Too smooth — real physical systems have finite smoothness. Can overfit. |
| **Matérn 3/2** | C¹ (once differentiable) | Slightly too rough for physical systems. |
| **Matérn 5/2** | C² (twice differentiable) | **Just right** — physical systems like PPDF ray-tracing are smooth but not infinitely so. Standard choice for engineering optimization. |

### 3.3 ARD Lengthscales — Automatic Feature Importance

With ARD, the GP learns a separate lengthscale lᵢ for each dimension:
- **Small lᵢ** → function varies rapidly in dimension i → parameter i is **important**
- **Large lᵢ** → function varies slowly in dimension i → parameter i has **less effect**

After fitting, we can inspect the lengthscales to see which parameters the GP considers most influential. We expect `l_aperture_diam` to be smallest (most impactful per literature).

---

## 4. Expected Improvement (EI) — Why This Acquisition Function

### 4.1 The Exploration-Exploitation Dilemma

At each BO iteration, we must choose: evaluate near the current best (exploit) or explore uncertain regions (explore)? The acquisition function α(x) balances this tradeoff.

### 4.2 Expected Improvement Definition

Given current best observed value f* = max(y₁, ..., yₙ), the Expected Improvement at point x is:

```
EI(x) = E[max(f(x) - f*, 0)]

     = (μ(x) - f*) · Φ(z) + σ(x) · φ(z)

where z = (μ(x) - f*) / σ(x)
      Φ = standard normal CDF
      φ = standard normal PDF
```

**Two terms:**
- **(μ(x) - f*) · Φ(z)** — exploitation: high when predicted mean μ(x) is much better than f*
- **σ(x) · φ(z)** — exploration: high when uncertainty σ(x) is large (we don't know what's there)

### 4.3 Why EI and Not Other Acquisition Functions?

| Acquisition | Pros | Cons | Choice |
|------------|------|------|--------|
| **Expected Improvement (EI)** | Principled balance of explore/exploit; well-understood; batch version (qEI) available | Can get stuck if landscape is very noisy | **Our choice** |
| Upper Confidence Bound (UCB) | Simple; tunable β parameter | Requires manual tuning of exploration parameter β | Not chosen |
| Knowledge Gradient (KG) | Optimal for finite horizons | Much more expensive to compute; overkill for our problem | Not chosen |
| Thompson Sampling | Good for high-dimensional | Less sample-efficient in 3D than EI | Not chosen |

### 4.4 Batch Expected Improvement (qEI)

For parallel HPC evaluation, we need to select q candidates simultaneously. **qEI** (quasi-Monte Carlo Expected Improvement) selects q points that jointly maximize expected improvement:

```
qEI(x₁, ..., xq) = E[max(max(f(x₁), ..., f(xq)) - f*, 0)]
```

This is computed via Monte Carlo sampling (256 posterior samples). The q points are selected to be *diverse* — they don't all cluster near the same promising region.

**Implementation**: BoTorch's `qExpectedImprovement` with `optimize_acqf` and L-BFGS-B optimizer with 10 random restarts.

---

## 5. Implementation Architecture

### 5.1 Pipeline Per Configuration

```
┌──────────────────────────────────────────────────────────────────┐
│  BO Agent proposes: (aperture_diam=0.6, a=0.3, b=0.15)         │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 0: Geometry Generation (~1 min)                           │
│  generate_mph_scanner_circularfov.py --aperture_diam 0.6        │
│  → scanner_layouts_<md5>_rot2_ang1p0deg.tensor                  │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: PPDF Ray-Tracing (~25 min, parallelized)               │
│  2 layouts × 8 T8 poses = 16 HDF5 files                        │
│  arg_ppdf_t8.py --a_mm 0.3 --b_mm 0.15 (for each layout/pose)  │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Steps 2-4: Beam Analysis (~5 min)                              │
│  Extract masks → Extract properties → ASCI histograms           │
│  Aggregate over 16 HDF5 files                                   │
└──────────────────────┬───────────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 5: Compute JI                                             │
│  JI = (mean_sensitivity / mean_FWHM²) × mean_ASCI_pct / 100    │
│  → Append to results_summary.csv                                │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 BO Loop Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Initial Data: 150 configs via Latin Hypercube Sampling     │
│  Each evaluated on HPC via SLURM array job                  │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Fit GP: SingleTaskGP(Matérn 5/2 + ARD) on 150 points      │
│  Validate: R² > 0.85 on 20% held-out                       │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
         ┌────────────────────────────┐
         │  BO Loop (13 rounds)       │◄────────────────────┐
         │                            │                     │
         │  1. qEI selects q=8 points │                     │
         │  2. Write to manifest CSV  │                     │
         │  3. sbatch --array=0-7     │                     │
         │  4. Wait for SLURM jobs    │                     │
         │  5. Collect JI results     │                     │
         │  6. Refit GP               │                     │
         │  7. Check convergence      │─── not converged ───┘
         │                            │
         └─────────┬──────────────────┘
                   │ converged / max iters
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  Report: best (aperture_diam*, a*, b*), convergence curve   │
│  Validate top-5 with full reconstruction (MLEM + CNR)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Potential Problems & Solutions

| Problem | Likelihood | Solution |
|---------|-----------|---------|
| **GP underfitting** (R² < 0.75) | Low (3D is easy for GP) | Increase LHS to 200 points; check for outliers; try log-transform of JI |
| **BO gets stuck** in local optimum | Medium | qEI with q=8 naturally explores diverse regions; add random restarts |
| **Geometry generation fails** for extreme aperture_diam | Low | Add physical constraint: aperture_diam < chord_spacing (1.22mm with 180 apertures). Bounds [0.2, 1.0] are already safe. |
| **PPDF computation fails** for extreme (a,b) values | Low | Bounds [0.1, 1.0] keep FOV center within safe region |
| **Beam analysis crashes** on configs with very low sensitivity | Medium | Add try/except in pipeline; assign JI=0 to failed configs; GP handles noisy observations |
| **SLURM queue delays** slow BO loop | Medium | Submit batch as high-priority; use shorter walltime (2h vs 6h); preemptive jobs |
| **Convergence too slow** | Low | Increase q to 12; reduce stopping threshold; or use UCB with high β for more exploration |
| **Different aperture_diam changes number of crystals that "see" each pixel** | Possible | This is actually the desired physical effect — larger apertures increase sensitivity but blur resolution. GP will learn this tradeoff. |

---

## 7. Expected Results

### 7.1 GP Surrogate
- **R² > 0.85** on held-out data (3D continuous GP with 150 training points is well within GP's strength)
- GP will likely show that `aperture_diam` has the shortest lengthscale (most important)
- Smooth response surface with clear resolution-sensitivity tradeoff axis

### 7.2 Single-Objective BO on JI
- **Convergence within 30–50 evaluations** (typical for 3D BO)
- Expect JI improvement of **20–50%** over baseline (aperture_diam=0.4, a=b=0.2)
- Optimal aperture_diam likely around **0.3–0.6mm** (balancing resolution and sensitivity)
- Optimal (a, b) likely **non-circular** ellipse — different sampling density in X vs Y

### 7.3 MOBO Pareto Front
- Pareto front will reveal the **resolution-sensitivity tradeoff** driven primarily by aperture_diam
- Small aperture_diam → high resolution (low FWHM) + low sensitivity → one end of Pareto front
- Large aperture_diam → low resolution + high sensitivity → other end
- T8 trajectory (a, b) affects ASCI independently of aperture_diam → third axis of tradeoff
- **"Knee" configuration** = best compromise, likely dominates single-metric JI optimal

### 7.4 Validation
- Top-5 configs validated with full MLEM reconstruction
- Expect visually sharper hot rod phantom images for BO-optimal configs
- CNR improvement over baseline

---

## 8. Phases of Implementation

| Phase | Week | What | Deliverable |
|-------|------|------|-------------|
| **1a** | 1 | Parameterize aperture_diam in geometry generator; LHS 150 configs | `generate_configs.py`, `configs_manifest.csv` |
| **1b** | 1 | Adapt SLURM pipeline for SAI; submit LHS sweep | `run_sai_pipeline.sh`, 2400 HDF5 files |
| **1c** | 1 | Adapt beam analysis + JI computation for SAI | `compute_ji.py`, `results_summary.csv` |
| **2a** | 2 | Build BO agent (adapt Kirtiraj's); fit GP on LHS data | `bo_agent.py`, GP R² validation |
| **2b** | 2 | Build batch BO orchestrator with SLURM | `run_bo_loop.py` |
| **3** | 3 | Run single-objective BO (~100 evals); convergence analysis | Convergence plots, best config |
| **4** | 4 | MOBO with qNEHVI; Pareto front analysis | `mobo_agent.py`, Pareto plots |
| **5** | 5 | Validate top-5 with MLEM reconstruction; generate figures | Reconstruction comparisons, publication figures |
| **6–7** | 6–7 | Write IEEE MIC abstract; revise; submit by May 12 | Abstract + figures |

---

# Part II: Technical Implementation Plan

## Context & Motivation

**Problem**: The SC-SPECT SAI system has two types of design parameters: hardware (aperture diameter, fixed at 0.4mm) and acquisition (T8 ellipse trajectory, a=b=0.2mm). Only one configuration has been evaluated. The 3D design space (aperture_diam × a × b) is unexplored, and aperture diameter is the #1 most impactful SPECT parameter per literature.

**Goal**: Apply Bayesian Optimization to co-optimize hardware (aperture diameter) and acquisition (T8 trajectory) parameters that maximize JI. Adapt Kirtiraj's working BO implementation for the SAI system. Start with 3D continuous BO, extend to more parameters later.

**Target**: IEEE MIC abstract deadline May 12, 2026 (~6 weeks from Mar 30).

**Lab context**: Kirtiraj does scalar BO on JI for MPH SPECT with (pinhole_diameter, displacement) — 2D. We do the SAI analogue: co-optimize (aperture_diameter, T8_a, T8_b) — 3D. Same BoTorch framework, same SingleTaskGP, applied to SC-SPECT with batch BO for HPC parallelism.

---

## A. System Comparison: Kirtiraj's MPH vs Our SAI

| Aspect | Kirtiraj (MPH) | Ours (SAI SC-SPECT) |
|--------|----------------|---------------------|
| **Design vector** | (Diameter_mm, Displacement_mm) — 2D | (aperture_diam_mm, a_mm, b_mm) — 3D |
| **What varies** | Collimator geometry (pinhole size + detector shift) | Aperture size (hardware) + T8 trajectory (acquisition) |
| **Geometry** | 24 pinholes, 215mm collimator ring, 757mm detector ring | 180 apertures, 67.5mm HR ring, 4 detector rings (260–650mm) |
| **Crystals** | ~100 detectors, 3.5mm tangential | 3360 crystals, 0.84mm tangential |
| **FOV** | 280×280 px, 70×70mm | 200×200 px, 10×10mm |
| **Rotations** | 15 collimator rotations | 2 collimator rotations (0°, 1°) + 8 T8 poses |
| **HDF5 per config** | 15 files (one per rotation) | 16 files (2 layouts × 8 poses, fixed) |
| **PPDF cost** | ~30 min/config (estimated) | ~75 min/config (measured) |
| **Initial data** | 25 configs (5×5 grid) | 1 config (a=0.2, b=0.2, rot=1°, 8 poses) |
| **JI formula** | JI = (sens_mean / FWHM²) × ASCI_pct / 100 | Same formula |
| **BO approach** | Sequential q=1, 100 iters | Proposed: batch q=4–8 |
| **Code** | [bo_agent.py](Bayesian-Optimization-For-Self-Collimating-SPEC/bo_agent.py), [12_run_bo_loop_checkpointed.py](Bayesian-Optimization-For-Self-Collimating-SPEC/12_run_bo_loop_checkpointed.py) | To be adapted |

**Key insight**: Both Kirtiraj and us co-optimize hardware + acquisition. Kirtiraj: pinhole diameter (hardware) + detector displacement (hardware). Us: aperture diameter (hardware) + T8 trajectory (acquisition). Each config requires geometry regeneration (new `.tensor` file for different aperture_diam), exactly like Kirtiraj regenerates geometry per config. Same BoTorch framework, same SingleTaskGP, one additional dimension (3D vs 2D).

---

## B. Design Decisions (Brainstorm)

### B.1 Design Vector: What Parameters to Optimize

**Design vector (3D, all continuous)**:

| Parameter | Type | Range | Current value | Literature impact rank |
|-----------|------|-------|---------------|----------------------|
| `aperture_diam_mm` | continuous | [0.2, 1.0] | 0.4 | **#1** — "most impactful SPECT parameter" (Van Audenhaege 2015, PMC4337004) |
| `a_mm` | continuous | [0.1, 1.0] | 0.2 | **#3** — trajectory controls spatial sampling pattern (Metzler 2016) |
| `b_mm` | continuous | [0.1, 1.0] | 0.2 | **#3** — non-circular ellipse may outperform circular (unique to SC-SPECT T8) |

**Fixed parameters** (not optimized in initial 3D BO):
- `n_apertures` = 180 (could be added later for 4D extension)
- `phase_deg` = 0.0
- `rot_angle_deg` = 1.0° (2 collimator rotations: 0° and 1°)
- `n_poses` = 8 (T8 protocol)

**Why these bounds?**
- `aperture_diam_mm` 0.2–1.0mm: smaller = better resolution but lower sensitivity; upper bound limited by physical overlap constraint (chord spacing ≈ 1.22mm with 180 apertures on r=35mm ring)
- `a_mm`, `b_mm` 0.1–1.0mm: lower bound must be nonzero (otherwise T8 degenerates to static); FOV is 10mm diameter, so 1.0mm offset = 10% of FOV radius

**Code changes required:**
1. **`aperture_diam_mm`**: Modify [generate_mph_scanner_circularfov.py:171](omer/spebt/geometry/generate_mph_scanner_circularfov.py#L171) — change `APERTURE_DIAM_MM = 0.4` to accept CLI arg `--aperture_diam`. This regenerates the `.tensor` file per config (same as Kirtiraj's geometry regen step).
2. **`a_mm, b_mm`**: Already parameterized in [arg_ppdf_t8.py](omer/spebt/pymatcal/arg_ppdf_t8.py) via `--a_mm` and `--b_mm` CLI args. No code change needed.

**GP handling**: All 3 parameters are continuous → standard `SingleTaskGP` with Matérn 5/2 kernel, identical to Kirtiraj's approach. No need for `MixedSingleTaskGP`.

**Evaluation cost**: Fixed at ~30 min/config (2 layouts × 8 T8 poses), same as current pipeline. The only added step is geometry regeneration (~1 min, negligible).

**Literature backing for parameter choices:**

| Parameter | Literature support | Expected impact |
|-----------|-------------------|-----------------|
| `aperture_diam_mm` | Van Audenhaege 2015: "aperture diameter = dominant factor"; PMC4337004: "pinhole geometry is critical factor in image quality"; "smaller diameter → better resolution but lower sensitivity" | **HIGH** — #1 most impactful parameter in SPECT design. Controls fundamental resolution-sensitivity tradeoff. |
| `a_mm, b_mm` (T8 ellipse) | Metzler 2016 (PMC5113736): "non-uniform angular sampling → significantly lowered image variance"; AdaptiSPECT-C: spatial sampling augmentation improves image quality | **MODERATE** — controls how densely the FOV is sampled. Non-circular T8 could fill angular gaps. |

**Analogy to Kirtiraj's parameters:**
- `aperture_diam_mm` ↔ Kirtiraj's `Diameter_mm` (pinhole diameter) — direct analogue, same physical meaning
- `a_mm, b_mm` ↔ Kirtiraj's `Displacement_mm` (detector shift) — both affect sampling diversity, different mechanisms

**Future extension to 4D+**: Add `n_apertures` ∈ {90, 120, 150, 180, 240} (discrete) → `MixedSingleTaskGP`. Add `phase_deg`, `rot_angle_deg`, `n_poses` for full 7D optimization. Start 3D, prove the pipeline works, then expand.

### B.2 Metrics: What to Optimize For — Literature-Backed Analysis

**Literature hierarchy of SPECT optimization metrics:**

| Tier | Metric | What it measures | Literature support | Our use |
|------|--------|-----------------|-------------------|---------|
| Gold standard | CHO SNR / AUC | Task-based detection performance | Van Audenhaege 2015 (Med Phys), Barrett 2013 | Too expensive for BO loop |
| Gold standard | Resolution-variance tradeoff | Minimize variance at target resolution | Furenlid/Barrett (PMC3703762) | Conceptually aligned with JI |
| System-level | **FWHM** (resolution) | Smallest detectable feature | All SPECT optimization papers; "high-resolution compensates 3-4× for sensitivity loss in CNR" | **BO objective (via JI)** |
| System-level | **Sensitivity** | Detection efficiency → noise level | Van Audenhaege: "fix target resolution, maximize sensitivity" | **BO objective (via JI)** |
| SC-SPECT specific | **ASCI** | Angular sampling completeness | Han, Tripathi, Yao (JNM 2026): "great potential as SC-SPET optimization metrics" | **BO objective (via JI)** |
| Uniformity | Sensitivity uniformity | Non-uniformity → ring artifacts | Literature: <1-3% non-uniformity needed | **Consider adding** |
| End-to-end | **CNR** | Lesion detectability (Rose criterion >3-5) | Standard validation metric | **Validation only** |
| Composite | **JI** = sens×ASCI/FWHM² | Combines 3 system-level metrics | Harsh SNMMI 2026: JI-optimal → near-peak CNR | **Primary BO objective** |

**Assessment: Our metrics are well-aligned with literature.**

- **JI** combines the three most impactful system-level metrics per literature consensus
- **MOBO on (ASCI, FWHM, sensitivity) individually** — exactly what the literature recommends for understanding trade-offs
- **CNR for validation only** — correct approach (too expensive for BO loop, but gold standard for final assessment)
- **Potential enhancement**: Add **sensitivity uniformity** (std/mean across FOV) as a 4th MOBO objective — literature shows non-uniform sensitivity creates reconstruction artifacts

**Primary objective — JI (Joint Index)**:
- `JI = (mean_sensitivity / mean_FWHM²) × mean_ASCI_pct / 100`
- Harsh validated that JI-optimal configs produce best reconstruction quality (SNMMI 2026)
- Same formula as Kirtiraj → direct comparison possible
- Computed from beam analysis only (no reconstruction needed) → fast
- Literature backing: resolution-variance tradeoff framework (minimize variance = maximize sensitivity, at fixed resolution = fixed FWHM); ASCI adds SC-SPECT-specific angular completeness

**Sub-metrics for MOBO**:
1. **ASCI** (maximize) — angular sampling completeness, 0–100%. Literature: "ASCI maps strongly correlate to SC-SPET operation scheme; higher rotational steps → increased ASCI → higher-resolution images" (Han et al. JNM 2026)
2. **FWHM** (minimize) — spatial resolution in mm. Literature: most impactful physical parameter; "improved resolution can compensate 3-4× for sensitivity loss" (Van Audenhaege 2015)
3. **Sensitivity** (maximize) — detection efficiency. Literature: directly controls noise/variance; "volume sensitivity averaged across FOV" recommended over point sensitivity
4. **Sensitivity uniformity** (maximize, optional) — std/mean across FOV. Literature: non-uniformity creates ring artifacts in reconstruction

**Validation metric — CNR (Contrast-to-Noise Ratio)**:
- Requires full forward projection + MLEM reconstruction (~2× more compute)
- Use for top-K validation only, not for BO objective
- Literature: CNR is the standard end-to-end image quality metric (Rose criterion: CNR > 3-5 for detectability)

**Recommendation**: BO on JI first (single-objective, fast). Then MOBO on (ASCI, -FWHM, sensitivity) to find Pareto trade-offs. Validate top configs with CNR.

### B.3 Surrogate Model Selection

**SingleTaskGP with Matérn 5/2 (same as Kirtiraj)**:
- Identical to Kirtiraj's proven `SingleTaskGP` from BoTorch — works for SPECT optimization
- 3D continuous input, 1D output → textbook GP regime (works well with 100–200 points)
- Built-in uncertainty quantification → required for EI acquisition function
- Matérn 5/2 kernel with ARD (Automatic Relevance Determination) — learns which parameters matter most
- Exact GP — feasible up to ~2000 data points

**Why this is ideal**:
- 3D continuous = same complexity as Kirtiraj's 2D (just one more dimension)
- No discrete parameters → no need for `MixedSingleTaskGP`
- Proven to work for SPECT BO (Kirtiraj's results)

**Implementation** (directly adapted from Kirtiraj's `bo_agent.py`):
```python
from botorch.models import SingleTaskGP
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

BOUNDS_MIN = [0.2, 0.1, 0.1]    # aperture_diam_mm, a_mm, b_mm
BOUNDS_MAX = [1.0, 1.0, 1.0]

gp = SingleTaskGP(train_x_norm, train_y_std)
MC_EI = qExpectedImprovement(model=gp, best_f=train_y_std.max(), num_samples=256)
candidates, acq_value = optimize_acqf(
    acq_function=MC_EI,
    bounds=bounds_norm,
    q=q,
    num_restarts=10,
    raw_samples=512,
)
# candidates shape: (q, 3) → q configs to submit in parallel
```

For MOBO, use `ModelListGP` (one `SingleTaskGP` per objective) + `qNoisyExpectedHypervolumeImprovement`.

### B.4 Batch BO for CCR/SLURM

**Kirtiraj's approach (sequential, q=1)**:
- Proposes 1 candidate → submits SLURM job → waits → collects result → repeat
- 100 iterations × ~30 min = ~50 hours wall time
- Wastes HPC parallelism (only 1 SLURM job active at a time)

**Our approach (batch, q=4–8)**:
- Propose q candidates per round using `qExpectedImprovement` (not `ExpectedImprovement`)
- Submit q SLURM jobs simultaneously as array job
- Wait for all q to finish → collect results → refit GP → repeat
- 100 evals with q=8 = 13 rounds × ~30 min = ~6.5 hours wall time (vs ~50 hours sequential)

**Implementation**:
```python
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf

q = 8  # batch size = number of parallel SLURM jobs
MC_EI = qExpectedImprovement(model=gp, best_f=train_y.max(), num_samples=256)
candidates, acq_value = optimize_acqf(
    acq_function=MC_EI,
    bounds=bounds_norm,
    q=q,
    num_restarts=10,
    raw_samples=512,
)
# candidates shape: (q, 3) → 8 configs to submit in parallel
```

**SLURM batch submission**:
```bash
# Write q configs to manifest, submit as array job
sbatch --array=0-$(($q-1)) run_config_pipeline.sh
```

**Batch size choice**:
- q=4: conservative, faster GP refit, more adaptive
- q=8: matches `--cpus-per-task=16` (2 configs per node), faster convergence
- q>8: diminishing returns (candidates become redundant)
- Recommendation: q=8 (matches SLURM resource allocation)

### B.5 Stopping Criteria

Multiple criteria (stop when ANY is met):

1. **Budget**: Max 100 total evaluations (13 batch rounds × q=8 ≈ 104 evals)
2. **Convergence**: `|JI_best(t) - JI_best(t-5)| / JI_best(t) < 0.01` for 3 consecutive rounds
3. **GP confidence**: Max predictive std over design space < 5% of JI range
4. **Wall time**: Hard limit of 24 hours (HPC allocation constraint)

**Kirtiraj uses**: fixed 100 iterations. We should add convergence checks since each eval is more expensive.

### B.6 Two-Fidelity BO Feasibility

**Fidelity levels available**:
| Level | Steps | Cost | Output |
|-------|-------|------|--------|
| Low (beam-only) | PPDF → masks → props → ASCI → JI | ~30 min | JI from beam analysis |
| High (full recon) | + forward projection → MLEM → CNR | ~75 min | JI + CNR |

**Assessment**:
- Cost ratio is only ~2.5× (low → high) — typical multi-fidelity BO needs >10× ratio to be worthwhile
- JI already correlates well with reconstruction quality (Harsh's SNMMI finding)
- BoTorch's `SingleTaskMultiFidelityGP` with `MFKG` (Multi-Fidelity Knowledge Gradient) acquisition could work
- BUT: adds significant implementation complexity for modest speedup

**Recommendation**: NOT worth it for MIC deadline. Use single-fidelity BO on JI (beam-only, ~30 min). Only run reconstruction on top-5 final configs for validation. This effectively gives us the "low fidelity" evaluation at ~30 min/config, making 100 evals = ~13 rounds × 30 min = ~6.5 hours total.

**Wait — this is important**: If we skip reconstruction during BO and only compute JI from beam analysis (steps 1–4 of the pipeline), each eval is ~30 min, not ~75 min. This is the practical approach:
- BO loop: optimize JI from beam analysis only (fast)
- Post-BO: run full reconstruction on top-5 configs (validate with CNR)

---

## C. Adaptation Plan: Kirtiraj → SAI

### What to reuse from Kirtiraj's code:

| Kirtiraj file | Adaptation needed |
|---------------|-------------------|
| `bo_agent.py` | Change bounds to 3D (aperture_diam, a, b), keep `SingleTaskGP` + qEI core — minimal changes |
| `12_run_bo_loop_checkpointed.py` | Change from q=1 to q=8, adapt SLURM submission, add convergence check |
| `6_calc_ji.py` | Change FOV from 280×280 to 200×200, 16 layouts (2×8 T8 poses) |
| `run_config_pipeline.sh` | Simplify to steps 1–4 only (PPDF → beam analysis → JI), adapt paths |
| `FINAL_SUMMARY_RANKING.csv` format | Reuse CSV schema for results tracking |

### What to modify in existing SAI code:

| File | Change |
|------|--------|
| [generate_mph_scanner_circularfov.py:171](omer/spebt/geometry/generate_mph_scanner_circularfov.py#L171) | Add `--aperture_diam` CLI arg. Replace `APERTURE_DIAM_MM = 0.4` with argparse value. |
| Beam analysis scripts | Uncomment and adapt for T8 multi-pose aggregation (16 HDF5 files) |

### What to build new:

| Component | Reason |
|-----------|--------|
| `generate_configs.py` | LHS sampling in 3D (aperture_diam, a, b) — all continuous |
| `run_sai_pipeline.sh` | SAI-specific SLURM: geometry regen + PPDF + beam analysis + JI |
| `bo_agent.py` | `SingleTaskGP` + `qEI` — adapted from Kirtiraj with 3D bounds |
| Batch SLURM orchestrator | Kirtiraj does sequential; we need parallel batch submission |

### Critical code path for one SAI config evaluation:

```
(aperture_diam, a, b) →

Step 0: Geometry generation
  generate_mph_scanner_circularfov.py --aperture_diam <d>
    → scanner_layouts_<md5>_rot2_ang1p0deg_....tensor
  (Generates new .tensor file with different aperture diameter)

Step 1: PPDF computation (8 T8 poses × 2 layouts = 16 HDF5 files)
  ellipse_offsets_t8(a, b, phase=0) → 8 (dx, dy) offsets
  For each layout_idx in [0, 1]:
    For each pose_idx in [0..7]:
      arg_ppdf_t8.py --layout_idx <L> --a_mm <a> --b_mm <b>
        → layout_{L}_pose_{P}_subvoxels.hdf5

Steps 2-4: Beam analysis (aggregate over 16 HDF5 files)
  arg_extract_beam_masks.py → beams_masks_configuration_XX.hdf5
  arg_extract_beam_properties.py → beams_properties_configuration_XX.hdf5
  arg_analyze_asci.py → asci_histogram_XX.pt

Step 5: compute_ji.py → JI scalar (+ ASCI, FWHM, sensitivity)
```

**Geometry caching**: Each unique `aperture_diam` value generates a unique `.tensor` file (MD5 hash changes with aperture diameter). Files are cached — same aperture_diam reuses existing file. In practice, LHS samples will all have unique aperture_diam values, so each config generates a fresh geometry (~1 min, negligible vs ~30 min PPDF cost).

---

## D. Implementation Plan

### Phase 1: Initial Data + Single-Objective BO (Weeks 1–3)

#### Week 1: Data Generation Infrastructure

**1a. Code modification: parameterize aperture diameter**
- Modify [generate_mph_scanner_circularfov.py](omer/spebt/geometry/generate_mph_scanner_circularfov.py):
  - Add `argparse` with `--aperture_diam` (default 0.4)
  - Replace `APERTURE_DIAM_MM = 0.4` with CLI value
  - All other geometry params stay fixed (n_apertures=180, ring dims, crystal dims)
  - This is the ONLY code change to existing files needed

**1b. LHS sampling** — `omer/spebt/optimization/generate_configs.py`
- 150 configs via `scipy.stats.qmc.LatinHypercube(d=3)`
- Scale to bounds: aperture_diam∈[0.2,1.0], a∈[0.1,1.0], b∈[0.1,1.0]
- Output: `configs_manifest.csv` with columns `[idx, aperture_diam_mm, a_mm, b_mm, work_dir]`

**1c. SAI pipeline script** — `omer/spebt/optimization/run_sai_pipeline.sh`
- Adapt from Kirtiraj's `run_config_pipeline.sh`
- Steps: geometry regen → PPDF (16 files) → beam analysis → JI
- Key differences from Kirtiraj:
  - Geometry regen uses `--aperture_diam` (not diameter+displacement)
  - 2 layouts × 8 T8 poses (not 15 rotations)
  - 200×200 FOV (not 280×280)
  - T8 poses parallelized within job using GNU parallel
- SLURM: `--array=0-149`, `--cpus-per-task=16`, `--mem=40G`, `--time=02:00:00`

**1d. Uncomment + adapt beam analysis scripts**
- [arg_extract_beam_masks.py](omer/spebt/pymatana/ppdf-analysis/beam-analysis/arg_extract_beam_masks.py) — currently commented out, adapt for T8 16-file aggregation
- [arg_extract_beam_properties.py](omer/spebt/pymatana/ppdf-analysis/beam-analysis/arg_extract_beam_properties.py) — same
- [arg_analyze_extracted_properties.py](omer/spebt/pymatana/ppdf-analysis/beam-analysis/arg_analyze_extracted_properties.py) — same

**1e. JI computation** — `omer/spebt/optimization/compute_ji.py`
- Adapt Kirtiraj's `6_calc_ji.py` for 200×200 FOV, 16 layouts (2×8 T8)
- Output: append row to `results_summary.csv`

**1f. Submit LHS sweep** on CCR
- ~150 jobs × ~30 min each = ~75 node-hours total
- With array job scheduling, completes in <6 hours wall time

#### Week 2: Surrogate Model + BO Agent

**2a. BO agent** — `omer/spebt/optimization/bo_agent.py`
- Adapt from Kirtiraj's `bo_agent.py`
- Change: 3D bounds (aperture_diam, a, b), CSV column names
- Keep: `SingleTaskGP`, `qExpectedImprovement`, `optimize_acqf`, normalize/unnormalize — identical to Kirtiraj
- Add: batch selection (q=8), convergence check

**2b. BO orchestrator** — `omer/spebt/optimization/run_bo_loop.py`
- Adapt from `12_run_bo_loop_checkpointed.py`
- Loop: propose q=8 candidates → write manifest → submit SLURM array → wait → collect JI → append to CSV → repeat
- Add checkpointing (save GP state + results CSV after each round)
- Add convergence stopping criterion

**2c. Train initial GP on LHS data**
- Fit `SingleTaskGP` on 150 points, compute R² on 20% held-out
- Target: R² > 0.85 (3D continuous GP on smooth physical system — same regime as Kirtiraj's 2D)
- If R² < 0.75: check for outliers, try ARD kernel, increase LHS to 200

#### Week 3: Run BO Loop + Analysis

**3a. Execute BO** on CCR
- Start from 150 LHS points
- Run 50–100 additional evaluations in batches of 8
- ~13 rounds × 30 min = ~6.5 hours wall time
- Monitor convergence: plot best JI vs iteration

**3b. Analysis**
- Convergence plot: best JI vs evaluation count
- GP surface visualization: 2D slices at optimal phase
- Regret plot: simple regret vs iteration
- Compare best-found config vs baseline (aperture_diam=0.4, a=0.2, b=0.2)

### Phase 2: MOBO + Validation (Weeks 4–5)

#### Week 4: Multi-Objective BO

**4a. MOBO agent** — `omer/spebt/optimization/mobo_agent.py`
- `ModelListGP` wrapping 3 `SingleTaskGP` instances (ASCI, FWHM, sensitivity)
- `qNoisyExpectedHypervolumeImprovement` (qNEHVI) acquisition
- Standard `optimize_acqf` (all continuous, no mixed needed)
- Reference point: set to worst observed values × 0.9
- Batch q=4 (MOBO is more expensive per acquisition optimization)

**4b. Run MOBO** on CCR
- Warm-start from same 150 LHS data (+ any BO-found points)
- 30–50 additional evaluations
- Extract Pareto front after each round

**4c. Pareto analysis** — `omer/spebt/optimization/analyze_pareto.py`
- Pareto front visualization: ASCI vs FWHM, ASCI vs sensitivity, 3D scatter
- Hypervolume indicator convergence
- Identify "knee" config (closest to utopia point)
- Compare Pareto configs vs JI-optimal config — do they differ?
- Analyze: how does aperture_diam trade resolution vs sensitivity across Pareto front?

#### Week 5: Validation + Figures

**5a. Validate top-5 configs** (from both BO and MOBO)
- Run full pipeline including reconstruction (steps 1–7)
- Forward projection + MLEM + CNR measurement
- Compare reconstructed images: baseline vs optimized

**5b. Publication figures** — `omer/spebt/optimization/plot_results.py`
- Fig 1: Pipeline overview (design vector → PPDF → metrics → surrogate → BO loop)
- Fig 2: BO convergence (JI vs iteration, with uncertainty bands)
- Fig 3: MOBO Pareto front (3 objectives, 2D projections)
- Fig 4: Sensitivity/ASCI/FWHM maps — baseline vs BO-optimal vs MOBO-knee
- Fig 5: Reconstruction comparison — hot rod phantom, baseline vs optimized

### Phase 3: Abstract Writing (Weeks 6–7)

**Weeks 6–7**: Write IEEE MIC abstract, polish figures, revise, submit by May 12.

---

## E. Directory Structure

```
omer/spebt/optimization/              # NEW directory
├── generate_configs.py               # LHS sampling in 3D → configs_manifest.csv
├── run_sai_pipeline.sh               # SLURM: geom regen + PPDF + beam analysis + JI
├── compute_ji.py                     # JI calculation adapted for SAI 200×200
├── bo_agent.py                       # SingleTaskGP + qEI (adapted from Kirtiraj, 3D)
├── mobo_agent.py                     # Multi-objective GP + qNEHVI
├── run_bo_loop.py                    # Batch BO orchestrator with SLURM submission
├── analyze_pareto.py                 # Pareto front extraction + analysis
├── plot_results.py                   # All publication figures
├── configs/
│   └── bo_config.yml                 # 3D bounds, q, max_iters, paths
└── results/
    ├── configs_manifest.csv          # LHS configs (3D)
    └── results_summary.csv           # All evaluated configs + metrics
```

## F. Existing Files to Reuse

| File | Role | Adaptation |
|------|------|------------|
| [arg_ppdf_t8.py](omer/spebt/pymatcal/arg_ppdf_t8.py) | PPDF computation | Already parameterized for (a, b) — no changes needed |
| [run_t8_parallel.sh](omer/spebt/pymatcal/run_t8_parallel.sh) | SLURM template | Adapt for per-config jobs with geometry regen step |
| [generate_mph_scanner_circularfov.py](omer/spebt/geometry/generate_mph_scanner_circularfov.py) | Geometry generation | Add `--aperture_diam` CLI arg (currently hardcoded 0.4mm) |
| [beam_property_extract.py](omer/spebt/pymatana/ppdf-analysis/beam-analysis/beam_property_extract.py) | FWHM, angle, sensitivity | Reuse — called by beam analysis scripts |
| [asci_plot_generation.py](omer/spebt/pymatana/ppdf-analysis/beam-analysis/asci_plot_generation.py) | ASCI map visualization | Reuse for validation figures |
| [mlem_torch_nonmpi.py](omer/spebt/recon/mlem_torch_nonmpi.py) | ML-EM reconstruction | Reuse for top-K validation |
| [fake_projections.py](omer/spebt/recon/fake_projections.py) | Forward projection | Reuse for validation |
| [bo_agent.py](Bayesian-Optimization-For-Self-Collimating-SPEC/bo_agent.py) | Kirtiraj's BO core | Adapt bounds, CSV columns, add batch q |
| [12_run_bo_loop_checkpointed.py](Bayesian-Optimization-For-Self-Collimating-SPEC/12_run_bo_loop_checkpointed.py) | BO loop orchestrator | Adapt for batch submission + SAI pipeline |
| [6_calc_ji.py](Bayesian-Optimization-For-Self-Collimating-SPEC/6_calc_ji.py) | JI formula | Adapt for 200×200 FOV, T8 aggregation |

## G. Dependencies

```
botorch>=0.11.0       # Bayesian optimization (includes gpytorch)
gpytorch>=1.12        # Gaussian process models
scipy>=1.11           # LHS sampling (scipy.stats.qmc)
rich                  # Progress display (optional, Kirtiraj uses it)
# Already available: torch, numpy, h5py, matplotlib
```

## H. Timeline

| Week | Dates | Milestone |
|------|-------|-----------|
| 1 | Mar 29–Apr 5 | LHS 150 configs, adapt SAI pipeline script, submit sweep on CCR |
| 2 | Apr 6–12 | Collect metrics, fit GP, build BO agent + batch orchestrator |
| 3 | Apr 13–19 | Run single-objective BO (50–100 evals), convergence analysis |
| 4 | Apr 20–26 | MOBO with qNEHVI, Pareto fronts |
| 5 | Apr 27–May 3 | Validate top-5 configs (full recon), generate figures |
| 6–7 | May 4–12 | Write + revise IEEE MIC abstract, submit |

## I. Verification Checklist

1. **Data pipeline**: 150 configs × 16 HDF5 files each = 2400 files generated; JI computed for all
2. **Surrogate accuracy**: `SingleTaskGP` R² > 0.85 on held-out 20% (3D continuous, same regime as Kirtiraj)
3. **BO convergence**: JI improves and stabilizes within 50 iterations
4. **Batch BO speedup**: wall time < 8 hours for 100 evaluations (vs ~50 hours sequential)
5. **MOBO Pareto**: identifies configs dominating baseline (aperture_diam=0.4, a=0.2, b=0.2)
6. **Reconstruction validation**: top-5 configs produce visibly better hot rod reconstructions
7. **Ground-truth match**: GP predictions within 15% of actual JI for validated configs

## J. IEEE MIC Abstract Outline

**Title**: "Bayesian Optimization of Acquisition Configuration for Self-Collimating SPECT"

**Key contributions**:
1. First co-optimization of SC-SPECT hardware (aperture diameter) and acquisition (T8 trajectory) using BO
2. GP surrogate (Matérn 5/2) achieves R²>0.85 on 3D design space in ~150 evaluations
3. Batch BO (q=8) with `qExpectedImprovement` leverages HPC parallelism (~6.5 hours vs ~50 hours sequential)
4. Multi-objective Pareto analysis reveals ASCI–FWHM–sensitivity trade-offs across design space
5. BO-optimal config validated with full PPDF computation + ML-EM reconstruction

---

## K. Literature References

**SPECT system optimization methodology:**
- [SPECT System Optimization Against A Discrete Parameter Space](https://pmc.ncbi.nlm.nih.gov/articles/PMC3703762/) — Furenlid/Barrett. Resolution-variance tradeoff framework, MUCRB. Pinhole position = dominant parameter.
- [Review of SPECT collimator selection, optimization, and fabrication](https://pmc.ncbi.nlm.nih.gov/articles/PMC5148182/) — Van Audenhaege 2015. Comprehensive review. "Fix target resolution, maximize sensitivity." Task-based metrics (CHO) recommended. Aperture diameter = most impactful hardware parameter.

**Angular and axial sampling:**
- [Investigation of Axial and Angular Sampling in Multi-Detector Pinhole-SPECT Brain Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC7875096/) — AdaptiSPECT-C. 2-fold angular increase = "substantial enhancement"; 4-fold = diminishing returns. 3 axial positions → 37% NRMSE improvement.
- [Adaptive Angular Sampling for SPECT Imaging](https://pmc.ncbi.nlm.nih.gov/articles/PMC5113736/) — Metzler 2016. Non-uniform angular sampling → significantly lowered image variance.
- [Angular Sampling Necessary for Clinical SPECT](https://jnm.snmjournals.org/content/jnumed/37/11/1915.full.pdf) — JNM 1997. 30 views OK, 15 introduces artifacts, 10 unacceptable.

**SC-SPECT specific:**
- [Self-Collimating SPECT With Multi-Layer Interspaced Mosaic Detectors](https://pubmed.ncbi.nlm.nih.gov/33852384/) — Ma, Yao et al. TMI 2021. Foundational SC-SPECT paper.
- [ASCI and width of PPDF strips as spatial resolution metrics in SC-SPET](https://jnm.snmjournals.org/content/66/supplement_1/251903) — Han, Tripathi, Yao. JNM 2026. "ASCI shows great potential as SC-SPET optimization metric."

**Multi-pinhole design:**
- [Advances in Pinhole and Multi-Pinhole Collimators](https://pmc.ncbi.nlm.nih.gov/articles/PMC4337004/) — Pinhole geometry = critical factor. Aperture diameter controls resolution-sensitivity tradeoff.

**Pareto optimization in SPECT:**
- [Pareto optimization of SPECT acquisition and reconstruction settings for 177Lu](https://link.springer.com/article/10.1186/s40658-024-00667-7) — EJNMMI Physics 2024. Number of projections had "limited impact"; total time and reconstruction iterations more important.

**Task-based assessment:**
- [Collimator optimization using ideal observer with model mismatch](https://pubmed.ncbi.nlm.nih.gov/26894376/) — CHO as figure of merit for collimator design. Optimal design depends on specific detection task.
