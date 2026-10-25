# Empire AI campaign: analysis plan

Written on 25 Oct 2026, when production started and before any production
result was looked at. The selection rule and the claims below are fixed in
advance so that the headline design is not chosen after seeing the numbers.

## The campaign

- Code: tag `eai-v1` (commit d27f1b4). Grace CPU nodes, six searches per node,
  each search's designs run as child processes of its controller.
- Every search: CNR alone (qLogNEI) with a soft MPXI window of 1.5 to 2.5
  (`MOBO_CONSTRAINTS=mpxi_windowed_active_mean:1.5:2.5`), ring 2 at most 960
  crystals (`MOBO_RING2_MAX=960`), 21 starting designs plus 100 designs chosen
  by the optimizer, in-loop CNR at 150 ML-EM iterations with the noise seed
  derived from the design name, as on CCR.
- Node 1: three paired starts (`warm_r0`, `warm_r1`, `warm_r2`), the 21
  starting designs of CCR head-to-head arms r0, r1 (= r4) and r2 (= r5), plus
  fresh starts with LHS seeds 10 to 12. Node 2: fresh starts, seeds 13 to 18.
  Further nodes, if launched, are recorded in the results email.
- Step 2 validated the platform: Grace reproduces CCR's metrics to float
  rounding and its CNR statistically (equivalent within 0.15 over 20 seeds).

## When to read it

- Main comparison point: 21 + 80 designs per search, the CCR head-to-head
  budget. Final point: 21 + 100.

## Selection rule (re-measurement)

1. Candidates: in each search, the three optimizer designs with the highest
   in-loop CNR among those whose measured MPXI lies in [1.5, 2.5]; plus the
   benchmark (`tmi_reference_000`) and the MIC design (`nd2fine_960`).
2. Re-measure every candidate on noise seeds 0 to 19, every ML-EM iteration
   from 1 to 400, at T_sec 10 (`run_iter_count_pair.py`), on Grace.
3. The headline design is the candidate with the highest mean CNR at 150
   iterations over seeds 0 to 19.
4. Report the headline design, the benchmark and `nd2fine_960` on held-out
   seeds 20 to 39: at 150 iterations, and at each design's own best iteration
   count (chosen on seeds 0 to 19). Give mean +/- SD and the Welch 95% interval
   of each difference.
5. A CNR gain over `nd2fine_960` is claimed only if its held-out 95% interval
   excludes zero; otherwise the result is reported as equivalence.

## Search-level measures

- Best-so-far in-window in-loop CNR against evaluations, per search, with the
  median and interquartile range across searches.
- Reliability: how many searches have a candidate whose re-measured mean
  (seeds 0 to 19, 150 iterations) is within 0.10 of `nd2fine_960`'s.
- Window adherence: the share of optimizer designs whose measured MPXI lies in
  [1.5, 2.5]. CCR's arms, which maximized MPXI, were at 53 to 73 percent.
- Paired starts: `warm_r0` to `warm_r2` against CCR's two- and five-objective
  arms with the same 21 starting designs (r0, r4, r5), best-so-far in-loop CNR
  at 80 optimizer designs. The design boxes differ (ring 2 at most 960 here,
  1362 on CCR), which is stated.
- Cost: SUs from ColdFront, designs per node-hour, time per design.

## Not to be claimed

- GPU speed-ups (no GPU was used), or "batch" or "parallel" BO: each search is
  sequential (q = 1); only the searches run side by side.
- That MPXI was "held" in the window: the constraint is soft ("steered toward").
- A CNR gain over the MIC design unless rule 5 allows it.
- That CNR alone beats several objectives: not tested by these searches.
- Anything about 3-D, measured data, other phantoms or other count levels.
- That the benchmark is the built SAI system: it is our 2-D model of it, with
  evenly spaced apertures.
- That in-loop CNR values are measurements of a design's CNR: each is one noise
  draw (sigma about 0.08), and picking the largest inflates it by about 0.10.

## Addendum, 25 Oct 2026 15:30 (before any result was analyzed)

ColdFront's report confirmed that Grace is billed per node-hour (3.2 test-QoS
node-hours were charged 0.8 SU), so the second half of the campaign was
launched:

- Node 3, the multi-objective arm: the same three paired starts
  (`armB_warm_r0` to `armB_warm_r2`), optimizing CNR together with FWHM, ASCI
  and PPDS (qLogNEHVI) under the same soft MPXI window and box. Compared with
  `warm_r0` to `warm_r2` (CNR alone) and with CCR's arms on the same starts.
  Its designs are candidates under the selection rule like any other search's.
- Node 4: six more fresh starts, CNR alone, LHS seeds 19 to 24.

Jobs: nodes 1 to 4 are 148500, 148502, 148535 and 148537, each with an
afterany follow-up (148501, 148503, 148536, 148538), all standard QoS.

Correction, 15:32: the first node 3 job (148535) received only `fwhm_weighted_mean`
as its objective list, because sbatch splits `--export` values on commas. It
was cancelled after one minute, before any design finished, and resubmitted as
148540 (follow-up 148541) with the full list and new search names
(`armB2_warm_r0` to `armB2_warm_r2`). Nothing from the first job is used.
