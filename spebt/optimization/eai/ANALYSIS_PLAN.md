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

## Addendum 2, 25 Oct 2026 15:40 (before any result was analyzed)

The budget was raised to 500 to 1,000 SU ("don't compromise on results"). Changes:

- Nodes 5 and 6: the multi-objective arm on fresh starts, LHS seeds 13 to 15
  and 16 to 18 (`armB2_fresh_s13` to `armB2_fresh_s18`), three searches per
  node for memory. Their 21 starting designs are the same as those of node 2's
  CNR-only searches `fresh_s13` to `fresh_s18`, so CNR alone and several
  objectives are compared on 9 paired starts (3 warm, 6 fresh).
- Node 7: six more CNR-only fresh starts, LHS seeds 25 to 30.
- Re-measurement (rules 2 and 4): seeds 0 to 39 for selection and 40 to 79
  held out, instead of 0 to 19 and 20 to 39.
- The headline design and both references are also re-measured at T_sec 2.5,
  5, 20 and 40 (seeds 40 to 59), every iteration 1 to 400.
- The top 3 in-window designs of each of CCR's six complete head-to-head arms
  (2- and 5-objective, r0, r4, r5) are rebuilt on Grace and re-measured with the
  same protocol, so search outcomes are compared on equal terms.

## Addendum 3, 25 Oct 2026 16:00 (after a review, before any re-measurement)

These replace or sharpen the rules above. The production searches had been
running for about an hour; no re-measurement had been run or looked at.

**Pool and read point.** The headline pool is the optimizer designs of the
CNR-alone searches (nodes 1, 2, 4 and 7). The multi-objective searches (nodes 3,
5, 6; `armB2_`) are re-measured the same way but analysed separately, and never
supply the headline (this corrects addendum 1). Candidates are read at the final
point, 21 + 100 manifest rows (orphaned or lost designs count toward the 100; the
number of completed designs is reported). A candidate whose PPDFs are missing is
rebuilt from its parameters and re-measured; no candidate is dropped.

**Rule 3.** The headline is the CNR-alone candidate with the highest mean CNR at
150 iterations over the selection seeds 0 to 39. The benchmark and `nd2fine_960`
are comparators only and are never the headline; `nd2fine_960` is therefore
reported on all of its seeds.

**Rule 4.** The held-out seeds 40 to 79 are run only after the headline's name is
committed to git, only for the headline, the benchmark and `nd2fine_960`, and are
used once. Any later change of pool uses fresh seeds 80 to 119. The primary
endpoint is CNR at 150 iterations; each design's own best iteration count is
secondary and carries no claim. Percentage gains are quoted with the interval of
the difference.

**Rule 5 (replaced).** At the primary endpoint, against `nd2fine_960`, with
Welch intervals: "higher" if the 95% interval lies above 0; "lower" if it lies
below 0; "equivalent" if the 90% interval lies inside +/-0.10; otherwise
"inconclusive", with the interval quoted. With 40 seeds a side and SD about
0.10, a difference of about 0.06 is resolved at 80% power.

**Reliability.** A CNR-alone search is reliable if its single best candidate
(highest selection-seed mean at 150 iterations) has a selection-seed mean of at
least 4.603, which is `nd2fine_960`'s step-2 Grace mean (4.703) minus 0.10,
one-sided. Report k of N with its Clopper-Pearson 95% interval, and the counts at
thresholds 0.05 and 0.15 below 4.703 as sensitivity checks.

**CNR alone against several objectives.** Nine pairs share their starting
designs (warm r0 to r2; fresh seeds 13 to 18). Per pair, the difference between
the two searches' best candidates (selection-seed mean at 150 iterations);
report the nine differences, their mean and its 95% t-interval, and claim a
difference only if the interval excludes 0.

**Paired starts with CCR.** Descriptive only (three pairs, no test, neither
"matches" nor "beats"): best in-window in-loop CNR read at the furthest point all
arms reached (78 to 80 optimizer designs), with each arm's in-window count, and
CCR's values both as run and restricted to ring 2 at most 960.

**Window adherence** is a check that the constraint works: x of y measured
optimizer designs in the window, read at 80 and at the end. CCR's arms maximized
MPXI with no upper bound, so their share is context, not a baseline.

**Ring 2 at 960.** Report each search's best candidate's ring-2 count and how
many sit at exactly 960. If the headline sits at 960 and is "higher", also
re-measure it at 944 and 980 crystals.

**Data incident.** The eai-v1 launcher re-ran a fresh search's starting designs
whenever it restarted (its resume test read the wrong column), so node 1's
`fresh_s10` to `fresh_s12` hold 4 to 5 duplicate starting rows each. Duplicates
are dropped (the first row is kept), a CNR blanked by a pruned re-run is restored
from its six sector values, and a fake failure row of a measured design is
dropped. The optimizer applies the same cleaning (`mobo_agent.clean_repeated_rows`)
from before those searches' controllers started. The launcher, the prune and the
follow-up jobs were fixed the same day; the fixes change no metric.

**Corrections to the text above.** Node 1 includes the step 2 rehearsal's 11
optimizer and 12 starting designs, which were seen before the plan was written;
they are treated like any other design. Step 2 compared 20 Grace seeds with CCR's
5. CCR's head-to-head arms ran in August with unseeded in-loop CNR (seeding came
on 2 Sep), and their 2-objective r0 arm ran with ring 2 capped at 960. "CNR alone
beats several objectives: not tested" is superseded by the nine-pair comparison.

**Jobs and code after the fixes (16:10).** The live tree was moved to commit
b4e88f0, which changes only the prune and, for controllers started after it,
the cleaning of repeated rows (the optimizer's proposals on four CCR archives
are identical to those of the original code). The queued follow-ups were
cancelled and requeued with the fixed launcher: nodes 1 to 7 are first jobs
148500, 148502, 148540, 148537, 148544, 148546 and 148548, with follow-ups
148578, 148579, 148582, 148580, 148583, 148584 and 148581.

## Addendum 4, 25 Oct 2026 21:45 (data incident, before any re-measurement)

**The 16:10 code move stopped the pipelines that were running.** Moving the
live tree to b4e88f0 replaced every file under /projects, an NFS mount, so each
pipeline then running on a compute node lost its open script ("Stale file
handle") at its next read, and stopped. 26 designs were lost:

- 5 optimizer designs (`warm_r0` mobo_0009, `warm_r1` mobo_0007, `warm_r2`
  mobo_0009, `armB2_warm_r0` mobo_0001, `armB2_warm_r1` mobo_0001). They are
  recorded as lost (no_row) and count toward their search's 100.
- 21 starting designs, one in each fresh search except `fresh_s14`,
  `fresh_s26`, `fresh_s28` and `armB2_fresh_s13`, `s15`, `s17`. At 21:25, 18
  of them were requeued by appending each design's own line to its search's
  lhs_pending.csv, which the running launcher was still reading
  (eai/fix_lost_starts.sh); they run with their own parameters and noise seed,
  before the optimizer starts. `fresh_s10`, `fresh_s11` and `fresh_s12` had
  started their optimizer by then (21:10 to 21:17) and keep 20 starting
  designs (`lhs6d_e1x_006` lost); fresh_s12's late append was removed.

The follow-ups were requeued again (148891 to 148897) with a launcher that
leaves a search's starting designs alone once its optimizer has started, so
those three designs are never evaluated late. Nothing under /projects/.../spebt
is replaced while jobs run; later code goes to new paths. The nine shared starts
stay identical (the repaired searches have all 21). Per search, the report gives
the number of starting designs and of lost optimizer designs.
