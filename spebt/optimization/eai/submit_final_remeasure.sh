#!/bin/bash -l
# Selection re-measurement of the campaign's candidates (27 Oct 2026, code 01a6069,
# ANALYSIS_PLAN addendum 6): rebuild any of the 99 campaign candidates whose PPDFs were
# pruned, then re-measure them on the selection seeds 0-39 at T_sec 10 in six shards that
# start only if the rebuild succeeded. The two references and CCR's 18 candidates were
# measured on the same seeds in RUN remeasure_sel_early; apply_rules.py reads both runs.
P=/projects/su/ryao1/develop/oshah
C=$P/code_01a6069
I=$P/remeasure_inputs/final
a=$(sbatch --parsable --qos=standard --time=06:00:00 \
     --export=ALL,CODE=$C,CANDIDATES=$I/campaign_candidates.csv,LIST=$I/designs.txt \
     "$C/optimization/eai/rebuild.sh") || exit 1
echo "rebuild $a"
for k in 0 1 2 3 4 5; do
  b=$(sbatch --parsable --qos=standard --time=12:00:00 --dependency=afterok:"$a" \
       --export=ALL,CODE=$C,RUN=remeasure_sel,DESIGNS=$I/designs.txt,SEED_FIRST=0,SEED_LAST=39,SHARD=$k/6 \
       "$C/optimization/eai/remeasure.sh") || exit 1
  echo "remeasure shard $k/6: $b after $a"
done
