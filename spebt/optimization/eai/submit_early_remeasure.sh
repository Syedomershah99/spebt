#!/bin/bash -l
# Early re-measurement (25 Oct 2026, code 01a6069): the parts that do not depend
# on the campaign. Rebuild CCR's 18 candidates (the two references keep their
# step 2 PPDFs), then re-measure all 20 on the selection seeds 0-39 at T_sec 10,
# in two shards that start only if the rebuild succeeded. The campaign's own
# candidates follow in RUN remeasure_sel once every search has ended.
P=/projects/su/ryao1/develop/oshah
C=$P/code_01a6069
I=$P/remeasure_inputs/early
a=$(sbatch --parsable --qos=standard --time=06:00:00 \
     --export=ALL,CODE=$C,CANDIDATES="$I/refs.csv $I/ccr_candidates.csv",LIST=$I/designs.txt \
     "$C/optimization/eai/rebuild.sh") || exit 1
echo "rebuild $a"
for k in 0 1; do
  b=$(sbatch --parsable --qos=standard --time=12:00:00 --dependency=afterok:"$a" \
       --export=ALL,CODE=$C,RUN=remeasure_sel_early,DESIGNS=$I/designs.txt,SEED_FIRST=0,SEED_LAST=39,SHARD=$k/2 \
       "$C/optimization/eai/remeasure.sh") || exit 1
  echo "remeasure shard $k/2: $b after $a"
done
