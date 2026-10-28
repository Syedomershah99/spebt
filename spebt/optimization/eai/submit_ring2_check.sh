#!/bin/bash -l
# ANALYSIS_PLAN "Ring 2 at 960": the headline sits at the 960 cap and rule 5 said
# "higher" (28 Oct), so it is re-measured at 944 and 980 crystals, everything else
# the headline's (ring II diameter fixed). Build both (rebuild.sh), then seeds 0-39
# at T_sec 10 into RUN remeasure_ring2check; descriptive, against the headline's
# held-out seeds (noise draws are independent across designs).
P=/projects/su/ryao1/develop/oshah
C=${C:?set C to the code tree}
I=$P/remeasure_inputs/ring2check
a=$(sbatch --parsable --qos=standard --time=03:00:00 \
     --export=ALL,CODE=$C,CANDIDATES=$I/ring2check_candidates.csv,LIST=$I/designs.txt \
     "$C/optimization/eai/rebuild.sh") || exit 1
echo "rebuild $a"
b=$(sbatch --parsable --qos=standard --time=03:00:00 --dependency=afterok:"$a" \
     --export=ALL,CODE=$C,RUN=remeasure_ring2check,DESIGNS=$I/designs.txt,SEED_FIRST=0,SEED_LAST=39 \
     "$C/optimization/eai/remeasure.sh") || exit 1
echo "remeasure $b after $a"
