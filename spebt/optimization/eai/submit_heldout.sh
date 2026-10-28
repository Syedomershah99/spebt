#!/bin/bash -l
# Held-out re-measurement (rule 4, after the headline commit 9f6a955): the headline, the
# benchmark and nd2fine_960 on seeds 40-79 at T_sec 10 (rule 5), and the count levels,
# T_sec 2.5, 5, 20 and 40 on seeds 40-59 (descriptive), one job per level, all into
# RUN remeasure_held. remeasure.sh refuses seeds from 40 on for any other design.
P=/projects/su/ryao1/develop/oshah
C=${C:?set C to the code tree that holds the committed headline.txt}
I=$P/remeasure_inputs/heldout
for spec in 10:40:79 2.5:40:59 5:40:59 20:40:59 40:40:59; do
  IFS=: read -r t a b <<< "$spec"
  j=$(sbatch --parsable --qos=standard --time=06:00:00 \
       --export=ALL,CODE=$C,RUN=remeasure_held,DESIGNS=$I/designs.txt,T_SEC=$t,SEED_FIRST=$a,SEED_LAST=$b \
       "$C/optimization/eai/remeasure.sh") || exit 1
  echo "held-out T_sec $t seeds $a-$b: $j"
done
