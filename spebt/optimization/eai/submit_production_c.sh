#!/bin/bash -l
# Production, third part (25 Oct 2026, eai-v1; budget raised to 500-1000 SU):
#   nodes 5, 6: the multi-objective arm on fresh starts 13-15 and 16-18, paired
#               with the CNR-only searches of node 2 on the same starting designs
#   node 7:     six more CNR-only fresh starts, seeds 25-30
# MOBO_OBJECTIVES is exported, never put inside --export: sbatch splits on commas.
set -e
L=/projects/su/ryao1/develop/oshah/spebt/optimization/eai/run_node_searches.sh
S=/ddn/scratch/suny/oshah

sub() {  # job name, search root, make_searches arguments
  local a b
  a=$(sbatch --parsable --qos=standard --time=2-00:00:00 --job-name="$1" \
       --export=ALL,SEARCH_ROOT="$S/$2",SETUP="$3",MAX_ITERS=100 "$L")
  b=$(sbatch --parsable --qos=standard --time=2-00:00:00 --job-name="${1}b" --dependency=afterany:"$a" \
       --export=ALL,SEARCH_ROOT="$S/$2",SETUP="$3",MAX_ITERS=100 "$L")
  echo "$1 $a then $b"
}

export MOBO_OBJECTIVES=fwhm_weighted_mean,asci_pct_fwhm0p45,ppds_ring1,cnr_sector_mean
sub eai_node5 node5_armB2 "--warm 0 --fresh 3 --first_seed 13 --prefix armB2_"
sub eai_node6 node6_armB2 "--warm 0 --fresh 3 --first_seed 16 --prefix armB2_"
unset MOBO_OBJECTIVES
sub eai_node7 node7 "--warm 0 --fresh 6 --first_seed 25"
