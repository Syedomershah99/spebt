#!/bin/bash -l
# The follow-ups again (25 Oct 2026). The queued ones hold sbatch's own copy of
# the eai-v1 launcher, which re-runs every starting design of a fresh search
# whenever it restarts. These run the fixed launcher, with the same search roots
# and settings, each after its node's first job (afterany).
# MOBO_OBJECTIVES is exported, never put inside --export: sbatch splits on commas.
L=/projects/su/ryao1/develop/oshah/spebt/optimization/eai/run_node_searches.sh
S=/ddn/scratch/suny/oshah
grep -q "Read it with pandas" "$L" || { echo "$L is not the fixed launcher"; exit 1; }

sub() {  # job name, first job, queued follow-up, search root, make_searches arguments
  local b
  if [ "$(squeue -h -j "$3" -o %T 2>/dev/null)" != PENDING ]; then
    echo "$1: $3 is not pending, left alone"
    return
  fi
  scancel "$3"
  b=$(sbatch --parsable --qos=standard --time=2-00:00:00 --job-name="$1" --dependency=afterany:"$2" \
       --export=ALL,SEARCH_ROOT="$S/$4",SETUP="$5",MAX_ITERS=100 "$L")
  echo "$1: $3 cancelled, $b queued after $2"
}

unset MOBO_OBJECTIVES
sub eai_node1b 148500 148501 node1 ""
sub eai_node2b 148502 148503 node2 "--warm 0 --fresh 6 --first_seed 13"
sub eai_node4b 148537 148538 node4 "--warm 0 --fresh 6 --first_seed 19"
sub eai_node7b 148548 148549 node7 "--warm 0 --fresh 6 --first_seed 25"
export MOBO_OBJECTIVES=fwhm_weighted_mean,asci_pct_fwhm0p45,ppds_ring1,cnr_sector_mean
sub eai_node3b 148540 148541 node3_armB2 "--warm 3 --fresh 0 --prefix armB2_"
sub eai_node5b 148544 148545 node5_armB2 "--warm 0 --fresh 3 --first_seed 13 --prefix armB2_"
sub eai_node6b 148546 148547 node6_armB2 "--warm 0 --fresh 3 --first_seed 16 --prefix armB2_"
