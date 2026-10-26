#!/bin/bash -l
# The follow-ups again (25 Oct 2026, 21:30). Each runs the launcher L, with the
# same search roots and settings, after its node's first job (afterany). L is
# a copy outside the live tree $P/spebt: replacing a file there kills the
# pipelines reading it on the compute nodes (stale NFS handle, 16:10), and
# sbatch keeps its own copy of L anyway. This launcher skips the starting
# designs once a search's optimizer has started, so a starting design lost
# after that point stays lost instead of being evaluated late.
# MOBO_OBJECTIVES is exported, never put inside --export: sbatch splits on commas.
P=/projects/su/ryao1/develop/oshah
L=${L:-$P/launchers/run_node_searches_frozen_start.sh}
S=/ddn/scratch/suny/oshah
grep -q "starting set is fixed once the optimizer has started" "$L" || { echo "$L is not the fixed launcher"; exit 1; }

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
sub eai_node1b 148500 148578 node1 ""
sub eai_node2b 148502 148579 node2 "--warm 0 --fresh 6 --first_seed 13"
sub eai_node4b 148537 148580 node4 "--warm 0 --fresh 6 --first_seed 19"
sub eai_node7b 148548 148581 node7 "--warm 0 --fresh 6 --first_seed 25"
export MOBO_OBJECTIVES=fwhm_weighted_mean,asci_pct_fwhm0p45,ppds_ring1,cnr_sector_mean
sub eai_node3b 148540 148582 node3_armB2 "--warm 3 --fresh 0 --prefix armB2_"
sub eai_node5b 148544 148583 node5_armB2 "--warm 0 --fresh 3 --first_seed 13 --prefix armB2_"
sub eai_node6b 148546 148584 node6_armB2 "--warm 0 --fresh 3 --first_seed 16 --prefix armB2_"
