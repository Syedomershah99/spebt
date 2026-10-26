#!/bin/bash -l
# Re-measure designs over many noise seeds (ANALYSIS_PLAN.md rules 2 and 4), on
# one Grace node.
#
# DESIGNS is a file of absolute work directories, one per line, in the order to
# run them (rebuild.py writes it: the two references first, then the
# candidates). Each directory's 16 PPDF files are hard-linked into this job's
# own folder on scratch, so a prune in a live search cannot delete them mid-run,
# and opened once to check them. Then every design runs through seeds
# SEED_FIRST..SEED_LAST, every ML-EM iteration from 1 to ITER
# (run_iter_count_pair.py), PAR runs at a time. Each run's small outputs are
# copied to /projects/.../$RUN/<search>__<design>/ as soon as it finishes, and a
# run whose dense_cnr.csv is already there is skipped, so resubmitting with the
# same RUN resumes. SHARD=k/N runs only every N-th design from the k-th (k from
# 0), so N nodes can share one list and one RUN; jobs never share link folders.
#
#   sbatch --qos=standard --time=1-00:00:00 --export=ALL,CODE=$P/code_<commit>,RUN=remeasure_sel,\
#   DESIGNS=$P/remeasure_inputs/designs.txt,SEED_FIRST=0,SEED_LAST=39,SHARD=0/3 remeasure.sh
#
# CODE is required and may not be the live tree $P/spebt (replacing files there
# kills running pipelines). Seeds from 40 on are the held-out seeds (rule 4):
# they run only for the headline named in $CODE/optimization/eai/headline.txt
# (present only in a tree archived after the headline commit) and the two
# references. Defaults: T_SEC 10, ITER 400, PAR 36 (about 10 GB each). The job
# fails before any run if a design lacks 16 readable PPDF files, and at the end
# if any run has no dense_cnr.csv. Its links are removed once all of its runs
# succeeded (KEEP_LINKS=1 keeps them).
#
#SBATCH --job-name=eai_remeasure
#SBATCH --account=su_ryao1_develop
#SBATCH --partition=grace
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=/projects/su/ryao1/develop/oshah/logs/remeasure_%j.out

set -uo pipefail
P=/projects/su/ryao1/develop/oshah
CODE=${CODE:?set CODE to a code tree archived for this job, not $P/spebt}
[ "$(cd "$CODE" && pwd -P)" != "$(cd "$P/spebt" 2>/dev/null && pwd -P)" ] \
  || { echo "ERROR: CODE is the live tree $P/spebt"; exit 1; }
COMMIT=$(cat "$CODE/COMMIT") || { echo "ERROR: no $CODE/COMMIT"; exit 1; }
SCR=${SCR:-/ddn/scratch/suny/oshah}
RUN=${RUN:?set RUN, the name shared by every job of one re-measurement}
DESIGNS=${DESIGNS:?set DESIGNS to a file of work directories}
[ -s "$DESIGNS" ] || { echo "ERROR: $DESIGNS is missing or empty"; exit 1; }
SEED_FIRST=${SEED_FIRST:?} SEED_LAST=${SEED_LAST:?}
T_SEC=${T_SEC:-10} ITER=${ITER:-400} PAR=${PAR:-36}
JOB=${SLURM_JOB_ID:-local}
OUT=${OUT:-$P/$RUN}
W=$SCR/$RUN/job_$JOB
TAG=remeasure_N${ITER}_T${T_SEC}_seed
if [ -n "${SHARD:-}" ]; then
  [ "${SHARD%/*}" -ge 0 ] 2>/dev/null && [ "${SHARD%/*}" -lt "${SHARD#*/}" ] \
    || { echo "ERROR: SHARD=$SHARD is not k/N with 0 <= k < N"; exit 1; }
fi
HEAD=
if [ "$SEED_LAST" -ge 40 ]; then
  HEAD=$(head -1 "$CODE/optimization/eai/headline.txt" 2>/dev/null)
  [ -n "$HEAD" ] || { echo "ERROR: seeds from 40 on are held out: no headline.txt in $CODE"; exit 1; }
fi
mkdir -p "$W" "$OUT/logs"
if [ -f "$OUT/COMMIT" ]; then
  [ "$(cat "$OUT/COMMIT")" = "$COMMIT" ] || { echo "ERROR: $OUT was made with code $(cat "$OUT/COMMIT"), not $COMMIT"; exit 1; }
else
  echo "$COMMIT" > "$OUT/COMMIT"
fi
cp "$DESIGNS" "$OUT/designs_$JOB.txt"
export PHANTOM_PATH=$CODE/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt
export CUDA_VISIBLE_DEVICES="" HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source "${SPEBT_VENV:-$HOME/venvs/spebt-arm64}/bin/activate"
for f in "$CODE/optimization/run_iter_count_pair.py" "$PHANTOM_PATH"; do
  [ -f "$f" ] || { echo "ERROR: no $f"; exit 1; }
done
{ echo "job $JOB on $(hostname) at $(date)"; echo "CODE $CODE"; echo "COMMIT $COMMIT"
  echo "RUN $RUN DESIGNS $DESIGNS SEEDS $SEED_FIRST-$SEED_LAST T_SEC $T_SEC ITER $ITER PAR $PAR SHARD ${SHARD:-all} HEADLINE ${HEAD:-none}"
  lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)"; pip freeze 2>/dev/null; } > "$OUT/provenance_$JOB.txt"
echo "remeasure $RUN job $JOB on $(hostname) at $(date), code $COMMIT, seeds $SEED_FIRST-$SEED_LAST," \
     "T_sec $T_SEC, $ITER iterations, shard ${SHARD:-all}"

names=() bad=() seen=" " i=0
while IFS= read -r d || [ -n "$d" ]; do
  case "$d" in ''|'#'*) continue ;; esac
  [ "${d#/}" != "$d" ] || { echo "ERROR: not an absolute path: $d"; exit 1; }
  n="$(basename "$(dirname "$d")")__$(basename "$d")"     # search__design: unique
  case "$seen" in *" $n "*) continue ;; esac              # listed twice: run it once
  seen="$seen$n "
  i=$((i + 1))
  [ -z "${SHARD:-}" ] || [ $(((i - 1) % ${SHARD#*/})) -eq "${SHARD%/*}" ] || continue
  case "$n" in "$HEAD"|step2a__tmi_reference_000|step2a__nd2fine_960) ;;
    *) [ -z "$HEAD" ] || { echo "ERROR: $n is not the headline or a reference; seeds from 40 on are held out"; exit 1; } ;;
  esac
  mkdir -p "$W/$n"
  ln -f "$d"/position_*_ppdfs_t8_*.hdf5 "$W/$n/"
  k=$(ls "$W/$n"/position_*_ppdfs_t8_*.hdf5 2>/dev/null | wc -l)
  echo "$n: $k PPDF files linked from $d"
  [ "$k" -eq 16 ] || bad+=("$n ($k files)")
  names+=("$n")
done < "$DESIGNS"
[ ${#names[@]} -gt 0 ] || { echo "ERROR: no design selected from $DESIGNS (shard ${SHARD:-all})"; exit 1; }
if [ ${#bad[@]} -gt 0 ]; then
  printf 'ERROR: not 16 PPDF files, nothing was run: %s\n' "${bad[@]}"
  exit 1
fi
python - "$W" <<'EOF' || { echo "ERROR: unreadable PPDF files, nothing was run"; exit 1; }
import glob, os, sys
import h5py
bad = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "position_*_ppdfs_t8_*.hdf5"))):
    try:
        with h5py.File(f, "r") as h:
            if h["ppdfs"].ndim != 2 or h["ppdfs"].size == 0:
                bad.append(f)
    except Exception:
        bad.append(f)
for f in bad:
    print("unreadable:", f)
sys.exit(1 if bad else 0)
EOF

run_seed() {   # design, seed
  local o=$OUT/$1/$TAG$2 r=$W/$1/$TAG$2
  [ -s "$o/dense_cnr.csv" ] && { echo "have $1 $2"; return; }
  if OMP_NUM_THREADS=2 python "$CODE/optimization/run_iter_count_pair.py" --work_dir "$W/$1" \
       --phantom_path "$PHANTOM_PATH" --out_subdir "$TAG$2" --seed "$2" --iterations "$ITER" \
       --T_sec "$T_SEC" > "$OUT/logs/$1_$TAG$2.txt" 2>&1 \
     && mkdir -p "$o" && cp "$r/cnr_results.npz" "$o/" && cp "$r/dense_cnr.csv" "$o/dense_cnr.tmp" \
     && mv "$o/dense_cnr.tmp" "$o/dense_cnr.csv"; then
    rm -rf "$r"
    echo "ok $1 $2"
  else
    echo "FAILED $1 $2"
  fi
}
export -f run_seed
export CODE W OUT TAG ITER T_SEC
for n in "${names[@]}"; do
  for s in $(seq "$SEED_FIRST" "$SEED_LAST"); do echo "$n $s"; done
done | xargs -r -P "$PAR" -n 2 bash -c 'run_seed "$0" "$1"' | tee -a "$OUT/runs_$JOB.log"

missing=()
for n in "${names[@]}"; do
  for s in $(seq "$SEED_FIRST" "$SEED_LAST"); do
    [ -s "$OUT/$n/$TAG$s/dense_cnr.csv" ] || missing+=("$n seed $s")
  done
done
echo "done at $(date): $(( ${#names[@]} * (SEED_LAST - SEED_FIRST + 1) - ${#missing[@]} )) of" \
     "$(( ${#names[@]} * (SEED_LAST - SEED_FIRST + 1) )) runs have dense_cnr.csv in $OUT"
[ "$JOB" = local ] || sacct -j "$JOB" -X -n --format=JobID,Partition,QOS,AllocTRES%60,Elapsed
if [ ${#missing[@]} -gt 0 ]; then
  printf 'MISSING: %s\n' "${missing[@]}"
  exit 1
fi
[ -n "${KEEP_LINKS:-}" ] || rm -rf "${W:?}"
