#!/bin/bash -l
# Empire AI validation, step 2, node A (approved by Omer on 25 Oct 2026; one of
# two test-QoS node jobs, about 0.5 SU). On one whole Grace node:
#   1. run the full pipeline for six designs at once, as production would pack
#      them, with the PPDF cache on: the benchmark, the MIC design (nd2fine_960)
#      and four designs with CCR repeats (validation_refs.csv); the node's load
#      and memory are logged every 30 s
#   2. as soon as the benchmark and nd2fine_960 are done, reconstruct both over
#      noise seeds 0-19, every iteration 1-400 (run_iter_count_pair.py), for the
#      statistical comparison with CCR
# Metrics land in results_step2a.csv; the comparison is done afterwards.
#
# Submit (WORK_ROOT = where the six work directories go, scratch if possible):
#   sbatch --export=ALL,WORK_ROOT=/ddn/scratch/... step2_node_a.sh
#
#SBATCH --job-name=eai_step2a
#SBATCH --account=su_ryao1_develop
#SBATCH --partition=grace
#SBATCH --qos=test
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/projects/su/ryao1/develop/oshah/logs/step2a_%j.out

set -uo pipefail
P=/projects/su/ryao1/develop/oshah
CODE=$P/spebt
WORK_ROOT=${WORK_ROOT:?set WORK_ROOT with sbatch --export}
OUT=$P/step2a_${SLURM_JOB_ID}
mkdir -p "$OUT" "$WORK_ROOT"
export SPEBT_VENV=$HOME/venvs/spebt-arm64 CODE_DIR=$CODE
export PHANTOM_PATH=$CODE/data/sai_10mm/hot_rods_phantom_10.0_mm_x_10.0_mm.pt
export CNR_CACHE_PPDFS=1 CUDA_VISIBLE_DEVICES="" HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export RESULTS_CSV=$OUT/results_step2a.csv
T0=$(date +%s)
echo "step 2 node A on $(hostname) at $(date), code $(cat $CODE/COMMIT), work in $WORK_ROOT"
lscpu | grep -E "Model name|^CPU\(s\)"

( while true; do
    echo "$(( $(date +%s) - T0 )) $(cut -d' ' -f1 /proc/loadavg) $(free -g | awk '/Mem:/ {print $3}')"
    sleep 30
  done ) > "$OUT/node_usage.txt" &
MON=$!

# 1. Six pipelines at once.
declare -A PID
while IFS=, read -r name src diam nap nd1 nd2 d2 d3 rest; do
  [ "$name" = config ] && continue
  ( export CONFIG_NAME=$name WORK_DIR=$WORK_ROOT/$name APERTURE_DIAM=$diam \
           N_APERTURES=${nap%.0} N_DET_RING1=$nd1 N_DET_RING2=$nd2 D2_INNER=$d2 D3_INNER=$d3 \
           A_MM=0.2 B_MM=0.2
    s=$(date +%s)
    bash "$CODE/optimization/run_sai_pipeline.sh" > "$OUT/$name.log" 2>&1
    echo "pipeline $name exit $? after $(( $(date +%s) - s )) s" >> "$OUT/timings.txt" ) &
  PID[$name]=$!
  echo "launched $name"
done < "$P/validation_refs.csv"

# 2. Twenty seeds of the two references once their pipelines are done.
wait "${PID[tmi_reference_000]}" "${PID[nd2fine_960]}"
echo "references done after $(( $(date +%s) - T0 )) s; starting the 20-seed runs"
source "$SPEBT_VENV/bin/activate"
run_seed() {
  OMP_NUM_THREADS=2 python "$CODE/optimization/run_iter_count_pair.py" --work_dir "$WORK_ROOT/$1" \
    --phantom_path "$PHANTOM_PATH" --out_subdir "iterpair_N400_T10_seed$2" --seed "$2" \
    --iterations 400 --T_sec 10 > "$OUT/seeds_$1_$2.log" 2>&1 \
    && echo "seed run $1 $2 ok" || echo "seed run $1 $2 FAILED"
}
export -f run_seed
export CODE WORK_ROOT OUT
for d in tmi_reference_000 nd2fine_960; do
  for s in $(seq 0 19); do echo "$d $s"; done
done | xargs -P 20 -n 2 bash -c 'run_seed "$0" "$1"'
# The other four pipelines; never a bare wait, which would also wait on the monitor.
wait "${PID[@]}"
kill $MON 2>/dev/null

# PPDF summary statistics for the two references, to compare with the same
# statistics computed on CCR (the files themselves differ in float rounding).
python - "$WORK_ROOT" > "$OUT/ppdf_stats.csv" <<'EOF'
import glob, os, sys
import h5py, numpy as np
print("design,file,shape,sum,max,nonzero")
for d in ("tmi_reference_000", "nd2fine_960"):
    for f in sorted(glob.glob(os.path.join(sys.argv[1], d, "position_*_ppdfs_t8_*.hdf5"))):
        with h5py.File(f, "r") as h:
            a = h["ppdfs"][:]
        print(f"{d},{os.path.basename(f)},{'x'.join(map(str, a.shape))},{a.sum(dtype=np.float64):.10e},"
              f"{a.max():.8e},{int(np.count_nonzero(a))}")
EOF

# Copy the small outputs off scratch, which the login node cannot see.
for d in "$WORK_ROOT"/*/; do
  n=$(basename "$d")
  mkdir -p "$OUT/runs/$n"
  cp "$d"/*.tensor "$OUT/runs/$n/" 2>/dev/null
  for r in "$d"iterpair_* "$d"cnr_inloop; do
    [ -d "$r" ] || continue
    mkdir -p "$OUT/runs/$n/$(basename "$r")"
    cp "$r"/dense_cnr.csv "$r"/cnr_results.npz "$r"/projections_T8.npy "$OUT/runs/$n/$(basename "$r")/" 2>/dev/null
  done
done
du -sh "$OUT"
cat "$OUT/timings.txt"
echo "step 2 node A done after $(( $(date +%s) - T0 )) s at $(date)"
sacct -j $SLURM_JOB_ID -X -n --format=JobID,Partition,QOS,AllocTRES%60,Elapsed
