#!/bin/bash -l
# Empire AI validation, step 1 (approved by Omer on 25 Oct 2026, about 0.19 SU).
# On one Grace node, with 8 cores for at most 45 minutes:
#   - pick an aarch64 Python 3.10 (the module if it runs here, else the
#     standalone build fetched to ~/opt) and build ~/venvs/spebt-arm64 offline
#     from ~/wheels/arm64 (CCR's package versions, CPU-only torch 2.13.0)
#   - run the test suite
#   - rebuild the benchmark's and the MIC design's geometry and compare it with
#     CCR's .tensor files
#   - test flock on scratch and on project space (the controller locks its
#     results folder)
#   - record the node, and the job's billing as Slurm records it
# Every result is a line starting "CHECK" in the log.
#
# Submit from the Alpha login node:
#   sbatch /projects/su/ryao1/develop/oshah/spebt/optimization/eai/step1_grace_check.sh
#
#SBATCH --job-name=eai_step1
#SBATCH --account=su_ryao1_develop
#SBATCH --partition=grace
#SBATCH --qos=test
#SBATCH --exclude=alphagh01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/projects/su/ryao1/develop/oshah/logs/step1_%j.out

set -uo pipefail
P=/projects/su/ryao1/develop/oshah
CODE=$P/spebt
export CUDA_VISIBLE_DEVICES="" HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
echo "step 1 on $(hostname) at $(date), code $(cat $CODE/COMMIT)"

echo "== node"
uname -m; nproc; free -g | head -2
lscpu | grep -E "Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|NUMA node\(s\)"

echo "== python"
if module load Python/3.10.15 2>/dev/null && python3 -c "import sys; assert sys.version_info[:2] == (3, 10)" 2>/dev/null; then
  PYARM=$(command -v python3)
  echo "CHECK python: the Python/3.10.15 module runs here ($PYARM)"
else
  module unload Python 2>/dev/null || true
  PYARM=$HOME/opt/python310-aarch64/bin/python3.10
  echo "CHECK python: the module does not run here; using the standalone $($PYARM -V 2>&1)"
fi

echo "== venv"
rm -rf ~/venvs/spebt-arm64
"$PYARM" -m venv ~/venvs/spebt-arm64 && source ~/venvs/spebt-arm64/bin/activate
pip install -q --no-index --find-links ~/wheels/arm64 -U pip setuptools wheel
if pip install -q --no-index --find-links ~/wheels/arm64 -r $P/requirements_arm64.txt; then
  echo "CHECK venv: installed"
else
  echo "CHECK venv: FAILED"
fi
pip freeze > $P/logs/step1_arm64_freeze_${SLURM_JOB_ID}.txt
if diff <(sed 's/^torch==2.13.0$/torch==2.13.0+cpu/' $P/requirements_arm64.txt | sort) \
        <(sort $P/logs/step1_arm64_freeze_${SLURM_JOB_ID}.txt); then
  echo "CHECK freeze: identical to CCR's versions (CPU-only torch)"
else
  echo "CHECK freeze: DIFFERS (lines above)"
fi
python -c "import torch, numpy, botorch, h5py; print('torch', torch.__version__, torch.backends.cpu.get_cpu_capability(), 'threads', torch.get_num_threads(), '| numpy', numpy.__version__, '| botorch', botorch.__version__, '| h5py', h5py.__version__)"

echo "== tests"
cd $CODE
# Temporary files on project space: a node's /tmp may be smaller than the 60 GB
# the controller insists on before it starts (TestLocalExecutor runs it).
T=$P/logs/pytest_tmp_${SLURM_JOB_ID}
timeout 20m python -m pytest tests/test_pipeline.py -q -p no:cacheprovider --basetemp=$T \
  > $P/logs/step1_pytest_${SLURM_JOB_ID}.txt 2>&1
tail -15 $P/logs/step1_pytest_${SLURM_JOB_ID}.txt
echo "CHECK tests: $(grep -E '[0-9]+ (passed|failed|error)' $P/logs/step1_pytest_${SLURM_JOB_ID}.txt | tail -1)"
rm -rf $T

echo "== geometry"
G=$P/step1_geom_${SLURM_JOB_ID}
mkdir -p $G/tmi_reference_000 $G/nd2fine_960
python $CODE/geometry/generate_mph_scanner_circularfov.py --aperture_diam 0.4 --n_apertures 180 \
  --scint_radial_mm 6.0 --ring_thickness 2.5 --n_det_ring1 480 --n_det_ring2 720 \
  --d2_inner 390.0 --d3_inner 520.0 --output_dir $G/tmi_reference_000 > $G/tmi_reference_000.log 2>&1
python $CODE/geometry/generate_mph_scanner_circularfov.py --aperture_diam 0.2717795628467184 \
  --n_apertures 211 --scint_radial_mm 6.0 --ring_thickness 2.5 --n_det_ring1 582 --n_det_ring2 960 \
  --d2_inner 393.1584022063654 --d3_inner 492.1966089700369 --output_dir $G/nd2fine_960 > $G/nd2fine_960.log 2>&1
for d in tmi_reference_000 nd2fine_960; do
  new=$(ls $G/$d/*.tensor 2>/dev/null | head -1)
  ref=$(ls $P/ref_geom/$d/*.tensor | head -1)
  if [ -z "$new" ]; then
    echo "CHECK geometry $d: NOT GENERATED"; tail -5 $G/$d.log
  else
    echo "CHECK geometry $d: $(python $CODE/optimization/eai/compare_tensors.py "$new" "$ref" 2>&1 | tail -1)"
    [ "$(basename $new)" = "$(basename $ref)" ] && echo "  same file name (same base-scanner hash)"
  fi
done

echo "== flock"
SCR=$(ls -d /ddn/scratch/*/oshah /ddn/scratch/*/*/oshah 2>/dev/null | head -1)
echo "scratch candidates: $(ls -d /ddn/scratch/* 2>/dev/null | head -5 | tr '\n' ' ')"
[ -n "$SCR" ] && echo "user scratch: $SCR" && (lfs quota -h -u oshah /ddn/scratch 2>/dev/null | tail -2)
for d in ${SCR:-} $P; do
  t="$d/.flock_test_$SLURM_JOB_ID"
  python - "$t" <<'EOF'
import fcntl, os, subprocess, sys
path = sys.argv[1]
f = open(path, "w")
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as e:
    print(f"CHECK flock {os.path.dirname(path)}: cannot lock ({e})")
    sys.exit(0)
code = ("import fcntl,sys\nf=open(sys.argv[1],'w')\ntry:\n fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        " print('second lock GRANTED')\nexcept OSError as e:\n print('second lock refused', e.errno)")
r = subprocess.run([sys.executable, "-c", code, path], capture_output=True, text=True)
ok = "refused" in r.stdout
print(f"CHECK flock {os.path.dirname(path)}: {'works (a second process is refused)' if ok else 'BROKEN: ' + r.stdout.strip()}")
os.remove(path)
EOF
done

echo "== billing"
sacct -j $SLURM_JOB_ID -X -n --format=JobID,Partition,QOS,AllocTRES%60,Elapsed
echo "step 1 done at $(date)"
