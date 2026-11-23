#!/bin/bash -l
#SBATCH --job-name=bootcamp_labs
#SBATCH --account=su_ryao1_develop
#SBATCH --partition=alpha
#SBATCH --qos=priority
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=mlem_profile/logs/bootcamp_labs_%j.out
# NVIDIA Profiling AI Software Bootcamp, Labs 1 and 3, as the notebooks run them, with
# "srun -p primary" replaced by this allocation (Empire AI partition alpha). Submit from $HOME.
# the Python/3.10.15 module needs libcrypt.so.2, absent on GPU nodes: use uv's standalone Python 3.12
source ~/venvs/gpu312/bin/activate
W=~/Profiling-AI-Software-Bootcamp/workspace
mkdir -p $W/reports $W/../lab_outputs
O=$W/../lab_outputs
cd $W/source_code
chmod u+w saved_models   # unzipped read-only, so torch.save failed on 10 Nov

echo "== Lab 1: system topology (GPUs $CUDA_VISIBLE_DEVICES)"
nvidia-smi topo -p2p r | tee $O/lab1_p2p_read.txt
nvidia-smi topo -p2p n | tee $O/lab1_p2p_nvlink_4gpu.txt
nvidia-smi topo -m | tee $O/lab1_topo_matrix.txt

TR="torchrun --nproc_per_node=4 --nnodes=1 --standalone --master_addr=localhost --master_port=1234"
NSYS="nsys profile --trace cuda,nvtx --sample=none --cpuctxsw=none --capture-range cudaProfilerApi --cuda-flush-interval=0 --force-overwrite true"
clean() { pkill -u $USER -f "$1" 2>/dev/null; sleep 5; }   # leftover ranks would join the next attempt's rendezvous
prof() {   # name script: the notebook's way (one report over torchrun); if that fails, one report per rank
  # on 10 Nov, under nsys (osrt traced, a failed GPU-metrics attempt first) DDP init failed and the ranks hung 10 min in NCCL
  if timeout 4m $NSYS --output $W/reports/$1 $TR $2 > $O/lab3_$1.log 2>&1 && [ -f $W/reports/$1.nsys-rep ]; then r=$W/reports/$1
  else clean $2; echo "  $1: whole-torchrun profile failed (lab3_$1.log); profiling each rank"
    timeout 4m $TR --no-python $NSYS --output "$W/reports/${1}_rank%q{LOCAL_RANK}" python $2 > $O/lab3_${1}_perrank.log 2>&1
    clean $2; r=$W/reports/${1}_rank0; fi
  ls -la $r.nsys-rep 2>/dev/null || echo "  $1: no report"
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum --format csv --output $O/stats_$1 $r.nsys-rep > /dev/null 2>&1
}

echo "== Lab 3: baseline DDP training, 4 GPUs"
/usr/bin/time -v $TR ddp_baseline.py > $O/lab3_baseline.log 2>&1; tail -5 $O/lab3_baseline.log
echo "== Lab 3: Nsight profile of the baseline (NVTX)"
prof baseline_nvtx ddp-baseline_nvtx.py; tail -3 $O/lab3_baseline_nvtx.log
echo "== Lab 3: Nsight profile of the first optimization"
prof firstOptim ddp_optimize.py; tail -3 $O/lab3_firstOptim.log
echo "== Lab 3: final optimized run"
/usr/bin/time -v $TR ddp_run_optimize_update.py > $O/lab3_optimized.log 2>&1; tail -5 $O/lab3_optimized.log
grep -h "Elapsed (wall clock)" $O/lab3_baseline.log $O/lab3_optimized.log
ls -la $W/reports $O
echo LABS_DONE
