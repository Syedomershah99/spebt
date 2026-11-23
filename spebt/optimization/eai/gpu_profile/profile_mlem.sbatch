#!/bin/bash -l
#SBATCH --job-name=mlem_nsys
#SBATCH --account=su_ryao1_develop
#SBATCH --partition=alpha
#SBATCH --qos=priority
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=mlem_profile/logs/mlem_nsys_%j.out
# ML-EM of the MIC design (nd2fine_960) on one GPU (Alpha sends every 1-GPU job to an RTX PRO 6000): timings over 150 iterations for
# the campaign code and the GPU-resident version, then a 10-iteration Nsight Systems
# report of each. Submit from $HOME.
set -e
# uv's Python 3.12 (the Python/3.10.15 module needs libcrypt.so.2, absent on GPU nodes) with torch
# 2.7.1+cu128, whose sm_120 kernels the RTX PRO 6000 needs (torch 2.6+cu124 stops at sm_90)
source ~/venvs/gpu312cu128/bin/activate
cd ~/mlem_profile
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nsys --version
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, 'cuda ok', torch.cuda.is_available())"
ARGS="--ppdf_dir $HOME/mlem_profile/nd2fine_960 --phantom /projects/su/ryao1/develop/oshah/code_a7e678e/recon/hot_rods_phantom_10.0_mm_x_10.0_mm.pt"
python profile_mlem.py $ARGS --out run_150 --mode both --iters 150
mkdir -p reports
for m in campaign resident; do
  nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite true -o reports/mlem_${m}_10it \
    python profile_mlem.py $ARGS --out run_nsys_$m --mode $m --iters 10
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum \
    --format csv --output reports/stats_${m}_10it reports/mlem_${m}_10it.nsys-rep
done
ls -la reports
echo PROFILE_DONE
