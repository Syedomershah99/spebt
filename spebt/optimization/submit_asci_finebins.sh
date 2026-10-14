#!/bin/bash
# Recompute ASCI at 0.1 degree for every config with surviving beam files,
# then backfill the column into every archive.
#
# The 1-degree column saturates (87% of designs at 100%) and cannot rank
# designs. This adds the finer column alongside it; nothing is overwritten.
#
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --cpus-per-task=4
# A boolean coverage grid at 3600 bins is 40000 x 3600 = 144 MB, two live at
# once. 48G is generous and queues quickly.
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --job-name=asci_fine
#SBATCH --mail-user=syedomer@buffalo.edu
#SBATCH --mail-type=FAIL,TIMEOUT
#SBATCH --output=results/slurm_logs/out/asci_fine_%j.out
#SBATCH --error=results/slurm_logs/err/asci_fine_%j.err

set -euo pipefail
source "${SPEBT_VENV:-$HOME/spebt-venv}/bin/activate"
cd /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/optimization
mkdir -p results/slurm_logs/out results/slurm_logs/err

echo "=== computing ASCI at 0.1 degree ==="
python compute_asci_finebins.py --bins 3600 --out results/asci_finebins.csv

echo
echo "=== backfilling into every archive ==="
for CSV in results/results_summary_mobo.csv ../results_h2h_*/results_summary_mobo.csv \
           results_h2h_*/results_summary_mobo.csv; do
  [ -f "$CSV" ] || continue
  echo "--- $CSV"
  python backfill_mpxi_variants.py --results_csv "$CSV" \
      --variants_csv results/asci_finebins.csv
done

echo
echo "=== refreshing the archive backup ==="
bash ../archive_backup/refresh.sh
