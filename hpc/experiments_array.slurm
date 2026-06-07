#!/usr/bin/env bash
# Run many experiments in parallel as a job array (occurrence/severity/risk x
# regions x seeds, or baseline comparisons). One array task per line in
# hpc/experiments.txt. Needs NO internet: reads already-built labeled parquet.
#
# Set --array to match the number of lines in hpc/experiments.txt, e.g. a file
# with 6 lines -> --array=0-5.
#
#SBATCH --job-name=agni-exp
#SBATCH --array=0-5                 # EDIT: 0..(num lines - 1)
#SBATCH --gres=gpu:1               # EDIT: drop for tree-only experiments
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/exp_%A_%a.out
##SBATCH --partition=gpu           # EDIT (DRAC: --gpus-per-node=h100:1; Vector: -p a40)
##SBATCH --account=def-yourpi      # EDIT: DRAC allocation (def-<pi>) or Vector account
set -euo pipefail

REPO="${REPO:-$HOME/agni-modern-aaai}"
LIST="${LIST:-hpc/experiments.txt}"
mkdir -p logs
cd "$REPO"

module load python/3.12 2>/dev/null || true   # EDIT
source .venv/bin/activate

# Pick the config for this array index (skip blank lines / comments).
CONFIG="$(grep -vE '^\s*(#|$)' "$LIST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")"
if [[ -z "${CONFIG:-}" ]]; then
    echo "FATAL: no config for array index $SLURM_ARRAY_TASK_ID in $LIST"; exit 1
fi
echo "Array $SLURM_ARRAY_TASK_ID -> $CONFIG"

python scripts/train.py "$CONFIG"
python scripts/evaluate.py "$CONFIG"
