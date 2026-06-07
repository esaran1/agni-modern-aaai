#!/usr/bin/env bash
# Train + evaluate a single experiment. This is the part where HPC actually helps
# (GPU for the transformer / joint-risk models). Needs NO internet: it reads the
# already-built labeled parquet.
#
#SBATCH --job-name=agni-train
#SBATCH --gres=gpu:1               # EDIT: drop this line for tree-model-only configs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/train_%j.out
##SBATCH --partition=gpu           # EDIT (DRAC: --gpus-per-node=h100:1; Vector: -p a40)
##SBATCH --account=def-yourpi      # EDIT: DRAC allocation (def-<pi>) or Vector account
set -euo pipefail

REPO="${REPO:-$HOME/agni-modern-aaai}"
CONFIG="${CONFIG:-configs/experiments/national_aaai.yaml}"   # EDIT
mkdir -p logs
cd "$REPO"

module load python/3.12 2>/dev/null || true   # EDIT
##module load cuda/12.1            # EDIT: load CUDA if your cluster requires it
source .venv/bin/activate

python scripts/train.py "$CONFIG"
python scripts/evaluate.py "$CONFIG"
