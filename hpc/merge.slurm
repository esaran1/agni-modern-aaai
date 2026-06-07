#!/usr/bin/env bash
# Merge shards -> features -> labels. CPU-only; needs NO internet for occurrence.
# (Severity/risk label builds DO call Earth Engine; see note below.)
#
#SBATCH --job-name=agni-merge
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                  # EDIT: scale to dataset size
#SBATCH --time=4:00:00
#SBATCH --output=logs/merge_%j.out
##SBATCH --partition=cpu           # EDIT
##SBATCH --account=def-yourpi      # EDIT: DRAC allocation (def-<pi>) or Vector account
set -euo pipefail

REPO="${REPO:-$HOME/agni-modern-aaai}"
CONFIG="${CONFIG:-configs/experiments/national_aaai.yaml}"   # EDIT
mkdir -p logs
cd "$REPO"

module load python/3.12 2>/dev/null || true   # EDIT
source .venv/bin/activate

# 1. Concatenate all per-patch shards into dataset.parquet (no EE access).
python scripts/build_dataset.py "$CONFIG" --merge

# 2. Feature engineering (offline).
python scripts/enrich_features.py "$CONFIG"

# 3. Labels. Occurrence is offline. For severity/risk configs this calls EE,
#    so it must run where there IS internet (export the key env vars first).
export EARTHENGINE_KEY_FILE="${EARTHENGINE_KEY_FILE:-$HOME/.secrets/agni-ee-key.json}"
export EARTHENGINE_PROJECT="${EARTHENGINE_PROJECT:-your-gcp-project}"
python scripts/build_labels.py "$CONFIG"
