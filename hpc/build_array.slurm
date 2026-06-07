#!/usr/bin/env bash
# Sharded Earth Engine data build as a Slurm job array.
# USE WHEN: compute nodes have internet (confirm with hpc/check_env.sh).
# Each array task builds a disjoint slice of patches into checkpointed shards;
# run hpc/merge.slurm afterwards to concatenate them.
#
#SBATCH --job-name=agni-build
#SBATCH --array=0-63                # EDIT: number of shard tasks (0..N-1)
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_%A_%a.out
##SBATCH --partition=cpu           # EDIT: uncomment/set your partition
##SBATCH --account=def-yourpi      # EDIT: DRAC allocation (def-<pi>) or Vector account
# DRAC: run the build only on Fir/Nibi (compute nodes have internet for Earth Engine).
set -euo pipefail

REPO="${REPO:-$HOME/agni-modern-aaai}"
CONFIG="${CONFIG:-configs/experiments/national_aaai.yaml}"   # EDIT: target config
mkdir -p logs
cd "$REPO"

module load python/3.12 2>/dev/null || true   # EDIT: your module name
source .venv/bin/activate

# EDIT: service-account auth so the array runs unattended.
export EARTHENGINE_KEY_FILE="${EARTHENGINE_KEY_FILE:-$HOME/.secrets/agni-ee-key.json}"
export EARTHENGINE_PROJECT="${EARTHENGINE_PROJECT:-your-gcp-project}"

# Build the grid once (idempotent; safe to call from every task).
python scripts/build_grid.py "$CONFIG"

# Keep --max-workers modest: total EE concurrency = array_size * max_workers.
# Too high => EE throttling (watch for "Transient error ... retrying" in logs).
python scripts/build_dataset.py "$CONFIG" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --max-workers 6 \
    --max-retries 5
