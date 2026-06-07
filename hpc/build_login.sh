#!/usr/bin/env bash
# Fallback build for clusters where compute nodes have NO internet.
# Run this ON A LOGIN or DATA-TRANSFER node, inside tmux/screen so it survives
# disconnects. It is checkpointed: if it dies, just re-run the same command.
#
# Usage:
#   tmux new -s agni-build
#   bash hpc/build_login.sh
#   # detach with Ctrl-b then d ; reattach later with: tmux attach -t agni-build
set -euo pipefail

REPO="${REPO:-$HOME/agni-modern-aaai}"
CONFIG="${CONFIG:-configs/experiments/kalimantan_pilot.yaml}"   # EDIT: target config
cd "$REPO"

module load python/3.12 2>/dev/null || true   # EDIT: your module name
source .venv/bin/activate

export EARTHENGINE_KEY_FILE="${EARTHENGINE_KEY_FILE:-$HOME/.secrets/agni-ee-key.json}"
export EARTHENGINE_PROJECT="${EARTHENGINE_PROJECT:-your-gcp-project}"

# Be a good citizen on the shared login node: keep workers low.
python scripts/build_grid.py "$CONFIG"
python scripts/build_dataset.py "$CONFIG" --max-workers 4 --max-retries 5
echo "Build complete. Next: sbatch hpc/merge.slurm (or run merge/enrich/labels here)."
