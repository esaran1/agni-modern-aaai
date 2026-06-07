#!/usr/bin/env bash
# Run this INSIDE an interactive job (salloc) to learn what your cluster allows.
# It is read-only and safe. It answers the make-or-break questions:
#   1. Do compute nodes have internet?
#   2. Does Earth Engine initialize here?
#   3. Does the package import and test cleanly?
set -uo pipefail

REPO="${REPO:-$HOME/agni-modern-aaai}"   # EDIT if your repo lives elsewhere
cd "$REPO" || { echo "FATAL: repo not found at $REPO"; exit 1; }

echo "== Node: $(hostname) =="

echo "== 1. Outbound internet to Earth Engine? =="
if curl -sSf -m 15 -I https://earthengine.googleapis.com >/dev/null 2>&1; then
    echo "RESULT: compute node HAS internet -> use hpc/build_array.slurm"
else
    echo "RESULT: NO internet from this node -> build on login node (hpc/build_login.sh) or off-cluster"
fi

echo "== 2. Python env =="
# EDIT: match your cluster's module name, or remove if you manage python yourself.
module load python/3.12 2>/dev/null || echo "(no module system / adjust module name)"
source .venv/bin/activate 2>/dev/null || { echo "FATAL: .venv missing; create it first"; exit 1; }
python --version

echo "== 3. Earth Engine init (uses EARTHENGINE_KEY_FILE if set) =="
python - <<'PY'
from agni.data.sources.ee_session import initialize_earth_engine
ok = initialize_earth_engine(high_volume=True)
print("EE initialized:", ok)
if ok:
    import ee
    print("EE sanity (1+1):", ee.Number(1).add(1).getInfo())
PY

echo "== 4. Test suite =="
pytest -q
