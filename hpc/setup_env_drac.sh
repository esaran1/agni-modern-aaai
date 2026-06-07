#!/usr/bin/env bash
# Environment setup for Digital Research Alliance of Canada (DRAC) clusters.
# RUN THIS ON A LOGIN NODE (compute nodes cannot reach PyPI).
#
#   bash hpc/setup_env_drac.sh
#
# DRAC convention: build a --no-download virtualenv and install from the cluster
# wheelhouse with --no-index. A few packages (earthengine-api, osmnx) are usually
# NOT in the wheelhouse, so we install those normally (login node has internet).
set -euo pipefail

REPO="${REPO:-$HOME/agni-modern-aaai}"
PY_MODULE="${PY_MODULE:-python/3.12}"   # EDIT: `module avail python` to confirm
cd "$REPO"

module load "$PY_MODULE"
virtualenv --no-download .venv
source .venv/bin/activate
pip install --no-index --upgrade pip

# Try the fast path first: everything from the wheelhouse.
if pip install --no-index -e '.[dev]'; then
    echo "Installed fully from the DRAC wheelhouse."
else
    echo "Wheelhouse install failed (version pin or missing wheel)."
    echo "Falling back to PyPI for packages not in the wheelhouse (login node has internet)."
    # Install the EE/OSM packages from PyPI, then retry the project install.
    pip install earthengine-api osmnx || true
    pip install -e '.[dev]'
fi

echo "Done. Verify with: pytest -q"
