#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  echo "Python 3.10 is required. Create the environment first:" >&2
  echo "  conda env create -f environment/environment.yml" >&2
  exit 2
fi

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install -r "$ROOT/environment/requirements.txt"

# These extensions are intentionally built after Torch is installed. Shipping
# a server-specific .so would silently couple the release to one CUDA ABI.
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install --no-build-isolation -v \
  "$ROOT/third_party/pytorch3d_simplified"
MAX_JOBS="${MAX_JOBS:-8}" python -m pip install --no-build-isolation -v \
  "$ROOT/third_party/pointnet2_ops"

python -m pip install -e "$ROOT/PointSAM" --no-deps
python -m pip install -e "$ROOT/R3D" --no-deps

python "$ROOT/environment/verify_environment.py"
echo "Core environment installed. Add benchmark runtimes with environment/install_benchmarks.sh."
