#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
TARGET="${1:-all}"

R3D_BENCHMARK_COMMIT="e637c0148376ddc4b5e667fa8f8e108cb8ff7a85"
ROBOTWIN_COMMIT="74f4e99720b4b296a38af9e95ee31d9e400073af"

clone_at() {
  local url="$1" commit="$2" path="$3"
  if [[ ! -d "$path/.git" ]]; then
    git clone --filter=blob:none "$url" "$path"
  fi
  git -C "$path" fetch origin "$commit"
  git -C "$path" checkout --detach "$commit"
}

install_r3d_benchmarks() {
  local path="$THIRD_PARTY/r3d_benchmarks"
  clone_at "https://github.com/Wushr-Lance/R3D-Policy.git" "$R3D_BENCHMARK_COMMIT" "$path"
  python -m pip install -e "$path/third_party/Metaworld"
  python -m pip install -e "$path/third_party/rrl-dependencies/mj_envs"
  python -m pip install -e "$path/third_party/rrl-dependencies/mjrl"
  echo "export R3D_VRL3_ROOT=$path/third_party/VRL3"
}

install_maniskill2() {
  python -m pip install "mani-skill2==0.5.3"
  python -m mani_skill2.utils.download_asset PickCube-v0 || true
  python -m mani_skill2.utils.download_asset StackCube-v0 || true
  python -m mani_skill2.utils.download_asset PegInsertionSide-v0 || true
}

install_robotwin2() {
  local path="$THIRD_PARTY/robotwin2"
  clone_at "https://github.com/RoboTwin-Platform/RoboTwin.git" "$ROBOTWIN_COMMIT" "$path"
  echo "RoboTwin code pinned at $ROBOTWIN_COMMIT."
  echo "Download assets with the commands documented by that checkout, then set:"
  echo "export ROBOTWIN2_ROOT=$path"
}

mkdir -p "$THIRD_PARTY"
case "$TARGET" in
  adroit|metaworld) install_r3d_benchmarks ;;
  maniskill2) install_maniskill2 ;;
  robotwin2) install_robotwin2 ;;
  all)
    install_r3d_benchmarks
    install_maniskill2
    install_robotwin2
    ;;
  *) echo "usage: $0 [all|adroit|metaworld|maniskill2|robotwin2]" >&2; exit 2 ;;
esac
