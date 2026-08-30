#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir="${VANTA_VERIFY_BUILD_DIR:-$root/build/host}"

python3 -m pytest "$root/tests" -q
cmake -S "$root" -B "$build_dir" -DVANTA_ENABLE_CUDA=OFF
cmake --build "$build_dir" --parallel
ctest --test-dir "$build_dir" --output-on-failure

printf 'host verification passed: %s\n' "$build_dir"
