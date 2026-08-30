#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

"$root/scripts/verify.sh"

if [[ ! -x "$(command -v nvcc 2>/dev/null || true)" ||
      ! -x "$(command -v nvidia-smi 2>/dev/null || true)" ||
      ! -x "$(command -v compute-sanitizer 2>/dev/null || true)" ]]; then
    printf 'GPU confidence suite skipped: nvcc, nvidia-smi, and compute-sanitizer are required\n'
    exit 0
fi

build_dir="${VANTA_CUDA_BUILD_DIR:-$root/build/cuda}"
out_dir="${VANTA_CONFIDENCE_OUT:-$root/results/local/confidence}"
cmake -S "$root" -B "$build_dir" -DVANTA_ENABLE_CUDA=ON
cmake --build "$build_dir" --parallel
ctest --test-dir "$build_dir" --output-on-failure
python3 "$root/tools/colab_entry.py" --out "$out_dir" --seed "${VANTA_SEED:-2026}"
python3 "$root/experiments/cuda-mutations/run_campaign.py" \
    --manifest "$root/experiments/cuda-mutations/manifest-v3.json" \
    --out "$out_dir/cuda-mutations" \
    --seed "${VANTA_SEED:-2026}"

printf 'host and GPU confidence suites passed: %s\n' "$out_dir"
