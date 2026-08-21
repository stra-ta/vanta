# Vanta

A GPU validation laboratory for determining whether CUDA kernels are
correct, robust across launch configurations, diagnosable when they fail,
and acceptably performant under documented measurement conditions.

Vanta is not a CUDA tutorial, a BLAS replacement, a deep-learning framework,
or a kernel collection. The test framework is the primary deliverable.

## Current status

The first proof is complete and evidenced on a Tesla T4.
The repository now also contains a multi-block reduction implementation and a
second vector-add proof with matching correct, faulty, and replay modes.
A third proof, tiled matrix multiplication, is implemented with the same mode
structure and an operation-aware numerical tolerance. The reduction and
vector-add proofs are evidenced by the passing second campaign under
`results/second-campaign/`; the matmul proof is exercised by the third
campaign under `results/third-campaign/`.

The full development history is documented in
[`docs/development-report.md`](docs/development-report.md).

Proven on a Tesla T4 with CUDA 12.8:

- CUDA compilation and execution.
- Boundary/configuration testing for the reduction, vector add, and matmul.
- Compute Sanitizer memcheck clean on the correct implementations.
- Detection of deliberately broken implementations (tail drop, tile defect).
- Serialization of the failures.
- Artifact-driven replay of the failures.

The concrete captured failures:

```text
reduce_sum: 31 values, all 1.0, block 32, expected 31, actual 30
             defect: faulty_tail drops the final element
matmul:     N=32, faulty_tile drops the final inner product term
             defect: every C[i][j] misses A[i][N-1] * B[N-1][j]
```

Replay classification: `same_failure`.

The captured first-campaign evidence is under `results/first-campaign/`.
It records 72/72 correct reduction cases, 64 faulty reduction mismatches,
`same_failure` replay, and zero Compute Sanitizer errors.

## Architecture boundary

```text
cuda/reduction_proof.cu      multi-block reduction: correct + faulty + replay
cuda/vector_add_proof.cu     vector add: correct + faulty + replay
cuda/matmul_proof.cu         tiled matmul: correct + faulty_tile + replay
tools/replay_failure.py      JSON parsing, validation, kernel-aware replay
tools/colab_entry.py         Colab runtime: build, run, sanitize all proofs
tools/minimize_failure.py    reduce prefix search against a pre-fix binary
CMakeLists.txt               optional CUDA build and CTest integration
failures/                    captured counterexamples and minimization seeds
results/                     campaign artifacts
```

Python owns JSON, orchestration, and interpretation. CUDA owns execution.
The CUDA program is the single source of truth for the kernel logic, so the
replay path executes the same faulty implementation as the campaign.

Host-side checks live in `tests/` and run without a GPU:

```bash
python3 -m pytest tests/ -q
```

They cover artifact validation, the operation-aware mismatch tolerance, and the
four replay classifications (`same_failure`, `resolved`, `different_failure`,
`infrastructure_failure`).

## Run on a GPU machine

```bash
cmake -S . -B build -DVANTA_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For a direct reduction-only run:

```bash
mkdir -p build
nvcc -std=c++20 -O0 -lineinfo -arch=sm_75 \
    cuda/reduction_proof.cu -o build/reduction_proof

./build/reduction_proof correct
./build/reduction_proof faulty

python3 tools/replay_failure.py \
    failures/reduction-tail-v0.json \
    --executable build/reduction_proof

compute-sanitizer --tool memcheck --error-exitcode 1 \
    ./build/reduction_proof correct
```

The host-only CTest configuration works on the Mac development machine:

```bash
cmake -S . -B build-host -DVANTA_ENABLE_CUDA=OFF
ctest --test-dir build-host --output-on-failure
```

On Colab, clone this repository and run the complete entrypoint:

```bash
!git clone https://github.com/wheevu/vanta.git
%cd vanta
!python3 tools/colab_entry.py --out results/third-campaign
```

For a direct matmul-only run:

```bash
mkdir -p build
nvcc -std=c++20 -O0 -lineinfo -arch=sm_75 \
    cuda/matmul_proof.cu -o build/matmul_proof

./build/matmul_proof correct
./build/matmul_proof faulty

python3 tools/replay_failure.py \
    failures/matmul-tile-v0.json \
    --executable build/matmul_proof

compute-sanitizer --tool memcheck --error-exitcode 1 \
    ./build/matmul_proof correct
```

## Minimize a real failure

The first minimizer narrows a `reduce_sum` failure to the shortest input prefix
that still diverges from the CPU expectation. Its first target is the historical
multi-block grid-stride overlap bug: the load loop used `i += blockDim.x`
instead of `i += gridDim.x * blockDim.x`. It was introduced in `33a491d`,
revealed by the correct campaign recorded at `54bad3f` (expected 255, actual
1144), and fixed in `1c4b097`.

The bug lives in the **correct** implementation, not the faulty one, so the
seed is replayed as `correct`. `54bad3f` upstream only accepted `faulty_tail`
replay, and `main` already carries the fix, so neither can reproduce the defect.
The `repro/grid-overlap` branch is a reproduction harness: it starts from
`54bad3f`, keeps the overlap defect, and adds `correct` replay acceptance. It is
not merged into `main`.

The minimizer runs against that pre-fix binary, so it must execute where CUDA
is available. It does not touch `main` and leaves the current sources intact.

On Colab, check out the reproduction branch, build only the reduction binary,
and minimize the seed:

```bash
!git clone https://github.com/wheevu/vanta.git
%cd vanta
!git checkout repro/grid-overlap
!mkdir -p build
!nvcc -std=c++20 -O0 -lineinfo -arch=sm_75 \
    cuda/reduction_proof.cu -o build/reduction_sum
!python3 tools/minimize_failure.py \
    failures/reduction-grid-overlap-seed-v0.json \
    --executable build/reduction_sum \
    --out failures/reduction-grid-overlap-min-v0.json
```

The seed (`failures/reduction-grid-overlap-seed-v0.json`) is 255 ones with
`block_threads=32` and `implementation: correct`. On the reproduction binary the
single-block cases (`n <= 32`) sum correctly, while multi-block cases overlap
tiles and diverge from the expected 255. The minimizer performs a linear prefix
search and converges to the smallest multi-block case (`n = 33`), isolating the
overlap defect free of the tail-drop artifact. It writes the shortest
still-failing case plus a `minimization` provenance block to the output
artifact.

The search is intentionally linear, not binary: for arbitrary kernels a
mismatch predicate is not assumed monotonic. Fewer GPU launches matters less
than a trustworthy minimal case.

## Correctness model

Reduction cases compare the GPU result against a straightforward CPU sum in
double precision.
Vector-add cases compare every output element against the CPU formula
`a*x[i] + y[i]`.
Matmul cases compare the full output matrix against a CPU double-precision
reference `C = A · B`.

The tolerance is **operation-aware**, not a single global constant:

```text
reduce_sum / vector_add:  tol = 1e-5 * max(1.0, |expected|)
matmul:                   tol = max(1.0, |expected|) * K * 3 * epsilon_32
```

For small integer sums the first form is effectively exact. The matmul form
tracks float32 accumulation: a tiled matmul adds K products per output
element, so the worst-case rounding over K serial additions is bounded by
`K * epsilon * |result|`, with a small 3x safety margin rather than a tuned
knob. The same bound is mirrored in the CUDA campaign gate and the Python
replay classifier. If Vanta ever "passes" because a tolerance was loosened,
that is a defect in Vanta, not in the kernel.

## Known limitations

- Minimization covers `reduce_sum` 1D prefixes only. Other kernels and
  structured (non-prefix) reductions are out of scope for the first tool.
- Blind spots remain in the campaign: pattern 2 (`(i%7)-3`) at `N=32` and
  `N=256` injects a defect on a zero-valued tail element, so the faulty
  variant passes for 8 cases per kernel. The campaign needs redesign so every
  injected defect is observable.
- No softmax or general GEMM yet. Matmul is a square, tiled proof only.
- No benchmarking or performance claims.
- No GPU CI. Colab is the execution target, not an authority for
  performance or stability.
- Compute Sanitizer clean does not imply race freedom; it detects a
  different defect class than differential validation.
