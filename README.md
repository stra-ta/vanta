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
The multi-block reduction and vector-add proofs are also evidenced by the
passing second campaign under `results/second-campaign/`.

The full development history is documented in
[`docs/development-report.md`](docs/development-report.md).

Proven on a Tesla T4 with CUDA 12.8:

- CUDA compilation and execution.
- Boundary/configuration testing for the reduction.
- Compute Sanitizer memcheck clean on the correct implementation.
- Detection of one deliberately broken implementation.
- Serialization of the failure.
- Artifact-driven replay of the failure.

The concrete captured failure:

```text
input:      31 values, all 1.0
block size: 32
expected:   31
actual:     30
defect:     faulty_tail drops the final element
```

Replay classification: `same_failure`.

The captured first-campaign evidence is under `results/first-campaign/`.
It records 72/72 correct reduction cases, 64 faulty reduction mismatches,
`same_failure` replay, and zero Compute Sanitizer errors.

## Architecture boundary

```text
cuda/reduction_proof.cu      multi-block reduction: correct + faulty + replay
cuda/vector_add_proof.cu     vector add: correct + faulty + replay
tools/replay_failure.py      JSON parsing, validation, kernel-aware replay
tools/colab_entry.py         Colab runtime: build, run, sanitize both proofs
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

They cover artifact validation, the mismatch tolerance, and the four replay
classifications (`same_failure`, `resolved`, `different_failure`,
`infrastructure_failure`).

## Run on a GPU machine

```bash
cmake -S . -B build -DVANTA_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For a direct reduction-only run:

```bash
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
!python3 tools/colab_entry.py --out results/second-campaign
```

## Minimize a real failure

The first minimizer narrows a `reduce_sum` failure to the shortest input prefix
that still diverges from the CPU expectation. Its first target is the historical
multi-block grid-stride overlap bug fixed in `1c4b097`, introduced in `54bad3f`.

The minimizer runs against a **pre-fix binary**, so it must execute where CUDA
is available. It does not touch `main` and leaves the current sources intact.

On Colab, check out the buggy revision, build only the reduction binary, and
minimize the seed:

```bash
!git clone https://github.com/wheevu/vanta.git
%cd vanta
!git checkout 54bad3f
!nvcc -std=c++20 -O0 -lineinfo -arch=sm_75 \
    cuda/reduction_proof.cu -o build/reduction_sum
!python3 tools/minimize_failure.py \
    failures/reduction-grid-overlap-seed-v0.json \
    --executable build/reduction_sum \
    --out failures/reduction-grid-overlap-min-v0.json
```

The seed (`failures/reduction-grid-overlap-seed-v0.json`) is 255 ones with
`block_threads=32`, which on the pre-fix binary overlaps tiles and diverges
from the expected 255. The minimizer performs a linear prefix search and writes
the shortest still-failing case plus a `minimization` provenance block to the
output artifact.

The search is intentionally linear, not binary: for arbitrary kernels a
mismatch predicate is not assumed monotonic. Fewer GPU launches matters less
than a trustworthy minimal case.

## Correctness model

Reduction cases compare the GPU result against a straightforward CPU sum in
double precision.
Vector-add cases compare every output element against the CPU formula
`a*x[i] + y[i]`.
Both use the same tolerance:

```text
tol = 1e-5 * max(1.0, |expected|)
```

For small integer sums this is effectively exact. The tolerance is a
documented property of the milestone, not a tuned knob. If Vanta ever
"passes" because the tolerance was loosened, that is a defect in Vanta,
not in the kernel.

## Known limitations

- Minimization covers `reduce_sum` 1D prefixes only. Other kernels and
  structured (non-prefix) reductions are out of scope for the first tool.
- Blind spots remain in the campaign: pattern 2 (`(i%7)-3`) at `N=32` and
  `N=256` injects a defect on a zero-valued tail element, so the faulty
  variant passes for 8 cases per kernel. The campaign needs redesign so every
  injected defect is observable.
- No matrix multiplication or softmax yet.
- No operation-aware numerical tolerances yet. The fixed
  `1e-5 * max(1, |expected|)` tolerance is fine for integer sums but will
  need to become operation-aware before matmul/softmax.
- No benchmarking or performance claims.
- No GPU CI. Colab is the execution target, not an authority for
  performance or stability.
- Compute Sanitizer clean does not imply race freedom; it detects a
  different defect class than differential validation.
