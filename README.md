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
The new code needs to be run on Colab before its GPU results are treated as
committed evidence.

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
CMakeLists.txt               optional CUDA build and CTest integration
failures/                    captured counterexamples
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

- No failure minimization.
- No matrix multiplication or softmax.
- No benchmarking or performance claims.
- No GPU CI. Colab is the execution target, not an authority for
  performance or stability.
- Compute Sanitizer clean does not imply race freedom; it detects a
  different defect class than differential validation.
