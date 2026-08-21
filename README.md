# Vanta

A GPU validation laboratory for determining whether CUDA kernels are
correct, robust across launch configurations, diagnosable when they fail,
and acceptably performant under documented measurement conditions.

Vanta is not a CUDA tutorial, a BLAS replacement, a deep-learning framework,
or a kernel collection. The test framework is the primary deliverable.

## Current status: first proof

The repository currently contains one milestone: a single-block sum
reduction validated against a CPU double-precision reference.

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

## Architecture boundary

```text
cuda/reduction_proof.cu      one CUDA program: correct + faulty + replay
tools/replay_failure.py      JSON parsing, validation, classification
tools/colab_entry.py         Colab runtime: build, run, sanitize
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

On Colab, use the entrypoint:

```bash
python3 tools/colab_entry.py --source cuda/reduction_proof.cu
```

## Correctness model

Each case compares the GPU result against a straightforward CPU sum in
double precision. The tolerance is:

```text
tol = 1e-5 * max(1.0, |expected|)
```

For small integer sums this is effectively exact. The tolerance is a
documented property of the milestone, not a tuned knob. If Vanta ever
"passes" because the tolerance was loosened, that is a defect in Vanta,
not in the kernel.

## Known limitations

- One kernel only: single-block sum reduction.
- Single-block launch only; no grid-stride or multi-block reduction yet.
- No failure minimization.
- No matrix multiplication or softmax.
- No benchmarking or performance claims.
- No GPU CI. Colab is the execution target, not an authority for
  performance or stability.
- Compute Sanitizer clean does not imply race freedom; it detects a
  different defect class than differential validation.
