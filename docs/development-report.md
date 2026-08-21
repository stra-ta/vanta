# Vanta development report

This report records how Vanta moved from a CUDA validation idea to a tested
two-kernel proof with replayable failure artifacts.

The report describes the repository through the failure-minimization milestone
on `main`, with the historical overlap bug isolated on the `repro/grid-overlap`
branch.

## Executive summary

Vanta is a small GPU kernel validation laboratory.

It compares CUDA results with CPU references, runs boundary and launch
configuration campaigns, detects deliberately faulty implementations,
serializes failures, replays those failures, and runs Compute Sanitizer.

The final Colab campaign passed for both kernels:

- `reduce_sum` compiled, passed its correct campaign, detected its faulty
  implementation, replayed the captured failure as `same_failure`, and passed
  memcheck with zero errors.
- `vector_add` compiled, passed its correct campaign, detected its faulty
  implementation, replayed the captured failure as `same_failure`, and passed
  memcheck with zero errors.

The final aggregate result is recorded in
[`results/second-campaign/campaign_summary.json`](../results/second-campaign/campaign_summary.json).

The final campaign used a Tesla T4 with CUDA 12.8 and Compute Sanitizer
2025.1.0.

## Starting constraints

Development started on a MacBook Air with an Apple Silicon M1 processor.

The local machine did not have `nvcc`, `nvidia-smi`, or a CUDA-capable GPU.

That created a hard split in the workflow:

- The Mac handled source code, Python tooling, tests, artifacts, Git, and
  orchestration.
- Google Colab handled CUDA compilation, GPU execution, and Compute
  Sanitizer.

Lightning AI was considered as an execution environment, but phone
verification prevented that route.

Colab became the execution target after a T4 runtime was selected and verified.

The working Colab environment reported the following versions:

- GPU: Tesla T4.
- Compute capability: 7.5.
- Driver: 580.82.07.
- CUDA compiler: 12.8.93.
- Compute Sanitizer: 2025.1.0.0.
- Python: 3.12.13.

## Development path

### 1. The first proof was deliberately narrow

The initial scope was reduced to one sum-reduction kernel, one CPU reference,
one faulty implementation, and one captured failure.

The first campaign used these input sizes:

```text
31, 32, 33, 255, 256, 257
```

It used these block sizes:

```text
32, 64, 128, 256
```

It used three input patterns:

- Pattern 0: all values are `1.0`.
- Pattern 1: alternating `1.0` and `-1.0`.
- Pattern 2: `(i % 7) - 3`.

The CPU reference sums `float32` inputs in `double` precision.

The comparison tolerance is fixed at:

```text
tol = 1e-5 * max(1.0, abs(expected))
```

The tolerance is implemented on both the CUDA campaign side and the Python
replay side.

### 2. The first reduction implementation

The original reduction used one CUDA program with three modes:

```text
reduction_proof correct
reduction_proof faulty
reduction_proof replay
```

The faulty variant dropped the final input element by excluding `i == n - 1`
from the accumulation.

The first concrete failure used 31 values of `1.0` with a block size of 32:

```text
expected: 31
actual:   30
```

The CUDA program wrote that failure to `failure.json`.

The tracked version is
[`failures/reduction-tail-v0.json`](../failures/reduction-tail-v0.json).

### 3. Failure replay became the first real tool boundary

The first replay path separated responsibilities between CUDA and Python.

CUDA consumed a small line protocol and returned the observed result as JSON.

Python loaded and validated the failure artifact, built the input protocol,
computed the tolerance, and classified the replay.

The classification rules are:

- `same_failure`: the replay still mismatches and its actual value matches the
  recorded actual value.
- `resolved`: the replayed result is within tolerance of the reference.
- `different_failure`: the replay still fails, but the reference or actual
  value differs from the recorded failure.
- `infrastructure_failure`: the executable fails or returns unusable output.

An early version trusted a `mismatch` field that the CUDA replay program did
not emit.

That caused a real faulty replay to be classified as `resolved`.

The fix moved mismatch computation entirely into
[`tools/replay_failure.py`](../tools/replay_failure.py).

The Python tool now derives mismatch from `expected` and `actual` and returns
success only for `same_failure`.

Malformed JSON and nonzero executable exits now produce
`infrastructure_failure` instead of raising an uncaught exception.

### 4. The first campaign established the evidence format

The first campaign passed with these results:

- Correct reduction campaign: 72 passes out of 72 cases.
- Faulty reduction campaign: 64 mismatches and 8 expected blind spots.
- Replay: `same_failure`.
- Memcheck: zero errors.

The eight blind spots occur for Pattern 2 at N=32 and N=256.

In those cases the final input value is zero, so dropping it does not change
the result.

This behavior is documented rather than hidden.

The first campaign is preserved under `results/first-campaign/`.

It is historical evidence for the original reduction implementation.

### 5. The repository became reproducible

The project was moved into a public GitHub repository:

<https://github.com/wheevu/vanta>

The repository gained a Colab entrypoint,
[`tools/colab_entry.py`](../tools/colab_entry.py), which performs the full
GPU workflow:

1. Probe the environment.
2. Compile the selected kernels.
3. Run correct campaigns.
4. Run faulty campaigns.
5. Replay captured artifacts.
6. Run Compute Sanitizer memcheck.
7. Write logs and an aggregate campaign summary.

Each kernel now gets an isolated directory under the campaign output path.

The runner writes `build.log`, `correct.log`, `faulty.log`, `replay.log`, and
`memcheck.log` for each kernel.

The runner writes a `vanta.campaign.v1` summary containing both per-kernel and
aggregate status fields.

Missing GPU tools no longer cause the runner to crash before writing a
diagnostic summary.

### 6. Host-side regression coverage was added

The host tests live in [`tests/test_replay.py`](../tests/test_replay.py).

They cover:

- Reduction artifact validation.
- Vector-add artifact validation.
- Shape and auxiliary-vector validation.
- Payload construction for both kernels.
- Tolerance behavior.
- `same_failure` classification.
- `resolved` classification.
- `different_failure` classification.
- `infrastructure_failure` from malformed output.
- `infrastructure_failure` from nonzero executable exit.
- Compile-failure summary generation in the Colab runner.

The final host test result is 27 passing tests.

### 7. The build boundary became explicit

[`CMakeLists.txt`](../CMakeLists.txt) supports two configurations.

The host-only configuration disables CUDA and registers the Python test suite.

The CUDA configuration enables C++20 CUDA targets for both proofs and adds
CTest entries for correct campaigns, faulty campaigns, replay, and memcheck
when `compute-sanitizer` is available.

The Mac can configure and run the host-only test set with:

```bash
cmake -S . -B build-host -DVANTA_ENABLE_CUDA=OFF
ctest --test-dir build-host --output-on-failure
```

A GPU machine can configure and build the complete suite with:

```bash
cmake -S . -B build -DVANTA_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Multi-block reduction work

The first implementation reduced one block at a time.

The next implementation added a two-pass reduction:

1. `reduce_block` computes one partial sum per block.
2. `reduce_partials` reduces those partial sums to one output value.

The shared-memory tree uses an active-count reduction instead of assuming a
power-of-two block size.

The load loop uses a grid-stride step:

```text
gridDim.x * blockDim.x
```

That detail mattered.

The first multi-block version used `blockDim.x` as the stride.

Every block then reread the same global input tiles.

The first second-campaign run caught the defect:

```text
N=255, block=32, expected=255, actual=1144
```

The bug produced logical duplication rather than an out-of-bounds access.

Compute Sanitizer therefore did not identify it.

The correct campaign, rather than memcheck, provided the useful signal.

The grid-stride fix was committed as `1c4b097`.

The next Colab run passed the multi-block reduction campaign.

## Second kernel: vector add

The second proof is [`cuda/vector_add_proof.cu`](../cuda/vector_add_proof.cu).

It validates the operation:

```text
y_out[i] = a * x[i] + y[i]
```

It supports the same three modes:

```text
vector_add_proof correct
vector_add_proof faulty
vector_add_proof replay
```

The faulty variant writes zero for the final output element.

The vector-add failure artifact extends the common artifact shape with:

- `params.a` for the scalar multiplier.
- `aux.y` for the second input vector.
- `observed.index` for the failing output element.

The primary input vector remains under `input.values`.

[`tools/replay_failure.py`](../tools/replay_failure.py) dispatches payload
construction by the artifact's `kernel` field.

The final vector-add campaign produced 72 correct passes, 64 mismatches in the
faulty campaign, 8 expected blind spots, a `same_failure` replay, and a clean
memcheck.

## Final campaign

The final evidence is under `results/second-campaign/`.

The aggregate summary reports:

```json
{
  "compile_ok": true,
  "correct_campaign_ok": true,
  "faulty_campaign_detected": true,
  "memcheck_ok": true,
  "status": "pass"
}
```

Each kernel reports the following:

| Kernel | Correct cases | Faulty mismatches | Replay | Memcheck |
| --- | ---: | ---: | --- | --- |
| `reduce_sum` | 72 | 64 | `same_failure` | 0 errors |
| `vector_add` | 72 | 64 | `same_failure` | 0 errors |

The generated `failure.json` for each kernel matches its tracked artifact.

The compiled Colab binaries are not committed.

The campaign ran on a Tesla T4 with CUDA 12.8.

## Repository structure

```text
cuda/
  reduction_proof.cu       Two-pass multi-block sum reduction.
  vector_add_proof.cu      Elementwise vector add.

failures/
  reduction-tail-v0.json   Captured reduction counterexample.
  vector_add-tail-v0.json  Captured vector-add counterexample.

tools/
  replay_failure.py        Artifact validation, payloads, classification.
  colab_entry.py           GPU build and campaign orchestration.

tests/
  test_replay.py           Host-side replay and artifact tests.

results/
  first-campaign/           Historical reduction-only evidence.
  second-campaign/          Final two-kernel evidence.
  notebook-evidence/        Original exploratory notebooks and outputs.

CMakeLists.txt              Host and CUDA CTest configuration.
README.md                   Short project entrypoint.
```

## Verification commands

The host test suite passes with:

```bash
python3 -m py_compile tools/replay_failure.py tools/colab_entry.py
python3 -m pytest tests/ -q
```

The final host result is:

```text
27 passed
```

The host-only CTest configuration passes with:

```bash
cmake -S . -B build-host -DVANTA_ENABLE_CUDA=OFF
ctest --test-dir build-host --output-on-failure
```

The final GPU evidence comes from:

```bash
python3 tools/colab_entry.py --out results/second-campaign
```

The final campaign summary reports `pass` for both kernels.

## Decisions and tradeoffs

### Colab instead of local CUDA

The Mac remains useful for editing and host tests, but it cannot validate CUDA
execution.

Colab adds a manual execution step, but it supplies a known T4 environment and
the required CUDA tools without adding a local toolchain dependency.

### One source file per proof

Each CUDA proof keeps its correct implementation, faulty implementation, and
replay path in one source file.

That makes it harder for replay to accidentally execute a different kernel
than the campaign.

The cost is repetition between proof files.

At this stage, the repetition is preferable to an abstraction that would hide
the execution path while the validation model is still changing.

### JSON in Python, execution in CUDA

Python owns artifact structure and result classification.

CUDA owns allocation, kernel launches, synchronization, and device-to-host
results.

This split keeps the failure schema inspectable without putting JSON handling
inside the CUDA proofs.

### Differential validation and memcheck are separate signals

The faulty reduction first exposed a logical error that memcheck could not see.

The multi-block overlap bug did the same thing.

Memcheck checks a different class of failure from CPU differential validation.

Vanta keeps both checks because a clean sanitizer run is not evidence that the
kernel computes the right answer.

## Known limits

The project minimizes `reduce_sum` prefix failures through
`tools/minimize_failure.py`. Other kernels and structured (non-prefix)
reductions are out of scope for the first tool. The historical overlap bug is
reproduced on the `repro/grid-overlap` branch; `main` carries the fix.

The project does not yet benchmark kernel performance.

There is no GPU CI job.

Colab provides the current GPU execution environment, but the repository does
not yet enforce a scheduled or hosted GPU regression run.

The campaign covers three small proofs rather than a broad kernel library.

Matmul exists as a square, tiled proof; there is no softmax or general GEMM
yet.

The current artifacts use one captured failure per kernel.

The campaign no longer contains blind spots: pattern 2 was re-anchored at
`(i % 7) + 1` so the dropped final value is never zero, and every injected
defect is now observable across all tested sizes.

## Commit milestones

| Commit | Milestone |
| --- | --- |
| `24f4ce4` | Corrected replay classification and added the first host tests. |
| `be6878f` | Hardened malformed replay and build-failure handling. |
| `f7bd8e8` | Documented the public Colab clone workflow. |
| `de4b531` | Added complete first-campaign evidence. |
| `33a491d` | Added multi-block reduction, vector add, and CMake/CTest. |
| `54bad3f` | Added arbitrary block-size handling and missing-tool diagnostics. |
| `1c4b097` | Fixed grid-stride loading in the multi-block reduction. |
| `3f81a51` | Added the passing second-campaign aggregate evidence. |
| `832589b` | Added nested per-kernel logs and generated failure artifacts. |
| `repro/grid-overlap` | Reproduction harness for the grid-stride overlap bug. |
| `1ddd18c` | Added tiled matmul proof with operation-aware tolerance. |
| `06752e1` | Corrected matmul artifact grid_blocks provenance. |
| `d3a5213` | Closed the campaign blind spot (pattern 2 re-anchored at +1). |

## Current conclusion

Vanta now demonstrates a complete validation loop for three CUDA proofs.

The loop begins with a CPU reference and a structured launch matrix.

It detects a deliberately broken implementation, writes a counterexample,
replays that counterexample through the same CUDA implementation, and checks
the correct implementation with Compute Sanitizer.

The final Colab campaign passed for both the reduction and vector-add kernels,
and the matmul proof is exercised by the third campaign.

## Failure minimization (reduce prefix)

The first minimizer targets a real historical defect rather than an invented
one: the multi-block grid-stride overlap bug. The load loop used
`i += blockDim.x` instead of `i += gridDim.x * blockDim.x`. It was introduced
in `33a491d`, revealed by the correct campaign recorded at `54bad3f` (expected
255, actual 1144), and fixed in `1c4b097`. At `N=255` with `block_threads=32`, the broken stride
(`i += blockDim.x`) overlaps tiles, so the correct campaign reported expected
255 against actual 1144, while memcheck stayed clean (legal memory, wrong
result).

`tools/minimize_failure.py` loads a `reduce_sum` seed artifact, then performs a
**linear** prefix search: it replays progressively shorter prefixes against a
pre-fix binary and keeps the shortest prefix whose replay still diverges from
the CPU expectation. The search is linear rather than binary on purpose. A
mismatch predicate is not assumed monotonic for arbitrary kernels, and a
trustworthy minimal case matters more than fewer GPU launches.

The tool runs only where CUDA exists, against a `--executable` built from the
buggy revision, so it never touches `main`. The shipped seed is
`failures/reduction-grid-overlap-seed-v0.json` (255 ones, `block_threads=32`).
The minimizer writes `failures/reduction-grid-overlap-min-v0.json` with a
`minimization` provenance block recording the original length, minimized
length, and tested case count.

Host tests cover the minimizer without a GPU by monkeypatching the replay
subprocess. One simulation reproduces the pre-fix behavior (single-block cases
correct, multi-block cases wrong) and asserts the minimizer converges to a 33
element prefix (the smallest multi-block case, `N=33`, `block_threads=32`).

## Matrix multiplication and operation-aware tolerance

The third proof is a tiled square matrix multiplication. `cuda/matmul_proof.cu`
keeps the same mode structure (correct / faulty_tile / replay) and a tiled
shared-memory kernel. The deliberate defect drops the final inner product term
(`k == N - 1`), so every output element misses `A[i][N-1] * B[N-1][j]`. Like
the earlier defects it is a logical error, not an out-of-bounds access, so
Compute Sanitizer memcheck stays clean while differential validation catches
it.

Adding matmul forced the tolerance model to become operation-aware. A tiled
matmul adds K products per output element; the worst-case float32 rounding
over K serial additions is bounded by `K * epsilon * |result|`. Vanta now uses:

```text
reduce_sum / vector_add:  tol = 1e-5 * max(1.0, |expected|)
matmul:                   tol = max(1.0, |expected|) * K * 3 * epsilon_32
```

The matmul bound carries a small 3x safety margin rather than a tuned knob.
The same formula is mirrored in the CUDA campaign gate and the Python replay
classifier, so the correct kernel passes and a structural defect fails by a
wide margin. The replay path computes the CPU reference in Python and compares
the full output matrix element-wise; the artifact records the reference scale
(`observed.expected`) and the maximal per-element error of the captured faulty
run (`observed.actual`) so replay can confirm it reproduces the same defect.

The captured matmul case is `failures/matmul-tile-v0.json` (N=32). Its values
are computed with the same drop-last-term defect the kernel uses, so a GPU
replay reproduces `same_failure` without a live CUDA run on the Mac.

## Sanitizer proof

The first two kernels only exercised Compute Sanitizer `memcheck` on their
`correct` mode, which proves the infrastructure runs but never proves the
sanitizer can catch a real defect. That is a blind spot: a green memcheck can
mean "clean" or "nothing wrong to find".

`cuda/sanitizer_proof.cu` closes it with deliberately broken kernels instead of
relying on accidental product defects. It has four modes:

```text
correct      clean baseline, passes memcheck + racecheck + synccheck
broken_oob   one thread writes one element past the allocation -> memcheck
broken_sync  divergent __syncthreads (guarded by threadIdx.x == 0) -> synccheck
broken_race  one thread writes a shared variable, others read it without sync -> racecheck
```

The kernel is not a product: it takes a positional mode argument and an
optional tool hint, and runs a single 128-thread block over a 256-element
buffer. The breakages are independent of the earlier campaign logic.

`tools/check_sanitizer.py` owns the verdict, not the C++ binary. It shells out
to `compute-sanitizer --tool X --error-exitcode 1 <exe> <mode>`, then decides
whether the outcome matches the expectation: `correct` must stay clean, the
`broken_*` modes must be detected (`--expect-detected`). It emits one JSON line
and exits 0 only on a match, so CTest runs it directly. The host tests cover
the decision logic by monkeypatching the sanitizer subprocess, including the
mismatch case (expected detected but clean) and the missing-tool / missing-binary
exit codes.

`broken_oob` (out-of-bounds global write) and `broken_sync` (divergent
`__syncthreads`) are reliably flagged by memcheck and synccheck respectively.
`broken_race` is the construct most worth confirming on real silicon; if a
given Compute Sanitizer build does not flag the unsynchronized shared-memory
read, that variant is the one to adjust, not the verdict logic.

## Current state

The repository now has three product-style proofs (reduce_sum, vector_add,
matmul), a replayable minimization of a real historical grid-stride bug, and a
sanitizer proof that demonstrates both clean runs and caught defects. Host tests
cover the Python layer (artifact schema, matcher/tolerance, minimizer, sanitizer
verdict) without a GPU; the CUDA artifacts are validated on the Colab T4.

The next useful extension is a GPU CI path that reruns the campaign without
manual Colab steps.

