# Vanta contributor guide

## Scope

Vanta validates CUDA kernels through CPU references, metamorphic relations, replayable failures, and Compute Sanitizer.

CUDA execution owns observed outputs.

Python owns artifact parsing, tolerance, classification, and campaign bookkeeping.

## Invariants

- A canonical kernel source is never edited to add a mutant selector.
- Every mutation derives from one pinned source transformation whose match count is checked.
- `no_signal` means only that one detector and configuration reported no defect.
- A timeout, crash, or sanitizer tool error is infrastructure evidence, never a detection.
- `same_failure`, `resolved`, and `different_failure` remain distinct replay outcomes.
- Operation-aware tolerances stay derived from the numeric operation and are tested on the host.
- Committed results preserve raw logs and provenance, including commit, dirty state, compiler, kernel, CPU, architecture, build type, seed, and command line.
- GPU results require GPU evidence.

## Commands

Run host verification with `./scripts/verify.sh`.

Run `./scripts/confidence.sh` for host verification and the GPU proof campaign when CUDA tools are available.

Run `python3 -m pytest tests -q` for the focused host suite.

Run `cmake -S . -B build-host -DVANTA_ENABLE_CUDA=OFF` and `ctest --test-dir build-host --output-on-failure` for host CTest.

Run the CUDA proof entrypoint with `python3 tools/colab_entry.py --out results/local/campaign` on a GPU host.

Run mutation studies with `python3 experiments/cuda-mutations/run_campaign.py --manifest experiments/cuda-mutations/manifest-v3.json --out results/local/cuda-mutations` on a GPU host.

Use `tools/minimize_failure.py` only against a replayable failure artifact and preserve the pre-fix binary provenance.

## Verification

Host tests validate schemas, references, tolerance, replay classification, minimization, and sanitizer-output interpretation.

CUDA campaign results must distinguish functional correctness, sanitizer observations, and performance evidence.

Do not add performance claims without a committed campaign artifact and machine metadata.


## Lab-wide contracts

- See https://github.com/stra-ta/.github/blob/main/LAB_RULES.md and https://github.com/stra-ta/.github/blob/main/EVIDENCE.md and https://github.com/stra-ta/.github/blob/main/COMPATIBILITY.md for lab-wide naming, evidence, and schema contracts.
- Per https://github.com/stra-ta/.github/blob/main/CONTRIBUTING.md, contributions require the target repo's AGENTS.md, README, and relevant design note, preserve repo boundaries, add the narrowest regression test, run one-command verification, and keep performance claims tied to committed manifests.
