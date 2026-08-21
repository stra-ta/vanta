# Second-campaign evidence

This directory contains the complete GPU validation run for the multi-block
reduction and vector-add proofs.
The run used a Colab Tesla T4 with CUDA 12.8 and Compute Sanitizer 2025.1.0.

## Aggregate result

`campaign_summary.json` uses `vanta.campaign.v1` and reports `status: pass`.

Both kernels passed:

- compilation
- correct campaign
- faulty implementation detection
- artifact-driven replay with `same_failure`
- Compute Sanitizer memcheck with zero errors

## Kernel evidence

Each kernel directory contains `build.log`, `correct.log`, `faulty.log`,
`replay.log`, `memcheck.log`, and the generated `failure.json`.

The generated failure artifacts match the tracked artifacts exactly:

- `reduce_sum/failure.json` matches `failures/reduction-tail-v0.json`.
- `vector_add/failure.json` matches `failures/vector_add-tail-v0.json`.

The compiled binaries from the Colab archive are intentionally excluded from
the repository.
