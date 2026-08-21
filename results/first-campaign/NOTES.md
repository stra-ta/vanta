# First-campaign evidence

This directory holds the complete artifacts from the first Vanta reduction
proof, produced by `tools/colab_entry.py` on a Colab Tesla T4 (CUDA 12.8)
and committed from the exact `main` revision.

## Contents
- `environment.json` - GPU / nvcc / Compute Sanitizer / Python versions.
- `build.log` - nvcc compilation transcript (empty on a clean build).
- `correct.log` - correct-kernel campaign output (72/72 PASS).
- `faulty.log` - faulty-kernel campaign output (64 mismatches detected).
- `replay.log` - replay of `failures/reduction-tail-v0.json`, classification `same_failure`.
- `memcheck.log` - Compute Sanitizer memcheck on the correct kernel (0 errors).
- `campaign_summary.json` - `vanta.campaign.v0` summary, `status: pass`.

## How it was generated
From a GPU runtime with this repo checked out:

```bash
python3 tools/colab_entry.py \
    --source cuda/reduction_proof.cu \
    --artifact failures/reduction-tail-v0.json \
    --out results/first-campaign
```

## Notes
- The compiled `reduction_proof` binary is intentionally NOT committed; it is
  a build artifact produced by nvcc and is gitignored under `build/`.
- The faulty tail drops the final element. For pattern 2 at N=32 and N=256
  the final value is 0, so the defect is invisible there (the 8 PASS lines in
  `faulty.log`). All other 64 cases mismatch, which is the expected signal.
