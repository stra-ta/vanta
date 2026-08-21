# First-campaign evidence notes

This directory holds the artifacts from the first Vanta reduction proof,
run on a Colab Tesla T4 (CUDA 12.8).

## Present
- `environment.json` - GPU / nvcc / Compute Sanitizer / Python versions.
- `correct.log` - correct-kernel campaign output (72/72 PASS).
- `replay.log` - replay of `failures/reduction-tail-v0.json`, classification `same_failure`.
- `memcheck.log` - Compute Sanitizer memcheck on the correct kernel (0 errors).
- `campaign_summary.json` - status `pass` (schema from the run that produced these logs).

## Missing from the local download
- `faulty.log` - faulty-kernel campaign output (detected 64 mismatches).
- `build.log` - nvcc compilation transcript.

Both are regenerated automatically by the corrected entrypoint. From a GPU
runtime with this repo checked out:

```bash
python3 tools/colab_entry.py \
    --source cuda/reduction_proof.cu \
    --artifact failures/reduction-tail-v0.json \
    --out results/first-campaign
```

That run overwrites every file here with a consistent `vanta.campaign.v0`
summary (including `compile_ok`, `faulty_campaign_detected`, and
`replay_classification`).
