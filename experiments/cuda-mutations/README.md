# CUDA mutation campaign

This experiment measures where Vanta's validation layers detect CUDA defects,
and where they do not.

It is not a mutation-score contest.

Each subject is the pinned canonical proof source plus one preregistered source
transformation.

The canonical proofs are never edited to add mutation selectors.

## Scope

`manifest.json` preregisters 18 mutations, six for each product proof:

```text
reduce_sum  historical stride, tail, reduction bound, shared sync, barrier, OOB partial
vector_add  launch floor, bound, block index, OOB read, scalar, operand order
matmul      K tail, tile count, edge guard, B index, tile sync, barrier
```

The runner executes independent CPU differential comparison, metamorphic
relations, memcheck, racecheck, synccheck, deterministic replay, and prefix
minimization where the failure predicate is suitable.

## Run on a CUDA host

```bash
python3 experiments/cuda-mutations/run_campaign.py \
    --out results/cuda-mutation-campaign-v0
```

The host must provide `nvcc`, `nvidia-smi`, and `compute-sanitizer`.

The initial protocol targets the T4 evidence environment (`sm_75`).

The runner returns zero when every baseline is clean and the campaign completes.

Mutants that survive are an experimental result, not a process failure.

## Reproducibility boundary

For every subject, the campaign records the canonical commit, canonical source
hash, study-adapter hash where needed, transformation hash, derived-source hash,
binary hash, compiler flags, environment probes, commands, raw-log hashes, and
the generated unified diff.

`vector_add` has a shared replay adapter because its canonical replay mode only
returns the deliberate tail fault's final output element.

The adapter expands replay output to the full vector without changing the kernel
or campaign path.

## Reading results

`campaign.json` is the authoritative machine-readable record.

`detection-matrix.md` is generated from it and must not be edited by hand.

Each detector cell is one of:

```text
detected
no_signal
execution_error
tool_error
timeout
not_run
```

`no_signal` means only that the detector reported nothing for the tested
configuration.

It does not mean the mutant is correct, memory-safe, race-free, or equivalent.

Sanitizer eligibility is recorded separately because memcheck, racecheck, and
synccheck observe different defect classes.

The report should emphasize unique kills, overlaps, survivors, flaky outcomes,
and known blind spots rather than a single score.
