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
reduce_sum  historical stride, tail, partial scaling, shared sync, syncwarp mask, OOB partial
vector_add  launch floor, bound, block index, OOB read, scalar, operand order
matmul      K tail, tile count, edge guard, B index, tile sync, barrier
```

### v1 outcome and v2 revision

The v1 campaign ran to completion on a Tesla T4 and is preserved as raw,
unmodified evidence. It produced three clean baselines and 16 killed mutants,
with two preregistration defects:

- `R03` (`odd_reduction_bound`) survived because it was an equivalent mutation.
  For the power-of-two block sizes the corpus exercises, `tid < half` already
  implies `tid + half < active`, so changing `<` to `<=` cannot alter execution.
- `R05` (`divergent_barrier`) deadlocked every probe, including synccheck.
  The runner recorded bounded timeouts and classified it
  `infrastructure_failure`. A timeout is never treated as detection.

`manifest-v2.json` (`study_id: cuda-mutations-t4-v2`) replaces exactly those
two subjects and changes nothing else:

- `R03` becomes `partial_sum_scaling`: the first-pass write doubles the partial
  sum. Deterministic, memory-safe, and detected by differential comparison.
- `R05` becomes `invalid_syncwarp_mask`: after the valid post-load barrier,
  lane 16 calls `__syncwarp` with a mask that does not include itself.
  Synccheck-only execution. A targeted T4 probe must confirm termination and
  detection before the full campaign runs.

v1 results are never resumed or overwritten with the v2 manifest.

### v3 revision

The v2 T4 probe accepted `R03` but rejected the replacement `R05`
(`invalid_syncwarp_mask`): synccheck reported 34 invalid-argument barrier
errors, but the uninstrumented program itself failed with an illegal
instruction, so the fault is a direct runtime failure rather than a
synccheck-only diagnostic boundary. The probe is preserved as rejected
evidence.

`manifest-v3.json` (`study_id: cuda-mutations-t4-v3`) differs from v2 only in
`R05`, which becomes `warp_divergent_barrier`: inside the reduction loop, the
lower warp and the upper warps reach two different `__syncthreads()` sites.
The split is warp-aligned (a single-warp launch keeps one valid barrier), so
the fault is active only at the preregistered trigger (`n=257`,
`block_threads=128`). Synccheck-only execution; a targeted probe must confirm
clean termination and 3/3 detection before the full campaign runs.

The runner executes independent CPU differential comparison, metamorphic
relations, memcheck, racecheck, synccheck, deterministic replay, and prefix
minimization where the failure predicate is suitable.

## Run on a CUDA host

```bash
# v1 protocol (original preregistration)
python3 experiments/cuda-mutations/run_campaign.py \
    --out results/cuda-mutation-campaign-v1

# v2 protocol (revised R03/R05)
python3 experiments/cuda-mutations/run_campaign.py \
    --manifest experiments/cuda-mutations/manifest-v2.json \
    --out results/cuda-mutation-campaign-v2

# v3 protocol (revised R05)
python3 experiments/cuda-mutations/run_campaign.py \
    --manifest experiments/cuda-mutations/manifest-v3.json \
    --out results/cuda-mutation-campaign-v3
```

The host must provide `nvcc`, `nvidia-smi`, and `compute-sanitizer`.

The initial protocol targets the T4 evidence environment (`sm_75`).

The runner returns zero when every baseline is clean and the campaign completes.

Mutants that survive are an experimental result, not a process failure.

Use `--resume` only with the same baseline commit and manifest hash.

The runner resumes completed subjects and reruns incomplete or infrastructure-
failed subjects.

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

The deliberately invalid-barrier subjects run synccheck only.

Their memcheck and racecheck cells are `not_applicable`, because instrumenting
an intentional block-barrier violation with a non-synchronization tool can
deadlock rather than produce a meaningful observation.

The report should emphasize unique kills, overlaps, survivors, flaky outcomes,
and known blind spots rather than a single score.
