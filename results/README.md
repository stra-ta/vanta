# Results

Committed result directories preserve raw logs, machine-readable summaries, and the environment that produced them.

`manifest.json` is the repository-level convention file.

It is a schema reminder, not a claim that the empty `artifacts` list contains an experiment.

Every measured artifact must record the commit SHA, dirty-tree state, compiler, kernel, CPU, architecture, build type, seed, and exact command line.

Campaign summaries may point to one shared environment file when each command remains in the campaign ledger.

Raw logs stay immutable after capture.

Generated tables must be regenerated from their JSON source and must not be edited by hand.

GPU results must identify GPU model, compute capability, driver, runtime, and toolkit in the environment file.

Host-only tests are functional evidence.

They are not GPU correctness, sanitizer, or performance evidence.
