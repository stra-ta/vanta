#!/usr/bin/env python3
"""Minimize a one-dimensional reduction failure by prefix search.

The tool runs a replay executable against progressively shorter prefixes of a
failure artifact's input. It keeps the shortest prefix whose replay still
differs from the CPU-reported expected value.

This first minimizer is intentionally narrow. It targets ``reduce_sum``
artifacts with a one-dimensional ``input.values`` array. The first use case is
the historical multi-block overlap bug from revision ``54bad3f``.

Usage:
    python3 tools/minimize_failure.py \
        failures/reduction-grid-overlap-seed-v0.json \
        --executable /path/to/pre-fix/reduction_sum \
        --out failures/reduction-grid-overlap-min-v0.json
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from replay_failure import build_payload, compute_mismatch, load_artifact


def find_minimal_prefix(
    values: list[float],
    still_fails: Callable[[list[float]], bool],
    *,
    min_length: int = 1,
) -> tuple[list[float], int]:
    """Return the shortest tested prefix that still fails.

    The search is linear rather than binary because failure predicates are not
    assumed to be monotonic for arbitrary kernels. The first implementation
    favors a trustworthy result over fewer GPU launches.
    """
    if not values:
        raise ValueError("cannot minimize an empty input")
    if min_length < 1 or min_length > len(values):
        raise ValueError("min_length must be within the input length")
    if not still_fails(values):
        raise ValueError("the seed artifact does not reproduce a mismatch")

    best = list(values)
    tested = 1
    for length in range(len(values) - 1, min_length - 1, -1):
        candidate = values[:length]
        tested += 1
        if still_fails(candidate):
            best = list(candidate)
    return best, tested


def candidate_artifact(artifact: dict, values: list[float], result: dict) -> dict:
    candidate = copy.deepcopy(artifact)
    candidate["input"]["values"] = values
    candidate["input"]["shape"] = [len(values)]
    candidate["observed"]["expected"] = float(result["expected"])
    candidate["observed"]["actual"] = float(result["actual"])
    return candidate


def replay_candidate(binary: Path, artifact: dict, values: list[float]) -> dict:
    candidate = copy.deepcopy(artifact)
    candidate["input"]["values"] = values
    candidate["input"]["shape"] = [len(values)]
    process = subprocess.run(
        [str(binary), "replay"],
        input=build_payload(candidate),
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"replay failed with exit {process.returncode}: "
            f"{process.stderr.strip()}"
        )
    try:
        result = json.loads(process.stdout.strip())
        float(result["expected"])
        float(result["actual"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid replay output: {exc}") from exc
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=1)
    args = parser.parse_args(argv)

    if not args.executable.exists():
        print(f"executable not found: {args.executable}", file=sys.stderr)
        return 2

    try:
        artifact = load_artifact(args.artifact)
        if artifact["kernel"] != "reduce_sum":
            raise ValueError("this minimizer only supports reduce_sum artifacts")
        values = [float(value) for value in artifact["input"]["values"]]
        results: list[dict] = []

        def still_fails(candidate: list[float]) -> bool:
            result = replay_candidate(args.executable, artifact, candidate)
            results.append(result)
            return compute_mismatch(float(result["expected"]),
                                    float(result["actual"]))

        minimized, tested = find_minimal_prefix(
            values, still_fails, min_length=args.min_length)
        final_result = replay_candidate(args.executable, artifact, minimized)
        output = candidate_artifact(artifact, minimized, final_result)
        output["minimization"] = {
            "strategy": "linear_prefix",
            "original_length": len(values),
            "minimized_length": len(minimized),
            "tested_cases": tested,
            "seed_artifact": str(args.artifact),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps({
            "status": "pass",
            "output": str(args.out),
            "original_length": len(values),
            "minimized_length": len(minimized),
            "tested_cases": tested,
            "expected": final_result["expected"],
            "actual": final_result["actual"],
        }, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, TypeError,
            json.JSONDecodeError) as exc:
        print(f"minimization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
