#!/usr/bin/env python3
"""Minimize Vanta failures with deterministic structured searches.

The tool runs a replay executable against progressively shorter prefixes of a
failure artifact's input. It keeps the shortest prefix whose replay still
differs from the CPU-reported expected value.

Reduction and vector-add artifacts use a linear prefix search.
Matmul artifacts use a top-left square-dimension search.
Optional value and launch-dimension searches are available for all supported
artifact shapes.

Usage:
    python3 tools/minimize_failure.py \
        failures/reduction-grid-overlap-seed-v0.json \
        --executable /path/to/pre-fix/reduction_sum \
        --out failures/reduction-grid-overlap-min-v0.json

    python3 tools/minimize_failure.py \
        failures/matmul-tile-v0.json \
        --executable /path/to/matmul_proof \
        --out failures/matmul-tile-min-v0.json
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

from replay_failure import (
    build_payload,
    compute_mismatch,
    load_artifact,
    reference_matmul,
    tolerance_for,
)


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


def find_minimal_dimension(
    initial: int,
    candidates: list[int],
    still_fails: Callable[[int], bool],
) -> tuple[int, int]:
    """Find the smallest tested scalar dimension that preserves a failure."""
    if initial < 1:
        raise ValueError("initial dimension must be positive")
    if not still_fails(initial):
        raise ValueError("the seed artifact does not reproduce a mismatch")
    best = initial
    tested = 1
    for candidate in candidates:
        if candidate < 1 or candidate >= initial:
            raise ValueError("dimension candidates must be between 1 and initial-1")
        tested += 1
        if still_fails(candidate):
            best = candidate
    return best, tested


def shrink_values(
    values: list[float],
    still_fails: Callable[[list[float]], bool],
    replacements: tuple[float, ...] = (0.0,),
) -> tuple[list[float], int]:
    """Zero or replace individual values while preserving a failure.

    The search is linear over current elements and replacement values.
    It makes no monotonicity assumption about the failure predicate.
    """
    if not values:
        raise ValueError("cannot shrink an empty input")
    if not replacements:
        raise ValueError("at least one replacement is required")
    if not still_fails(values):
        raise ValueError("the seed artifact does not reproduce a mismatch")
    best = list(values)
    tested = 1
    for index in range(len(best)):
        for replacement in replacements:
            if best[index] == replacement:
                continue
            candidate = list(best)
            candidate[index] = replacement
            tested += 1
            if still_fails(candidate):
                best = candidate
                break
    return best, tested


def matrix_prefix(values: list[float], n: int, new_n: int) -> list[float]:
    """Return the top-left ``new_n`` square from an ``n`` by ``n`` matrix."""
    if n < 1 or new_n < 1 or new_n > n or len(values) != n * n:
        raise ValueError("matrix dimensions do not match the input")
    return [values[row * n + col] for row in range(new_n) for col in range(new_n)]


def minimize_matrix_dimension(
    a: list[float],
    b: list[float],
    n: int,
    still_fails: Callable[[list[float], list[float], int], bool],
    *,
    min_n: int = 1,
) -> tuple[list[float], list[float], int, int]:
    """Shrink both dimensions of a square matrix while preserving failure."""
    if min_n < 1 or min_n > n:
        raise ValueError("min_n must be within the matrix dimension")
    if len(a) != n * n or len(b) != n * n:
        raise ValueError("matrix dimensions do not match the input")
    if not still_fails(a, b, n):
        raise ValueError("the seed artifact does not reproduce a mismatch")
    best_a, best_b, best_n = list(a), list(b), n
    tested = 1
    for candidate_n in range(n - 1, min_n - 1, -1):
        candidate_a = matrix_prefix(a, n, candidate_n)
        candidate_b = matrix_prefix(b, n, candidate_n)
        tested += 1
        if still_fails(candidate_a, candidate_b, candidate_n):
            best_a, best_b, best_n = candidate_a, candidate_b, candidate_n
    return best_a, best_b, best_n, tested


def minimize_launch_dimensions(
    launch: dict[str, int],
    still_fails: Callable[[dict[str, int]], bool],
) -> tuple[dict[str, int], int]:
    """Try smaller block/grid dimensions in deterministic score order."""
    if any(not isinstance(launch.get(key), int) or launch[key] < 1
           for key in ("block_threads", "grid_blocks")):
        raise ValueError("launch requires positive block_threads and grid_blocks")
    if not still_fails(dict(launch)):
        raise ValueError("the seed artifact does not reproduce a mismatch")
    candidates = []
    for block in range(1, launch["block_threads"]):
        candidates.append({"block_threads": block, "grid_blocks": launch["grid_blocks"]})
    for grid in range(1, launch["grid_blocks"]):
        candidates.append({"block_threads": launch["block_threads"], "grid_blocks": grid})
    candidates.sort(key=lambda item: (item["block_threads"] * item["grid_blocks"],
                                      item["block_threads"], item["grid_blocks"]))
    best = dict(launch)
    tested = 1
    for candidate in candidates:
        tested += 1
        candidate_score = candidate["block_threads"] * candidate["grid_blocks"]
        best_score = best["block_threads"] * best["grid_blocks"]
        if candidate_score < best_score and still_fails(candidate):
            best = candidate
    return best, tested


def candidate_artifact(artifact: dict, values: list[float], result: dict) -> dict:
    candidate = copy.deepcopy(artifact)
    candidate["input"]["values"] = values
    candidate["input"]["shape"] = [len(values)]
    if candidate.get("kernel") == "vector_add" and isinstance(candidate.get("aux"), dict):
        candidate["aux"]["y"] = candidate["aux"]["y"][:len(values)]
    candidate["observed"]["expected"] = float(result["expected"])
    candidate["observed"]["actual"] = float(result["actual"])
    return candidate


def candidate_matrix_artifact(
    artifact: dict,
    a: list[float],
    b: list[float],
    n: int,
    result: dict,
) -> dict:
    candidate = copy.deepcopy(artifact)
    candidate["input"]["a"] = a
    candidate["input"]["b"] = b
    candidate["input"]["shape"] = [n, n, n]
    candidate["meta"]["n"] = n
    candidate["meta"]["k"] = n
    reference = reference_matmul(a, b, n)
    candidate["observed"]["expected"] = max((abs(value) for value in reference), default=0.0)
    candidate["observed"]["actual"] = max(
        abs(float(result["actual"][index]) - reference[index])
        for index in range(n * n))
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


def replay_matrix_candidate(
    binary: Path,
    artifact: dict,
    a: list[float],
    b: list[float],
    n: int,
) -> dict:
    candidate = copy.deepcopy(artifact)
    candidate["input"]["a"] = a
    candidate["input"]["b"] = b
    candidate["input"]["shape"] = [n, n, n]
    candidate["meta"]["n"] = n
    candidate["meta"]["k"] = n
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
        actual = result["actual"]
        if not isinstance(actual, list) or len(actual) != n * n:
            raise ValueError("matmul replay returned the wrong output shape")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid replay output: {exc}") from exc
    return result


def replay_launch_candidate(binary: Path, artifact: dict, launch: dict[str, int]) -> dict:
    candidate = copy.deepcopy(artifact)
    candidate["launch"].update(launch)
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
        if not isinstance(result.get("actual"), (int, float, list)):
            raise ValueError("replay output has no actual value")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid replay output: {exc}") from exc
    return result


def matrix_still_fails(result: dict, a: list[float], b: list[float], n: int) -> bool:
    reference = reference_matmul(a, b, n)
    scale = max((abs(value) for value in reference), default=0.0)
    return max(abs(float(result["actual"][index]) - reference[index])
               for index in range(n * n)) > tolerance_for(scale, "matmul", n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--shrink-values", action="store_true",
                        help="try replacing individual input values with zero")
    parser.add_argument("--shrink-launch", action="store_true",
                        help="try smaller recorded block/grid dimensions")
    args = parser.parse_args(argv)

    if not args.executable.exists():
        print(f"executable not found: {args.executable}", file=sys.stderr)
        return 2

    try:
        artifact = load_artifact(args.artifact)
        if artifact["kernel"] in ("reduce_sum", "vector_add"):
            values = [float(value) for value in artifact["input"]["values"]]
            results: list[dict] = []

            def still_fails(candidate: list[float]) -> bool:
                result = replay_candidate(args.executable, artifact, candidate)
                results.append(result)
                return compute_mismatch(float(result["expected"]),
                                        float(result["actual"]),
                                        artifact["kernel"])

            minimized, tested = find_minimal_prefix(
                values, still_fails, min_length=args.min_length)
            if args.shrink_values:
                minimized, shrink_tests = shrink_values(minimized, still_fails)
                tested += shrink_tests
            final_result = replay_candidate(args.executable, artifact, minimized)
            output = candidate_artifact(artifact, minimized, final_result)
            output["minimization"] = {
                "strategy": "linear_prefix" + ("_then_value_shrink" if args.shrink_values else ""),
                "original_length": len(values),
                "minimized_length": len(minimized),
                "tested_cases": tested,
                "seed_artifact": str(args.artifact),
            }
            if args.shrink_launch:
                launch_artifact = copy.deepcopy(output)

                def launch_fails(candidate_launch: dict[str, int]) -> bool:
                    result = replay_launch_candidate(
                        args.executable, launch_artifact, candidate_launch)
                    return compute_mismatch(
                        float(result["expected"]), float(result["actual"]),
                        artifact["kernel"])

                launch, launch_tests = minimize_launch_dimensions(
                    launch_artifact["launch"], launch_fails)
                output["launch"].update(launch)
                output["minimization"]["strategy"] += "_then_launch"
                output["minimization"]["launch_tested_cases"] = launch_tests
        elif artifact["kernel"] == "matmul":
            n = int(artifact["meta"]["n"])
            original_a = [float(value) for value in artifact["input"]["a"]]
            original_b = [float(value) for value in artifact["input"]["b"]]
            matrix_results: list[dict] = []

            def matrix_fails(candidate_a: list[float], candidate_b: list[float], candidate_n: int) -> bool:
                result = replay_matrix_candidate(
                    args.executable, artifact, candidate_a, candidate_b, candidate_n)
                matrix_results.append(result)
                return matrix_still_fails(result, candidate_a, candidate_b, candidate_n)

            minimized_a, minimized_b, minimized_n, tested = minimize_matrix_dimension(
                original_a, original_b, n, matrix_fails, min_n=args.min_length)
            final_result = replay_matrix_candidate(
                args.executable, artifact, minimized_a, minimized_b, minimized_n)
            output = candidate_matrix_artifact(
                artifact, minimized_a, minimized_b, minimized_n, final_result)
            output["minimization"] = {
                "strategy": "linear_matrix_dimension",
                "original_dimension": n,
                "minimized_dimension": minimized_n,
                "tested_cases": tested,
                "seed_artifact": str(args.artifact),
            }
            if args.shrink_values:
                combined = minimized_a + minimized_b

                def combined_matrix_fails(candidate_values: list[float]) -> bool:
                    split = len(minimized_a)
                    candidate_a = candidate_values[:split]
                    candidate_b = candidate_values[split:]
                    result = replay_matrix_candidate(
                        args.executable, output, candidate_a, candidate_b, minimized_n)
                    return matrix_still_fails(
                        result, candidate_a, candidate_b, minimized_n)

                combined, value_tests = shrink_values(combined, combined_matrix_fails)
                minimized_a = combined[:len(minimized_a)]
                minimized_b = combined[len(minimized_a):]
                final_result = replay_matrix_candidate(
                    args.executable, output, minimized_a, minimized_b, minimized_n)
                output = candidate_matrix_artifact(
                    output, minimized_a, minimized_b, minimized_n, final_result)
                output["minimization"] = {
                    **output["minimization"],
                    "strategy": "linear_matrix_dimension_then_value_shrink",
                    "value_tested_cases": value_tests,
                }
            if args.shrink_launch:
                launch_artifact = copy.deepcopy(output)

                def matrix_launch_fails(candidate_launch: dict[str, int]) -> bool:
                    candidate = copy.deepcopy(launch_artifact)
                    candidate["launch"].update(candidate_launch)
                    result = replay_matrix_candidate(
                        args.executable, candidate, minimized_a, minimized_b, minimized_n)
                    return matrix_still_fails(
                        result, minimized_a, minimized_b, minimized_n)

                launch, launch_tests = minimize_launch_dimensions(
                    launch_artifact["launch"], matrix_launch_fails)
                output["launch"].update(launch)
                output["minimization"]["strategy"] += "_then_launch"
                output["minimization"]["launch_tested_cases"] = launch_tests
        else:
            raise ValueError("minimizer supports reduce_sum, vector_add, and matmul artifacts")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n")
        summary = {
            "status": "pass",
            "output": str(args.out),
            "tested_cases": tested,
        }
        if artifact["kernel"] == "matmul":
            summary.update({
                "original_dimension": n,
                "minimized_dimension": minimized_n,
            })
        else:
            summary.update({
                "original_length": len(values),
                "minimized_length": len(minimized),
                "expected": final_result["expected"],
                "actual": final_result["actual"],
            })
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, TypeError,
            json.JSONDecodeError) as exc:
        print(f"minimization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
