#!/usr/bin/env python3
"""Replay a single Vanta failure artifact.

Vanta keeps failure metadata in JSON. The CUDA executable owns execution;
this script owns JSON parsing, validation, and classification. It feeds one
case to `reduction_proof replay` over stdin and classifies the result.

The CUDA replay path only reports the observed result. This script is the
single owner of the tolerance and of the mismatch decision, so we never
trust a `mismatch` field that might have been produced elsewhere.

Usage:
    python3 tools/replay_failure.py failures/reduction-tail-v0.json \
        --executable build/reduction_proof
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REQUIRED_FIELDS = [
    ("schema", str),
    ("kernel", str),
    ("implementation", str),
    ("input", dict),
    ("launch", dict),
    ("observed", dict),
]


def tolerance_for(expected: float, operation: str = "reduce_sum",
                  k: int = 1) -> float:
    # Operation-aware tolerance.
    #
    # reduce_sum / vector_add accumulate a handful of float32 terms; a floor of
    # 1e-5 * max(1, |expected|) is effectively exact for small integer sums and
    # stays honest for larger magnitudes.
    #
    # matmul accumulates K products per output element. The worst-case float32
    # rounding over K serial additions is bounded by K * epsilon * |result|.
    # We use a small constant safety margin (3x) rather than a tuned knob, so
    # the correct kernel passes and a structural defect fails by a wide margin.
    if operation == "matmul":
        eps = 2.0 ** -23  # float32 machine epsilon
        return max(1.0, abs(expected)) * k * 3.0 * eps
    return 1e-5 * max(1.0, abs(expected))


def load_artifact(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("artifact top level must be a JSON object")
    for field, expected in REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"artifact missing field: {field}")
        if not isinstance(data[field], expected):
            raise ValueError(f"artifact field {field} has wrong type")
    values = data["input"].get("values")
    shape = data["input"].get("shape")
    if "block_threads" not in data["launch"]:
        raise ValueError("artifact launch.block_threads is required")

    kernel = data.get("kernel")
    if kernel in ("reduce_sum", "vector_add"):
        if not isinstance(values, list) or not values:
            raise ValueError("artifact input.values must be a non-empty list")
        if not isinstance(shape, list) or shape != [len(values)]:
            raise ValueError("artifact input.shape must equal [len(values)]")
    if kernel == "vector_add":
        params = data.get("params")
        aux = data.get("aux")
        if not isinstance(params, dict) or not isinstance(params.get("a"),
                                                          (int, float)):
            raise ValueError("vector_add artifact requires params.a (number)")
        if not isinstance(aux, dict) or not isinstance(aux.get("y"), list):
            raise ValueError("vector_add artifact requires aux.y (list)")
        if len(aux["y"]) != len(values):
            raise ValueError("vector_add aux.y length must equal input length")
    elif kernel == "matmul":
        a = data["input"].get("a")
        b = data["input"].get("b")
        meta = data.get("meta")
        if not isinstance(a, list) or not isinstance(b, list):
            raise ValueError("matmul artifact requires input.a and input.b")
        if not isinstance(meta, dict) or not isinstance(meta.get("n"), int):
            raise ValueError("matmul artifact requires meta.n (int)")
        n = meta["n"]
        if shape != [n, n, n]:
            raise ValueError("matmul input.shape must equal [n, n, n]")
        if len(a) != n * n or len(b) != n * n:
            raise ValueError("matmul input.a / input.b must have n*n elements")
        if "k" not in meta:
            meta["k"] = n
    return data


def reference_matmul(a: list[float], b: list[float], n: int) -> list[float]:
    """CPU double-precision reference for C = A · B (row-major, n x n)."""
    c = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += float(a[i * n + k]) * float(b[k * n + j])
            c[i * n + j] = total
    return c


def build_payload(artifact: dict) -> str:
    block_threads = int(artifact["launch"]["block_threads"])

    if artifact.get("kernel") == "vector_add":
        n = len(artifact["input"]["values"])
        values = artifact["input"]["values"]
        a = float(artifact["params"]["a"])
        y = artifact["aux"]["y"]
        return (
            f"{artifact['implementation']}\n"
            f"{n}\n"
            f"{block_threads}\n"
            f"{a!r}\n"
            + " ".join(repr(float(v)) for v in values)
            + "\n"
            + " ".join(repr(float(v)) for v in y)
            + "\n"
        )

    if artifact.get("kernel") == "matmul":
        a = artifact["input"]["a"]
        b = artifact["input"]["b"]
        n = int(artifact["meta"]["n"])
        return (
            f"{artifact['implementation']}\n"
            f"{n}\n"
            f"{block_threads}\n"
            + " ".join(repr(float(v)) for v in a)
            + "\n"
            + " ".join(repr(float(v)) for v in b)
            + "\n"
        )

    values = artifact["input"]["values"]
    n = len(values)
    return (
        f"{artifact['implementation']}\n"
        f"{n}\n"
        f"{block_threads}\n"
        + " ".join(repr(float(v)) for v in values)
        + "\n"
    )


def compute_mismatch(expected: float, actual: float,
                     operation: str = "reduce_sum", k: int = 1) -> bool:
    return abs(actual - expected) > tolerance_for(expected, operation, k)


def classify(artifact: dict, result: dict) -> str:
    recorded_expected = float(artifact["observed"]["expected"])
    recorded_actual = float(artifact["observed"]["actual"])
    replayed_expected = float(result["expected"])
    replayed_actual = float(result["actual"])

    if not math.isclose(replayed_expected, recorded_expected, abs_tol=1e-6):
        return "different_failure"

    mismatch = compute_mismatch(replayed_expected, replayed_actual)
    if not mismatch:
        return "resolved"
    if math.isclose(replayed_actual, recorded_actual, abs_tol=1e-6):
        return "same_failure"
    return "different_failure"


def classify_matmul(artifact: dict, result: dict) -> tuple[str, bool, dict]:
    """Return (classification, mismatch, details) for a matmul result.

    The CUDA replay emits only the actual output matrix; Python owns the
    reference computation and the operation-aware tolerance. The artifact
    records the reference scale (observed.expected) and the maximal per-element
    error of the captured faulty run (observed.actual), so replay can confirm
    it reproduces the same defect rather than a different one.
    """
    n = int(artifact["meta"]["n"])
    k = int(artifact["meta"].get("k", n))
    a = artifact["input"]["a"]
    b = artifact["input"]["b"]
    ref = reference_matmul(a, b, n)
    scale = max((abs(value) for value in ref), default=0.0)

    actual = result["actual"]
    if len(actual) != n * n:
        raise ValueError("matmul replay returned wrong number of elements")
    max_error = max(
        abs(float(actual[i]) - ref[i]) for i in range(n * n))

    recorded_scale = float(artifact["observed"]["expected"])
    recorded_error = float(artifact["observed"]["actual"])
    mismatch = max_error > tolerance_for(scale, "matmul", k)

    details = {
        "recorded_actual": recorded_error,
        "replayed_max_error": max_error,
        "reference_scale": scale,
        "recorded_scale": recorded_scale,
        "n": n,
    }

    if not mismatch:
        return "resolved", False, details
    if math.isclose(max_error, recorded_error, rel_tol=1e-2, abs_tol=1e-3):
        return "same_failure", True, details
    return "different_failure", True, details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build/reduction_proof"),
        help="path to the compiled reduction_proof binary",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable result instead of human text",
    )
    args = parser.parse_args(argv)

    if not args.artifact.exists():
        print(f"artifact not found: {args.artifact}", file=sys.stderr)
        return 2
    if not args.executable.exists():
        print(f"executable not found: {args.executable}", file=sys.stderr)
        return 2

    artifact = load_artifact(args.artifact)
    payload = build_payload(artifact)

    process = subprocess.run(
        [str(args.executable), "replay"],
        input=payload,
        text=True,
        capture_output=True,
    )

    if process.returncode != 0:
        result = {
            "artifact": str(args.artifact),
            "classification": "infrastructure_failure",
            "stderr": process.stderr.strip(),
        }
        print(json.dumps(result, indent=2))
        return 1

    try:
        result = json.loads(process.stdout.strip())
        if artifact.get("kernel") == "matmul":
            classification, mismatch, details = classify_matmul(artifact, result)
            output = {
                "artifact": str(args.artifact),
                "classification": classification,
                "mismatch": mismatch,
                "recorded_actual": details["recorded_actual"],
                "replayed_max_error": details["replayed_max_error"],
                "reference_scale": details["reference_scale"],
                "block_threads": int(artifact["launch"]["block_threads"]),
                "n": details["n"],
            }
        else:
            replayed_expected = float(result["expected"])
            replayed_actual = float(result["actual"])
            mismatch = compute_mismatch(
                replayed_expected, replayed_actual,
                artifact.get("kernel", "reduce_sum"),
                int(artifact.get("meta", {}).get("k", 1)))
            classification = classify(artifact, result)
            output = {
                "artifact": str(args.artifact),
                "classification": classification,
                "mismatch": mismatch,
                "recorded_actual": float(artifact["observed"]["actual"]),
                "replayed_actual": replayed_actual,
                "replayed_expected": replayed_expected,
                "block_threads": int(artifact["launch"]["block_threads"]),
                "n": len(artifact["input"]["values"]),
            }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        result = {
            "artifact": str(args.artifact),
            "classification": "infrastructure_failure",
            "stderr": f"replay output unparseable: {exc}",
        }
        print(json.dumps(result, indent=2))
        return 1

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"classification: {classification}")
        print(f"mismatch: {mismatch}")
        print(f"recorded actual: {output['recorded_actual']}")
        print(f"replayed actual: {output['replayed_actual']}")
        print(f"replayed expected: {output['replayed_expected']}")

    return 0 if classification == "same_failure" else 1


if __name__ == "__main__":
    sys.exit(main())
