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


def tolerance_for(expected: float) -> float:
    # Mirror the CUDA-side tolerance: scale with magnitude, floor at 1e-5.
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
    if not isinstance(values, list) or not values:
        raise ValueError("artifact input.values must be a non-empty list")
    if not isinstance(shape, list) or shape != [len(values)]:
        raise ValueError("artifact input.shape must equal [len(values)]")
    if "block_threads" not in data["launch"]:
        raise ValueError("artifact launch.block_threads is required")

    kernel = data.get("kernel")
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
    return data


def build_payload(artifact: dict) -> str:
    values = artifact["input"]["values"]
    n = len(values)
    block_threads = int(artifact["launch"]["block_threads"])

    if artifact.get("kernel") == "vector_add":
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

    return (
        f"{artifact['implementation']}\n"
        f"{n}\n"
        f"{block_threads}\n"
        + " ".join(repr(float(v)) for v in values)
        + "\n"
    )


def compute_mismatch(expected: float, actual: float) -> bool:
    return abs(actual - expected) > tolerance_for(expected)


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
        replayed_expected = float(result["expected"])
        replayed_actual = float(result["actual"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        result = {
            "artifact": str(args.artifact),
            "classification": "infrastructure_failure",
            "stderr": f"replay output unparseable: {exc}",
        }
        print(json.dumps(result, indent=2))
        return 1

    mismatch = compute_mismatch(replayed_expected, replayed_actual)
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
