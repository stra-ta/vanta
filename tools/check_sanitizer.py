#!/usr/bin/env python3
"""Check that Compute Sanitizer detects (or cleanly passes) a kernel mode.

Vanta treats the sanitizer verdict as data, not as pass/fail by itself. This
tool runs `compute-sanitizer --tool <tool> --error-exitcode 1 <executable>
<mode>` and decides whether the result matches the expected verdict:

- `--expect-detected` (broken modes): the tool must return nonzero, i.e. it
  found a hazard of the targeted defect class.
- without it (correct mode): the tool must return zero, i.e. clean.

This mirrors `replay_failure.py`: CUDA owns execution, Python owns the
interpretation of the result. It exits 0 only when the observed verdict
matches the expectation, so it can drive CTest directly.

Usage:
    python3 tools/check_sanitizer.py \
        --executable build/sanitizer_proof \
        --mode broken_race --tool racecheck --expect-detected
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["correct", "broken_oob", "broken_sync", "broken_race"],
    )
    parser.add_argument(
        "--tool",
        required=True,
        choices=["memcheck", "racecheck", "synccheck"],
    )
    parser.add_argument(
        "--expect-detected",
        action="store_true",
        help="the sanitizer must report a hazard (nonzero exit)",
    )
    args = parser.parse_args(argv)

    if not args.executable.exists():
        print(json.dumps({
            "executable": str(args.executable),
            "mode": args.mode,
            "tool": args.tool,
            "error": "executable not found",
        }))
        return 2

    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        result = {
            "executable": str(args.executable),
            "mode": args.mode,
            "tool": args.tool,
            "error": "compute-sanitizer not found on PATH",
        }
        print(json.dumps(result))
        return 2

    process = subprocess.run([
        sanitizer, "--tool", args.tool, "--error-exitcode", "1",
        str(args.executable.resolve()), args.mode,
    ], text=True, capture_output=True)

    detected = process.returncode != 0
    ok = detected == args.expect_detected

    report = {
        "executable": str(args.executable),
        "mode": args.mode,
        "tool": args.tool,
        "expect_detected": args.expect_detected,
        "detected": detected,
        "returncode": process.returncode,
        "verdict": "match" if ok else "mismatch",
    }
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
