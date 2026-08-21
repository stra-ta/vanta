"""Decide whether a Compute Sanitizer run produced the expected verdict.

CUDA owns execution. Python owns the verdict. This wraps
`compute-sanitizer --tool X --error-exitcode 1 <exe> <mode>` and decides whether
the outcome matches the expectation:

- a `correct` kernel must stay clean (no sanitizer errors)
- a `broken_*` kernel must be detected (`--expect-detected`)

It emits a single JSON line and exits 0 only on a match, so it drives CTest
directly. Exit 2 means the run could not be attempted (missing sanitizer binary
or missing target executable).

Usage:
    python3 tools/check_sanitizer.py \
        --executable build/sanitizer_proof \
        --mode broken_race --tool racecheck --expect-detected
"""

import argparse
import json
import re
import shutil
import subprocess
import sys


def parse_detected(output: str) -> bool:
    """Return True if Compute Sanitizer reported at least one error.

    Compute Sanitizer prints individual errors as `========= ERROR: ...` (the
    case varies across tools, e.g. `========= Error: Race reported`) and a
    trailing summary whose label depends on the tool: `ERROR SUMMARY`,
    `RACECHECK SUMMARY`, `SYNCCHECK SUMMARY`, or `MEMCHECK SUMMARY`. We match the
    individual error banner case-insensitively and any nonzero tool summary, so
    a crash that returns nonzero without error text does not masquerade as a
    successful catch.
    """
    if re.search(r"=========\s+error\s*:", output, re.IGNORECASE):
        return True
    m = re.search(
        r"(?:ERROR|RACECHECK|SYNCCHECK|MEMCHECK)\s+SUMMARY:\s*(\d+)",
        output, re.IGNORECASE,
    )
    if m and int(m.group(1)) > 0:
        return True
    return False


def run_sanitizer(executable: str, mode: str, tool: str):
    sanitizer = shutil.which("compute-sanitizer")
    if not sanitizer:
        return None, "compute-sanitizer not found on PATH"
    try:
        process = subprocess.run(
            [sanitizer, "--tool", tool, "--error-exitcode", "1",
             executable, mode],
            capture_output=True, text=True,
        )
    except OSError as exc:
        return None, f"failed to launch compute-sanitizer: {exc}"
    combined = process.stdout + process.stderr
    detected = parse_detected(combined)
    return detected, combined.strip(), process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", nargs="?", help="target executable (positional)")
    parser.add_argument("--executable", dest="exe_opt", help="target executable")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--tool", required=True,
                        choices=["memcheck", "racecheck", "synccheck"])
    parser.add_argument("--expect-detected", action="store_true",
                        help="the kernel under test is deliberately broken")
    args = parser.parse_args(argv)

    exe = args.exe or args.exe_opt
    if not exe:
        print(json.dumps({"ok": False,
                          "error": "missing executable argument"}))
        return 2
    import os
    if not os.path.exists(exe):
        print(json.dumps({"ok": False, "executable": exe,
                          "error": "executable not found"}))
        return 2

    result = run_sanitizer(exe, args.mode, args.tool)
    if result[0] is None:
        print(json.dumps({"ok": False, "executable": exe,
                          "mode": args.mode, "tool": args.tool,
                          "error": result[1]}))
        return 2

    detected, combined, returncode = result
    expect = args.expect_detected
    # A crash (nonzero exit, no reported error) is neither a clean run nor a
    # real detection, so it must never green a `correct` baseline or count as a
    # broken-mode catch.
    if expect:
        match = detected
    else:
        match = (returncode == 0 and not detected)
    report = {
        "ok": True,
        "executable": exe,
        "mode": args.mode,
        "tool": args.tool,
        "expected_detected": expect,
        "detected": detected,
        "returncode": returncode,
        "verdict": "match" if match else "mismatch",
    }
    print(json.dumps(report))
    if not match:
        sys.stderr.write(combined + "\n")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
